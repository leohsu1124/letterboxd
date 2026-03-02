import pandas as pd
import numpy as np
import scipy.sparse as sp
import json, os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Configs
TFIDF_MAX_FEATURES = 5000        # cap vocabulary size for overview TF-IDF
TFIDF_WEIGHT       = 0.5         # scale TF-IDF block relative to binary features
FEATURE_CACHE_PATH = "data/feature_matrix.npz"
FEATURE_META_PATH  = "data/feature_meta.json"


# --- Loaders ---

def load_merged(candidates_path: str = "data/candidates.parquet",
                enriched_path:   str = "data/tmdb_enriched.parquet") -> pd.DataFrame:
    """
    Merges candidates with enriched features on tmdb_id.
    Returns one row per movie with all columns needed for featurization.
    """
    candidates_df = pd.read_parquet(candidates_path)
    enriched_df   = pd.read_parquet(enriched_path)

    df = candidates_df.merge(enriched_df, on="tmdb_id", how="left")

    # parse pipe-delimited strings back to lists
    df["cast_list"]     = df["cast_list"].fillna("").apply(lambda x: [v for v in x.split("|") if v])
    df["keywords_list"] = df["keywords_list"].fillna("").apply(lambda x: [v for v in x.split("|") if v])
    df["director"]      = df["director"].fillna("")

    # genre_ids column comes in as a string repr of a list — parse it
    df["genre_list"] = df["genre_ids"].fillna("[]").apply(_parse_genre_ids)

    return df


# TMDB genre ID -> name mapping
GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance",
    878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}


def _parse_genre_ids(val) -> list[str]:
    """Handles genre_ids as a list of ints, or a string repr of same."""
    if isinstance(val, list):
        return [GENRE_MAP.get(int(v), str(v)) for v in val if v is not None]
    try:
        parsed = json.loads(str(val).replace("'", '"'))
        return [GENRE_MAP.get(int(v), str(v)) for v in parsed if v is not None]
    except Exception:
        return []


# --- Feature builders ---

def _build_binary_block(df: pd.DataFrame, col: str,
                        mlb: MultiLabelBinarizer | None = None,
                        prefix: str = "") -> tuple[sp.csr_matrix, list[str], MultiLabelBinarizer]:
    """
    One-hot encodes a list-valued column using MultiLabelBinarizer.
    Returns (sparse_matrix, feature_names, fitted_mlb).
    """
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True)
        matrix = mlb.fit_transform(df[col])
    else:
        matrix = mlb.transform(df[col])

    feature_names = [f"{prefix}{c}" for c in mlb.classes_]
    return matrix.astype(np.float32), feature_names, mlb


def _build_director_block(df: pd.DataFrame,
                          known_directors: list[str] | None = None
                          ) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """
    Binary column per director (single-label, so just a one-hot).
    Returns (sparse_matrix, feature_names, director_vocab).
    """
    if known_directors is None:
        known_directors = sorted(df["director"].unique().tolist())
        known_directors = [d for d in known_directors if d]  # drop empty

    dir_to_idx = {d: i for i, d in enumerate(known_directors)}
    n = len(df)
    k = len(known_directors)

    rows, cols = [], []
    for i, director in enumerate(df["director"]):
        if director in dir_to_idx:
            rows.append(i)
            cols.append(dir_to_idx[director])

    matrix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, k)
    )
    feature_names = [f"director:{d}" for d in known_directors]
    return matrix, feature_names, known_directors


def _build_tfidf_block(df: pd.DataFrame,
                       vectorizer: TfidfVectorizer | None = None
                       ) -> tuple[sp.csr_matrix, list[str], TfidfVectorizer]:
    """
    TF-IDF on the overview column, weighted down relative to binary features.
    Returns (sparse_matrix, feature_names, fitted_vectorizer).
    """
    overviews = df["overview"].fillna("").tolist()

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words="english",
            sublinear_tf=True,         # log(1 + tf) dampens very frequent terms
            min_df=3,                  # ignore terms appearing in < 3 docs
        )
        matrix = vectorizer.fit_transform(overviews)
    else:
        matrix = vectorizer.transform(overviews)

    matrix = matrix.astype(np.float32) * TFIDF_WEIGHT
    feature_names = [f"tfidf:{t}" for t in vectorizer.get_feature_names_out()]
    return matrix, feature_names, vectorizer


# --- Main builder ---

def build_feature_matrix(df: pd.DataFrame) -> tuple[sp.csr_matrix, list[str], dict]:
    """
    Builds the full feature matrix by horizontally stacking:
      [genres | cast | director | keywords | tfidf_overview]

    Returns:
      matrix       : (n_movies, n_features) sparse float32
      feature_names: list of feature name strings
      encoders     : dict of fitted encoders for reuse on liked set
    """
    print("[features] Building feature blocks...")

    genre_mat,    genre_names,    genre_mlb    = _build_binary_block(df, "genre_list",     prefix="genre:")
    cast_mat,     cast_names,     cast_mlb     = _build_binary_block(df, "cast_list",      prefix="cast:")
    kw_mat,       kw_names,       kw_mlb       = _build_binary_block(df, "keywords_list",  prefix="kw:")
    dir_mat,      dir_names,      dir_vocab    = _build_director_block(df)
    tfidf_mat,    tfidf_names,    tfidf_vec    = _build_tfidf_block(df)

    matrix = sp.hstack([genre_mat, cast_mat, dir_mat, kw_mat, tfidf_mat], format="csr")
    feature_names = genre_names + cast_names + dir_names + kw_names + tfidf_names

    encoders = {
        "genre_mlb":   genre_mlb,
        "cast_mlb":    cast_mlb,
        "kw_mlb":      kw_mlb,
        "dir_vocab":   dir_vocab,
        "tfidf_vec":   tfidf_vec,
    }

    print(f"[features] Matrix shape: {matrix.shape} | "
          f"genres={len(genre_names)} cast={len(cast_names)} "
          f"directors={len(dir_names)} keywords={len(kw_names)} tfidf={len(tfidf_names)}")

    return matrix, feature_names, encoders


def encode_liked(liked_df: pd.DataFrame, encoders: dict) -> sp.csr_matrix:
    """
    Applies fitted encoders to the liked set so vectors are in the same space
    as the candidate matrix. Call after build_feature_matrix.
    """
    liked_df = liked_df.copy()
    liked_df["cast_list"]     = liked_df["cast_list"].fillna("").apply(lambda x: [v for v in x.split("|") if v])
    liked_df["keywords_list"] = liked_df["keywords_list"].fillna("").apply(lambda x: [v for v in x.split("|") if v])
    liked_df["director"]      = liked_df["director"].fillna("")
    liked_df["genre_list"]    = liked_df["genre_ids"].fillna("[]").apply(_parse_genre_ids)

    genre_mat,  _, _ = _build_binary_block(liked_df, "genre_list",    mlb=encoders["genre_mlb"], prefix="genre:")
    cast_mat,   _, _ = _build_binary_block(liked_df, "cast_list",     mlb=encoders["cast_mlb"],  prefix="cast:")
    kw_mat,     _, _ = _build_binary_block(liked_df, "keywords_list", mlb=encoders["kw_mlb"],    prefix="kw:")
    dir_mat,    _, _ = _build_director_block(liked_df, known_directors=encoders["dir_vocab"])
    tfidf_mat,  _, _ = _build_tfidf_block(liked_df, vectorizer=encoders["tfidf_vec"])

    return sp.hstack([genre_mat, cast_mat, dir_mat, kw_mat, tfidf_mat], format="csr")


# --- Persistence ---

def save_features(matrix: sp.csr_matrix, feature_names: list[str],
                  tmdb_ids: list[int]) -> None:
    sp.save_npz(FEATURE_CACHE_PATH, matrix)
    meta = {"feature_names": feature_names, "tmdb_ids": tmdb_ids}
    with open(FEATURE_META_PATH, "w") as f:
        json.dump(meta, f)
    print(f"[features] Saved matrix -> {FEATURE_CACHE_PATH}")
    print(f"[features] Saved meta   -> {FEATURE_META_PATH}")


def load_features() -> tuple[sp.csr_matrix, list[str], list[int]]:
    matrix = sp.load_npz(FEATURE_CACHE_PATH)
    with open(FEATURE_META_PATH, "r") as f:
        meta = json.load(f)
    return matrix, meta["feature_names"], meta["tmdb_ids"]


if __name__ == "__main__":
    rated_df    = pd.read_csv("data/letterboxd_ratings_mapped.csv")
    enriched_df = pd.read_parquet("data/tmdb_enriched.parquet")

    # --- candidate matrix ---
    df = load_merged()
    matrix, feature_names, encoders = build_feature_matrix(df)
    save_features(matrix, feature_names, df["tmdb_id"].tolist())

    # --- liked set matrix (same feature space) ---
    liked_ids = (
        rated_df
        .loc[rated_df["Rating"] >= 4.0, "tmdb_id"]
        .dropna()
        .astype(int)
        .tolist()
    )
    liked_enriched = enriched_df[enriched_df["tmdb_id"].isin(liked_ids)].copy()

    # bring in genre_ids + overview from candidates for liked movies
    candidates_df  = pd.read_parquet("data/candidates.parquet")
    liked_meta     = candidates_df[candidates_df["tmdb_id"].isin(liked_ids)][
                        ["tmdb_id", "genre_ids", "overview"]
                     ]
    liked_enriched = liked_enriched.merge(liked_meta, on="tmdb_id", how="left")

    liked_matrix = encode_liked(liked_enriched, encoders)

    sp.save_npz("data/liked_feature_matrix.npz", liked_matrix)
    # save liked_ids in the same row order as the matrix
    with open("data/liked_meta.json", "w") as f:
        json.dump({"tmdb_ids": liked_enriched["tmdb_id"].tolist()}, f)

    print(f"[features] Liked matrix shape: {liked_matrix.shape}")
    print("Done.")