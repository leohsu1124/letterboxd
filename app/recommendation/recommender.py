import numpy as np
import scipy.sparse as sp
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity

# Configs
TOP_N_LIKED        = 20     # how many liked movies to average similarity over (top-N per candidate)
POPULARITY_WEIGHT  = 0.05   # weight of log-popularity prior in final score
MMR_LAMBDA         = 0.7    # MMR tradeoff: 1.0 = pure relevance, 0.0 = pure diversity
MMR_CANDIDATE_POOL = 200    # re-rank top-N by score before applying MMR
GENRE_CAP          = 3      # max recommendations allowed per primary genre
GENRE_CAP_POOL     = 200    # how far into the ranked list to look for cap replacements
SUPERHERO_CAP      = 2      # max superhero films regardless of genre

# Keywords that identify superhero content
SUPERHERO_KEYWORDS = {
    "superhero", "marvel comics", "dc comics", "based on comic book",
    "based on comic", "super villain", "supervillain", "superpowers",
    "super powers", "comic book", "marvel cinematic universe",
    "batman", "superman", "spider-man", "wonder woman", "aquaman",
    "the flash", "green lantern", "black panther", "iron man",
    "captain america", "thor", "avengers", "justice league",
    "x-men", "deadpool", "based on dc comics", "based on marvel comics",
}

# TMDB genre ID -> name
GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance",
    878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
}


# --- Loaders ---

def load_candidate_matrix(matrix_path: str = "data/feature_matrix.npz",
                           meta_path:   str = "data/feature_meta.json"
                           ) -> tuple[sp.csr_matrix, list[int]]:
    matrix = sp.load_npz(matrix_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return matrix, meta["tmdb_ids"]


def load_liked_matrix(matrix_path: str = "data/liked_feature_matrix.npz",
                      meta_path:   str = "data/liked_meta.json"
                      ) -> tuple[sp.csr_matrix, list[int]]:
    matrix = sp.load_npz(matrix_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return matrix, meta["tmdb_ids"]


# --- Scoring ---

def compute_similarity_scores(candidate_matrix: sp.csr_matrix,
                               liked_matrix:     sp.csr_matrix) -> np.ndarray:
    """
    For each candidate, computes mean cosine similarity over the top-N most
    similar liked movies (rather than all liked movies, to avoid noise from
    tangentially liked movies dragging down a strong match).

    Returns:
      sim_scores: (n_candidates,) float array
    """
    # cosine_similarity returns (n_candidates, n_liked)
    sims = cosine_similarity(candidate_matrix, liked_matrix)

    if sims.shape[1] <= TOP_N_LIKED:
        return sims.mean(axis=1)

    # mean over top-N liked per candidate
    top_n = np.partition(sims, -TOP_N_LIKED, axis=1)[:, -TOP_N_LIKED:]
    return top_n.mean(axis=1)


def compute_popularity_prior(candidates_df: pd.DataFrame) -> np.ndarray:
    """
    Log-scaled popularity prior, normalized to [0, 1].
    Gives a gentle nudge toward more well-known films without dominating.
    """
    pop = candidates_df["popularity"].fillna(0).values.astype(float)
    log_pop = np.log1p(pop)
    max_val = log_pop.max()
    return log_pop / max_val if max_val > 0 else log_pop


def compute_final_scores(sim_scores:    np.ndarray,
                          pop_prior:    np.ndarray) -> np.ndarray:
    """
    Combines similarity + popularity prior into a single score.
    Similarity is the dominant signal; popularity is a tiebreaker.
    """
    return (1.0 - POPULARITY_WEIGHT) * sim_scores + POPULARITY_WEIGHT * pop_prior


def mmr_rerank(candidate_matrix: sp.csr_matrix,
               scores:           np.ndarray,
               pool_indices:     np.ndarray,
               k:                int) -> list[int]:
    """
    Maximal Marginal Relevance reranking to add diversity.
    Picks items that are relevant but not too similar to already-selected items.

    Returns:
      selected: list of indices (into pool_indices) of the top-k diverse results
    """
    pool_matrix = candidate_matrix[pool_indices]
    pool_scores = scores[pool_indices]

    selected = []
    remaining = list(range(len(pool_indices)))

    for _ in range(k):
        if not remaining:
            break

        if not selected:
            # first pick: highest score
            best = max(remaining, key=lambda i: pool_scores[i])
        else:
            # compute similarity to already-selected set
            selected_matrix = pool_matrix[selected]
            sims_to_selected = cosine_similarity(
                pool_matrix[remaining], selected_matrix
            ).max(axis=1)  # max similarity to any already-selected item

            # MMR score: lambda * relevance - (1 - lambda) * redundancy
            mmr_scores = np.array([
                MMR_LAMBDA * pool_scores[i] - (1 - MMR_LAMBDA) * sims_to_selected[j]
                for j, i in enumerate(remaining)
            ])
            best = remaining[int(np.argmax(mmr_scores))]

        selected.append(best)
        remaining.remove(best)

    return [pool_indices[i] for i in selected]


# --- Genre cap ---

def _primary_genre(genre_ids) -> str:
    """Returns the first decoded genre name for a movie, or 'Unknown'."""
    if genre_ids is None:
        return "Unknown"
    ids = list(genre_ids) if not isinstance(genre_ids, list) else genre_ids
    for gid in ids:
        name = GENRE_MAP.get(int(gid))
        if name:
            return name
    return "Unknown"


def _is_superhero(keywords_list) -> bool:
    """Returns True if the movie's keywords overlap with SUPERHERO_KEYWORDS."""
    if not isinstance(keywords_list, str) or not keywords_list:
        return False
    kws = {kw.strip().lower() for kw in keywords_list.split("|")}
    return bool(kws & SUPERHERO_KEYWORDS)


def genre_cap_filter(ranked_df: pd.DataFrame, k: int,
                     cap: int = GENRE_CAP,
                     superhero_cap: int = SUPERHERO_CAP,
                     pool_size: int = GENRE_CAP_POOL) -> pd.DataFrame:
    """
    Post-MMR filter enforcing two caps simultaneously:
      1. No primary genre appears more than `cap` times.
      2. Superhero films (by keyword) appear no more than `superhero_cap` times.

    Pulls replacements greedily from the top pool_size candidates by score.
    ranked_df must already be sorted by final_score descending.
    """
    pool = ranked_df.head(pool_size).copy()
    pool["primary_genre"]  = pool["genre_ids"].apply(_primary_genre)
    pool["is_superhero"]   = pool["keywords_list"].apply(_is_superhero)

    selected = []
    genre_counts:     dict[str, int] = {}
    superhero_count:  int            = 0

    for _, row in pool.iterrows():
        if len(selected) >= k:
            break

        pg          = row["primary_genre"]
        is_hero     = row["is_superhero"]

        genre_ok    = genre_counts.get(pg, 0) < cap
        hero_ok     = (not is_hero) or (superhero_count < superhero_cap)

        if genre_ok and hero_ok:
            selected.append(row)
            genre_counts[pg] = genre_counts.get(pg, 0) + 1
            if is_hero:
                superhero_count += 1

    return pd.DataFrame(selected).reset_index(drop=True)


# --- Main recommender ---

def recommend(k: int = 10,
              candidates_path:   str = "data/candidates.parquet",
              enriched_path:     str = "data/tmdb_enriched.parquet",
              matrix_path:       str = "data/feature_matrix.npz",
              meta_path:         str = "data/feature_meta.json",
              liked_matrix_path: str = "data/liked_feature_matrix.npz",
              liked_meta_path:   str = "data/liked_meta.json",
              ) -> pd.DataFrame:
    """
    Returns a DataFrame of the top-k recommended movies with scores and
    enough metadata for the explainer step.

    Columns:
      tmdb_id | title | release_date | poster_path | overview |
      sim_score | final_score | genre_ids | keywords_list |
      seed_count | rec_hits | sim_hits
    """
    MIN_VOTE_COUNT = 300   # filter low-quality candidates

    # load everything
    candidate_matrix, candidate_ids = load_candidate_matrix(matrix_path, meta_path)
    liked_matrix,     _             = load_liked_matrix(liked_matrix_path, liked_meta_path)
    candidates_df = pd.read_parquet(candidates_path)
    enriched_df   = pd.read_parquet(enriched_path)[["tmdb_id", "keywords_list", "cast_list", "director"]]

    # bring in keywords, cast, director for explainer and superhero detection
    candidates_df = candidates_df.merge(enriched_df, on="tmdb_id", how="left")
    candidates_df["keywords_list"] = candidates_df["keywords_list"].fillna("")
    candidates_df["cast_list"]     = candidates_df["cast_list"].fillna("")
    candidates_df["director"]      = candidates_df["director"].fillna("")

    # align candidates_df rows to match matrix row order
    id_to_idx = {tid: i for i, tid in enumerate(candidate_ids)}
    candidates_df = candidates_df[candidates_df["tmdb_id"].isin(id_to_idx)].copy()
    candidates_df["matrix_idx"] = candidates_df["tmdb_id"].map(id_to_idx)
    candidates_df = candidates_df.sort_values("matrix_idx").reset_index(drop=True)

    # filter low-quality candidates before scoring
    before = len(candidates_df)
    candidates_df = candidates_df[candidates_df["vote_count"] >= MIN_VOTE_COUNT].copy()
    print(f"[recommend] Vote filter: {before} -> {len(candidates_df)} candidates (>= {MIN_VOTE_COUNT} votes)")

    # slice matrix to surviving candidates only
    surviving_indices = candidates_df["matrix_idx"].values
    candidate_matrix  = candidate_matrix[surviving_indices]

    # reset matrix_idx to new contiguous positions
    candidates_df = candidates_df.reset_index(drop=True)
    candidates_df["matrix_idx"] = range(len(candidates_df))

    # score
    sim_scores  = compute_similarity_scores(candidate_matrix, liked_matrix)
    pop_prior   = compute_popularity_prior(candidates_df)
    final_scores = compute_final_scores(sim_scores, pop_prior)

    candidates_df["sim_score"]   = sim_scores
    candidates_df["final_score"] = final_scores

    # MMR rerank over top pool
    pool_size    = min(MMR_CANDIDATE_POOL, len(candidates_df))
    pool_indices = np.argsort(final_scores)[::-1][:pool_size]
    top_indices  = mmr_rerank(candidate_matrix, final_scores, pool_indices, k=k)

    results = candidates_df.iloc[top_indices].copy()
    results = results.sort_values("final_score", ascending=False).reset_index(drop=True)

    # genre cap: pull replacements from a wider pool if needed
    wider_pool   = candidates_df.iloc[np.argsort(final_scores)[::-1][:MMR_CANDIDATE_POOL]].copy()
    wider_pool   = wider_pool.sort_values("final_score", ascending=False).reset_index(drop=True)
    results      = genre_cap_filter(wider_pool, k=k)

    return results[[
        "tmdb_id", "title", "release_date", "poster_path", "overview",
        "sim_score", "final_score", "genre_ids", "keywords_list",
        "cast_list", "director", "seed_count", "rec_hits", "sim_hits",
    ]].reset_index(drop=True)


if __name__ == "__main__":
    results = recommend(k=10)

    print(f"\nTop-10 Recommendations:\n{'=' * 50}")
    for _, row in results.iterrows():
        year     = str(row["release_date"])[:4] if pd.notna(row["release_date"]) else "?"
        hero_tag = " [SUPERHERO]" if _is_superhero(row["keywords_list"]) else ""
        print(f"  {row['title']} ({year}){hero_tag}")
        print(f"    sim={row['sim_score']:.4f}  final={row['final_score']:.4f}")
        print(f"    primary genre: {_primary_genre(row['genre_ids'])}  |  all: {row['genre_ids']}")
        print()