import pandas as pd
import json
from dataclasses import dataclass

# Configs
TOP_SEED_MOVIES    = 3    # max liked movies to mention in explanation
TOP_SHARED_GENRES  = 2    # max genres to mention
TOP_SHARED_CAST    = 2    # max shared cast members to mention
TOP_SHARED_KW      = 2    # max shared keywords to mention

# TMDB genre ID -> name (keep in sync with recommender.py)
GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance",
    878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
}


@dataclass
class Explanation:
    seed_titles:    list[str]   # liked movies that drove this rec
    shared_genres:  list[str]   # genre names in common
    shared_cast:    list[str]   # cast members in common
    shared_director: str | None # director match if any
    shared_keywords: list[str]  # keywords in common
    summary:        str         # human-readable one-liner


# --- Loaders ---

def load_enriched(enriched_path: str = "data/tmdb_enriched.parquet") -> pd.DataFrame:
    return pd.read_parquet(enriched_path)


def load_liked_enriched(rated_path:   str = "data/letterboxd_ratings_mapped.csv",
                        enriched_path: str = "data/tmdb_enriched.parquet",
                        candidates_path: str = "data/candidates.parquet",
                        like_threshold: float = 4.0) -> pd.DataFrame:
    """
    Returns enriched metadata for all liked movies, including genre_ids and title.
    Pulls genre_ids + overview from candidates where available, falls back to rated_df name.
    """
    rated_df      = pd.read_csv(rated_path)
    enriched_df   = pd.read_parquet(enriched_path)
    candidates_df = pd.read_parquet(candidates_path)[["tmdb_id", "genre_ids", "title"]]

    liked_ids = (
        rated_df
        .loc[rated_df["Rating"] >= like_threshold, ["tmdb_id", "Name", "Rating"]]
        .dropna(subset=["tmdb_id"])
        .copy()
    )
    liked_ids["tmdb_id"] = liked_ids["tmdb_id"].astype(int)

    # merge enriched features
    df = liked_ids.merge(enriched_df, on="tmdb_id", how="left")

    # fill title from candidates if available, fall back to letterboxd Name
    df = df.merge(candidates_df.rename(columns={"title": "tmdb_title"}),
                  on="tmdb_id", how="left")
    df["title"] = df["tmdb_title"].fillna(df["Name"])

    # parse pipe-delimited lists
    df["cast_list"]     = df["cast_list"].fillna("").apply(lambda x: [v for v in x.split("|") if v])
    df["keywords_list"] = df["keywords_list"].fillna("").apply(lambda x: [v for v in x.split("|") if v])
    df["director"]      = df["director"].fillna("")
    df["genre_ids"]     = df["genre_ids"].apply(_parse_genre_ids)

    return df


def _parse_genre_ids(val) -> list[str]:
    if isinstance(val, list):
        return [GENRE_MAP.get(int(v), str(v)) for v in val if v is not None]
    try:
        import json
        parsed = json.loads(str(val).replace("'", '"'))
        return [GENRE_MAP.get(int(v), str(v)) for v in parsed if v is not None]
    except Exception:
        return []


# --- Explanation builder ---

def _parse_candidate_features(row: pd.Series) -> dict:
    """Extracts features from a recommendation row."""
    cast = [v for v in str(row.get("cast_list", "")).split("|") if v]
    kws  = [v for v in str(row.get("keywords_list", "")).split("|") if v]
    genres = _parse_genre_ids(row.get("genre_ids", []))
    director = str(row.get("director", "")) if pd.notna(row.get("director")) else ""
    return {"cast": cast, "keywords": kws, "genres": genres, "director": director}


def explain_recommendation(candidate_row: pd.Series,
                            liked_df:      pd.DataFrame,
                            liked_meta_path: str = "data/liked_meta.json") -> Explanation:
    """
    Generates an Explanation for a single recommended movie by comparing
    its features against the full liked set.

    Strategy:
      1. Find which liked movies share the most features (seed movies)
      2. Collect shared genres, cast, director, keywords across all liked movies
      3. Build a natural-language summary
    """
    cand = _parse_candidate_features(candidate_row)

    cand_genres   = set(cand["genres"])
    cand_cast     = set(cand["cast"])
    cand_kws      = set(kw.lower() for kw in cand["keywords"])
    cand_director = cand["director"]

    # score each liked movie by feature overlap with this candidate
    liked_scores = []
    for _, liked_row in liked_df.iterrows():
        liked_genres = set(liked_row["genre_ids"])
        liked_cast   = set(liked_row["cast_list"])
        liked_kws    = set(kw.lower() for kw in liked_row["keywords_list"])
        liked_dir    = liked_row["director"]

        overlap = (
            2.0 * len(cand_genres & liked_genres)
            + 3.0 * len(cand_cast & liked_cast)
            + 5.0 * (1 if cand_director and cand_director == liked_dir else 0)
            + 1.0 * len(cand_kws & liked_kws)
        )
        liked_scores.append((overlap, liked_row))

    liked_scores.sort(key=lambda x: x[0], reverse=True)
    top_seeds = [row for score, row in liked_scores[:TOP_SEED_MOVIES] if score > 0]

    # aggregate shared features across all liked movies (not just top seeds)
    all_liked_genres   = set(g for _, r in liked_df.iterrows() for g in r["genre_ids"])
    all_liked_cast     = set(c for _, r in liked_df.iterrows() for c in r["cast_list"])
    all_liked_kws      = set(kw.lower() for _, r in liked_df.iterrows() for kw in r["keywords_list"])
    all_liked_directors = set(r["director"] for _, r in liked_df.iterrows() if r["director"])

    shared_genres   = list(cand_genres & all_liked_genres)[:TOP_SHARED_GENRES]
    shared_cast     = list(cand_cast & all_liked_cast)[:TOP_SHARED_CAST]
    shared_keywords = list(cand_kws & all_liked_kws)[:TOP_SHARED_KW]
    shared_director = cand_director if cand_director in all_liked_directors else None

    seed_titles = [r["title"] for r in top_seeds]
    summary     = _build_summary(seed_titles, shared_genres, shared_cast,
                                  shared_director, shared_keywords)

    return Explanation(
        seed_titles=seed_titles,
        shared_genres=shared_genres,
        shared_cast=shared_cast,
        shared_director=shared_director,
        shared_keywords=shared_keywords,
        summary=summary,
    )


def _build_summary(seed_titles:    list[str],
                   shared_genres:  list[str],
                   shared_cast:    list[str],
                   shared_director: str | None,
                   shared_keywords: list[str]) -> str:
    """
    Builds a natural one-liner explanation in the style:
      'Because you liked X / shares genre Y / directed by Z / features W'
    """
    parts = []

    if seed_titles:
        titles_str = " and ".join(f'"{t}"' for t in seed_titles[:2])
        parts.append(f"Because you liked {titles_str}")

    if shared_director:
        parts.append(f"directed by {shared_director}")

    if shared_genres:
        parts.append(f"shares {' & '.join(shared_genres)}")

    if shared_cast:
        parts.append(f"features {', '.join(shared_cast)}")

    if not parts and shared_keywords:
        parts.append(f"similar themes: {', '.join(shared_keywords)}")

    return " / ".join(parts) if parts else "Recommended based on your taste profile"


# --- Batch explainer ---

def explain_all(results_df: pd.DataFrame,
                liked_df:   pd.DataFrame) -> pd.DataFrame:
    """
    Adds an 'explanation' column to results_df with a summary string for each row.
    Also adds seed_titles, shared_genres, shared_cast, shared_director, shared_keywords.
    """
    explanations = []
    for _, row in results_df.iterrows():
        exp = explain_recommendation(row, liked_df)
        explanations.append({
            "explanation":      exp.summary,
            "seed_titles":      " | ".join(exp.seed_titles),
            "shared_genres":    " | ".join(exp.shared_genres),
            "shared_cast":      " | ".join(exp.shared_cast),
            "shared_director":  exp.shared_director or "",
            "shared_keywords":  " | ".join(exp.shared_keywords),
        })

    exp_df = pd.DataFrame(explanations)
    return pd.concat([results_df.reset_index(drop=True), exp_df], axis=1)


if __name__ == "__main__":
    from app.recommendation.recommender import recommend

    print("[explain] Loading liked set...")
    liked_df = load_liked_enriched()

    print("[explain] Running recommender...")
    results_df = recommend(k=10)

    print("[explain] Building explanations...")
    results_df = explain_all(results_df, liked_df)

    print(f"\nTop-10 Recommendations with Explanations:\n{'=' * 60}")
    for _, rec_row in results_df.iterrows():
        year = str(rec_row["release_date"])[:4] if pd.notna(rec_row["release_date"]) else "?"
        print(f"  {rec_row['title']} ({year})")
        print(f"  → {rec_row['explanation']}")
        print()