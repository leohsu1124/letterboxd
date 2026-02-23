import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class RecommenderState:
    rated_ids: frozenset[int]
    watched_ids: frozenset[int]
    seen_ids: frozenset[int]
    liked_ids: frozenset[int]

def build_state(rated_df: pd.DataFrame, watched_df: pd.DataFrame, like_threshold: float = 4.0) -> RecommenderState:
    # only keeping succesful mappings (excluding things like tv shows)
    rated_mapped = rated_df[rated_df["tmdb_status"] == "mapped"]
    watched_mapped = watched_df[watched_df["tmdb_status"] == "mapped"]

    rated_ids = frozenset(rated_mapped["tmdb_id"].dropna().astype(int))
    watched_ids = frozenset(watched_mapped["tmdb_id"].dropna().astype(int))
    
    seen_ids = frozenset(set(rated_ids) | set(watched_ids))
    liked_ids = frozenset(
        rated_df.loc[rated_df["Rating"] >= like_threshold, "tmdb_id"].dropna().astype(int).astype(int).tolist()
    )
    return RecommenderState(
        rated_ids=rated_ids,
        watched_ids=watched_ids,
        seen_ids=seen_ids,
        liked_ids=liked_ids,
    )