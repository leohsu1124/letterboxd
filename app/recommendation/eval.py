"""
eval.py — Offline evaluation for the movie recommender (Phase 4).

Performs a time-based train/test split on rated positives, runs the
recommender scoring pipeline on the train set, and evaluates how well
it recovers held-out test positives.

Key design decisions:
  - Test positives are INJECTED into the candidate pool (they were excluded
    during candidate generation as seen movies).
  - Genre IDs are fetched from TMDB API for injected movies (cached locally).
  - Supports parameter sweeps for experiment comparison.

Metrics:
  - nDCG@5, nDCG@10, Recall@10, Recall@50, Hit Rate@k
  - Catalog coverage, Novelty

Usage:
    python eval.py                                    # single run
    python eval.py --sweep                            # parameter sweep
    python eval.py --train_frac 0.85 --cast_w 0.3    # custom config
"""

import argparse
import json
import os
import time
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

# ── imports from your pipeline ─────────────────────────────────────────────────

from app.recommendation.features import (
    load_merged,
    build_feature_matrix,
    encode_liked,
    GENRE_WEIGHT as DEFAULT_GENRE_WEIGHT,
    CAST_WEIGHT as DEFAULT_CAST_WEIGHT,
    DIRECTOR_WEIGHT as DEFAULT_DIRECTOR_WEIGHT,
    KEYWORD_WEIGHT as DEFAULT_KEYWORD_WEIGHT,
    TFIDF_WEIGHT as DEFAULT_TFIDF_WEIGHT,
)
from app.recommendation.recommender import (
    compute_similarity_scores,
    compute_popularity_prior,
    compute_final_scores,
    TOP_N_LIKED,
    POPULARITY_WEIGHT,
)

# genre cache for TMDB lookups
GENRE_CACHE_PATH = "data/tmdb_genre_cache.json"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TMDB GENRE FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

def load_genre_cache() -> dict:
    try:
        with open(GENRE_CACHE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_genre_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(GENRE_CACHE_PATH), exist_ok=True)
    with open(GENRE_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_genre_ids(tmdb_ids: list[int]) -> dict[int, list[int]]:
    """
    Fetches genre_ids for a list of TMDB movie IDs using the TMDB API.
    Results are cached so subsequent runs don't hit the API.

    Returns:
        dict mapping tmdb_id -> list of genre_id ints
    """
    cache = load_genre_cache()
    result = {}
    ids_to_fetch = []

    for tid in tmdb_ids:
        key = str(tid)
        if key in cache:
            result[tid] = cache[key]
        else:
            ids_to_fetch.append(tid)

    if ids_to_fetch:
        try:
            import tmdbsimple as tmdb
            from app.api_keys.tmdb_api import API_KEY
            tmdb.API_KEY = API_KEY

            print(f"[eval] Fetching genre_ids from TMDB for {len(ids_to_fetch)} movies...")
            for i, tid in enumerate(ids_to_fetch):
                try:
                    movie = tmdb.Movies(tid)
                    info = movie.info()
                    genres = [g["id"] for g in info.get("genres", [])]
                    cache[str(tid)] = genres
                    result[tid] = genres
                    time.sleep(0.08)  # rate limit

                    if (i + 1) % 20 == 0:
                        save_genre_cache(cache)
                        print(f"  fetched {i + 1}/{len(ids_to_fetch)}")
                except Exception as e:
                    print(f"  [warn] tmdb_id={tid} failed: {e}")
                    cache[str(tid)] = []
                    result[tid] = []

            save_genre_cache(cache)
            print(f"[eval] Genre fetch complete. Cached {len(cache)} movies.")

        except ImportError:
            print("[eval] WARNING: tmdbsimple not available — skipping genre fetch.")
            print("  Install with: pip install tmdbsimple")
            print("  Or populate data/tmdb_genre_cache.json manually.")
            for tid in ids_to_fetch:
                result[tid] = []

    else:
        print(f"[eval] All {len(tmdb_ids)} genre lookups served from cache.")

    return result


def fetch_overviews(tmdb_ids: list[int]) -> dict[int, str]:
    """
    Fetches overview text for movies using the TMDB API.
    Reuses the genre cache file (adds 'overview' key).
    """
    cache = load_genre_cache()
    result = {}
    ids_to_fetch = []

    for tid in tmdb_ids:
        key = f"overview_{tid}"
        if key in cache:
            result[tid] = cache[key]
        else:
            ids_to_fetch.append(tid)

    if ids_to_fetch:
        try:
            import tmdbsimple as tmdb
            from app.api_keys.tmdb_api import API_KEY
            tmdb.API_KEY = API_KEY

            print(f"[eval] Fetching overviews from TMDB for {len(ids_to_fetch)} movies...")
            for i, tid in enumerate(ids_to_fetch):
                try:
                    movie = tmdb.Movies(tid)
                    info = movie.info()
                    overview = info.get("overview", "")
                    cache[f"overview_{tid}"] = overview
                    result[tid] = overview
                    time.sleep(0.08)
                except Exception as e:
                    cache[f"overview_{tid}"] = ""
                    result[tid] = ""

            save_genre_cache(cache)

        except ImportError:
            for tid in ids_to_fetch:
                result[tid] = ""
    else:
        print(f"[eval] All {len(tmdb_ids)} overview lookups served from cache.")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TIME-BASED TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def time_split_positives(
    rated_df: pd.DataFrame,
    like_threshold: float = 4.0,
    train_frac: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits ratings by time. Oldest train_frac of positives go to train,
    the rest to test. Negatives always go to train.
    """
    df = rated_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    mapped = df[df["tmdb_status"] == "mapped"].copy()
    positives = mapped[mapped["Rating"] >= like_threshold].reset_index(drop=True)

    n_train = int(len(positives) * train_frac)
    train_pos = positives.iloc[:n_train]
    test_pos  = positives.iloc[n_train:]

    cutoff_date = train_pos["Date"].max()
    train_df = mapped[mapped["Date"] <= cutoff_date].copy()

    print(f"[eval] Time-based split (train_frac={train_frac:.0%}):")
    print(f"  Cutoff date   : {cutoff_date.date()}")
    print(f"  Train ratings : {len(train_df)}  (positives: {len(train_pos)})")
    print(f"  Test positives: {len(test_pos)}")

    return train_df, test_pos, df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. INJECT TEST POSITIVES INTO CANDIDATE POOL
# ═══════════════════════════════════════════════════════════════════════════════

def inject_test_into_pool(
    test_pos: pd.DataFrame,
    candidate_matrix: sp.csr_matrix,
    candidate_ids: list[int],
    candidates_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    encoders: dict,
) -> tuple[sp.csr_matrix, list[int], pd.DataFrame]:
    """
    Encodes test positive movies into the candidate feature space and appends them.
    Fetches genre_ids and overviews from TMDB API (cached) for proper featurization.
    """
    existing_ids = set(candidate_ids)
    test_ids = test_pos["tmdb_id"].dropna().astype(int).tolist()
    new_ids = [tid for tid in test_ids if tid not in existing_ids]

    if not new_ids:
        print("[eval] All test positives already in candidate pool.")
        return candidate_matrix, candidate_ids, candidates_df

    # fetch genre_ids and overviews from TMDB
    genre_map = fetch_genre_ids(new_ids)
    overview_map = fetch_overviews(new_ids)

    # gather enrichment data
    test_enriched = enriched_df[enriched_df["tmdb_id"].isin(new_ids)].copy()

    test_meta = test_pos[test_pos["tmdb_id"].isin(new_ids)][
        ["tmdb_id", "Name"]
    ].drop_duplicates("tmdb_id").copy()
    test_meta["tmdb_id"] = test_meta["tmdb_id"].astype(int)
    test_meta = test_meta.rename(columns={"Name": "title"})

    # merge enrichment
    test_merged = test_meta.merge(
        test_enriched[["tmdb_id", "cast_list", "keywords_list", "director"]],
        on="tmdb_id", how="left"
    )

    test_merged["cast_list"]     = test_merged["cast_list"].fillna("")
    test_merged["keywords_list"] = test_merged["keywords_list"].fillna("")
    test_merged["director"]      = test_merged["director"].fillna("")

    # add genre_ids from TMDB fetch
    test_merged["genre_ids"] = test_merged["tmdb_id"].map(
        lambda tid: str(genre_map.get(tid, []))
    )

    # add overviews from TMDB fetch
    test_merged["overview"] = test_merged["tmdb_id"].map(
        lambda tid: overview_map.get(tid, "")
    )

    has_genres = sum(1 for tid in new_ids if genre_map.get(tid))
    has_overview = sum(1 for tid in new_ids if overview_map.get(tid))
    has_enrich = len(test_enriched)

    # encode into candidate feature space
    test_matrix = encode_liked(test_merged, encoders)

    injected_ids = test_merged["tmdb_id"].tolist()

    print(f"[eval] Injecting {len(injected_ids)} test positives into candidate pool")
    print(f"  With genres:     {has_genres}/{len(new_ids)}")
    print(f"  With overview:   {has_overview}/{len(new_ids)}")
    print(f"  With enrichment: {has_enrich}/{len(new_ids)}")

    # build minimal candidates_df rows
    inject_rows = []
    for _, row in test_merged.iterrows():
        inject_rows.append({
            "tmdb_id": int(row["tmdb_id"]),
            "title": row.get("title", ""),
            "release_date": "",
            "popularity": 0.0,
            "vote_average": 0.0,
            "vote_count": 999,
            "overview": row.get("overview", ""),
            "poster_path": "",
            "genre_ids": row.get("genre_ids", "[]"),
            "rec_hits": 0,
            "sim_hits": 0,
            "discover_hits": 0,
            "seed_count": 0,
            "score_seed": 0,
        })
    inject_df = pd.DataFrame(inject_rows)

    for col in candidates_df.columns:
        if col not in inject_df.columns:
            inject_df[col] = 0 if candidates_df[col].dtype in ("float64", "int64", "float32") else ""

    augmented_matrix = sp.vstack([candidate_matrix, test_matrix], format="csr")
    augmented_ids = candidate_ids + injected_ids
    augmented_df = pd.concat(
        [candidates_df, inject_df[candidates_df.columns]],
        ignore_index=True
    )

    return augmented_matrix, augmented_ids, augmented_df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def score_all_candidates(
    candidate_matrix: sp.csr_matrix,
    candidate_ids: list[int],
    liked_matrix: sp.csr_matrix,
    candidates_df: pd.DataFrame,
    seen_ids: set[int],
    min_vote_count: int = 300,
) -> pd.DataFrame:
    """
    Scores every candidate against the liked set. Returns ranked DataFrame.
    Raw ranking without MMR/genre-cap for clean evaluation.
    """
    id_to_idx = {tid: i for i, tid in enumerate(candidate_ids)}

    df = candidates_df[candidates_df["tmdb_id"].isin(id_to_idx)].copy()
    df["matrix_idx"] = df["tmdb_id"].map(id_to_idx)
    df = df.sort_values("matrix_idx").reset_index(drop=True)

    df = df[df["vote_count"] >= min_vote_count].copy()
    surviving_indices = df["matrix_idx"].values
    cand_mat = candidate_matrix[surviving_indices]

    sim_scores   = compute_similarity_scores(cand_mat, liked_matrix)
    pop_prior    = compute_popularity_prior(df)
    final_scores = compute_final_scores(sim_scores, pop_prior)

    df = df.reset_index(drop=True)
    df["sim_score"]   = sim_scores
    df["final_score"] = final_scores

    df = df[~df["tmdb_id"].isin(seen_ids)].copy()
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)

def ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, tid in enumerate(ranked_ids[:k]) if tid in relevant_ids)
    n_rel = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0

def hit_rate_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    return 1.0 if set(ranked_ids[:k]) & relevant_ids else 0.0

def catalog_coverage(recommended_ids: set[int], total_candidate_ids: set[int]) -> float:
    if not total_candidate_ids:
        return 0.0
    return len(recommended_ids & total_candidate_ids) / len(total_candidate_ids)

def novelty(recommended_ids: list[int], popularity_dict: dict[int, float]) -> float:
    if not recommended_ids:
        return 0.0
    total_pop = sum(popularity_dict.values())
    if total_pop == 0:
        return 0.0
    scores = []
    for tid in recommended_ids:
        pop = popularity_dict.get(tid, 1.0)
        prob = pop / total_pop
        if prob > 0:
            scores.append(-np.log2(prob))
    return float(np.mean(scores)) if scores else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN EVALUATION (SINGLE RUN)
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval(
    rated_path: str = "data/letterboxd_ratings_mapped.csv",
    watched_path: str = "data/letterboxd_watched_mapped.csv",
    candidates_path: str = "data/candidates.parquet",
    enriched_path: str = "data/tmdb_enriched.parquet",
    like_threshold: float = 4.0,
    train_frac: float = 0.80,
    ks: list[int] | None = None,
    min_vote_count: int = 300,
    output_dir: str = "data/eval_results",
    # feature weight overrides (None = use defaults from features.py)
    cast_weight: float | None = None,
    director_weight: float | None = None,
    genre_weight: float | None = None,
    keyword_weight: float | None = None,
    tfidf_weight: float | None = None,
    verbose: bool = True,
) -> dict:
    """
    End-to-end evaluation pipeline with optional feature weight overrides.
    """
    if ks is None:
        ks = [5, 10, 20, 50]

    # apply weight overrides by monkey-patching features module
    import app.recommendation.features as feat_mod
    original_weights = {
        "GENRE_WEIGHT": feat_mod.GENRE_WEIGHT,
        "CAST_WEIGHT": feat_mod.CAST_WEIGHT,
        "DIRECTOR_WEIGHT": feat_mod.DIRECTOR_WEIGHT,
        "KEYWORD_WEIGHT": feat_mod.KEYWORD_WEIGHT,
        "TFIDF_WEIGHT": feat_mod.TFIDF_WEIGHT,
    }
    if genre_weight is not None:    feat_mod.GENRE_WEIGHT = genre_weight
    if cast_weight is not None:     feat_mod.CAST_WEIGHT = cast_weight
    if director_weight is not None: feat_mod.DIRECTOR_WEIGHT = director_weight
    if keyword_weight is not None:  feat_mod.KEYWORD_WEIGHT = keyword_weight
    if tfidf_weight is not None:    feat_mod.TFIDF_WEIGHT = tfidf_weight

    start_time = time.time()

    try:
        return _run_eval_inner(
            rated_path, watched_path, candidates_path, enriched_path,
            like_threshold, train_frac, ks, min_vote_count, output_dir,
            feat_mod, verbose,
        )
    finally:
        # restore original weights
        for k, v in original_weights.items():
            setattr(feat_mod, k, v)


def _run_eval_inner(
    rated_path, watched_path, candidates_path, enriched_path,
    like_threshold, train_frac, ks, min_vote_count, output_dir,
    feat_mod, verbose,
):
    start_time = time.time()

    if verbose:
        print("[eval] Loading data...")
    rated_df   = pd.read_csv(rated_path)
    watched_df = pd.read_csv(watched_path)

    # split
    train_df, test_pos, _ = time_split_positives(rated_df, like_threshold, train_frac)

    train_liked_ids = set(
        train_df.loc[train_df["Rating"] >= like_threshold, "tmdb_id"]
        .dropna().astype(int)
    )

    train_seen_ids = set(train_df["tmdb_id"].dropna().astype(int))
    watched_mapped = watched_df[watched_df["tmdb_status"] == "mapped"]
    train_seen_ids |= set(watched_mapped["tmdb_id"].dropna().astype(int))

    test_relevant_ids = set(test_pos["tmdb_id"].dropna().astype(int))
    eval_seen_ids = train_seen_ids - test_relevant_ids

    if verbose:
        print(f"[eval] Train liked: {len(train_liked_ids)} | "
              f"Eval seen: {len(eval_seen_ids)} | "
              f"Test relevant: {len(test_relevant_ids)}")

    # build candidate feature matrix
    if verbose:
        print("[eval] Building feature matrices...")

    merged_df = load_merged(candidates_path, enriched_path)
    candidate_matrix, feature_names, encoders = build_feature_matrix(merged_df)
    candidate_ids = merged_df["tmdb_id"].tolist()
    candidates_df = pd.read_parquet(candidates_path)

    # inject test positives
    enriched_df = pd.read_parquet(enriched_path)

    candidate_matrix, candidate_ids, candidates_df = inject_test_into_pool(
        test_pos=test_pos,
        candidate_matrix=candidate_matrix,
        candidate_ids=candidate_ids,
        candidates_df=candidates_df,
        enriched_df=enriched_df,
        encoders=encoders,
    )

    # build liked matrix from TRAIN set only
    liked_enriched = enriched_df[enriched_df["tmdb_id"].isin(train_liked_ids)].copy()
    liked_meta = candidates_df[candidates_df["tmdb_id"].isin(train_liked_ids)][
        ["tmdb_id", "genre_ids", "overview"]
    ]
    liked_enriched = liked_enriched.merge(liked_meta, on="tmdb_id", how="left")
    liked_enriched["genre_ids"]     = liked_enriched["genre_ids"].fillna("[]")
    liked_enriched["overview"]      = liked_enriched["overview"].fillna("")
    liked_enriched["cast_list"]     = liked_enriched["cast_list"].fillna("")
    liked_enriched["keywords_list"] = liked_enriched["keywords_list"].fillna("")
    liked_enriched["director"]      = liked_enriched["director"].fillna("")

    liked_matrix = encode_liked(liked_enriched, encoders)
    if verbose:
        print(f"[eval] Liked matrix (train): {liked_matrix.shape}")

    # score
    if verbose:
        print("[eval] Scoring candidates...")
    ranked_df = score_all_candidates(
        candidate_matrix=candidate_matrix,
        candidate_ids=candidate_ids,
        liked_matrix=liked_matrix,
        candidates_df=candidates_df,
        seen_ids=eval_seen_ids,
        min_vote_count=min_vote_count,
    )

    ranked_ids = ranked_df["tmdb_id"].tolist()
    total_ranked = len(ranked_ids)

    candidate_id_set = set(int(x) for x in candidate_ids)
    test_in_pool   = test_relevant_ids & candidate_id_set
    test_in_ranked = test_relevant_ids & set(ranked_ids)

    if verbose:
        print(f"[eval] Ranked: {total_ranked} | "
              f"Test in pool: {len(test_in_pool)}/{len(test_relevant_ids)} | "
              f"Test in ranked: {len(test_in_ranked)}/{len(test_relevant_ids)}")

    # compute metrics
    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"]   = recall_at_k(ranked_ids, test_relevant_ids, k)
        metrics[f"ndcg@{k}"]     = ndcg_at_k(ranked_ids, test_relevant_ids, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(ranked_ids, test_relevant_ids, k)

    top_100_set = set(ranked_ids[:100])
    metrics["coverage@100"] = catalog_coverage(top_100_set, candidate_id_set)

    pop_dict = dict(zip(
        candidates_df["tmdb_id"].astype(int),
        candidates_df["popularity"].fillna(0).astype(float)
    ))
    metrics["novelty@10"] = novelty(ranked_ids[:10], pop_dict)
    metrics["novelty@50"] = novelty(ranked_ids[:50], pop_dict)

    # per-test-item details
    test_item_details = []
    for _, row in test_pos.iterrows():
        tid = int(row["tmdb_id"]) if pd.notna(row["tmdb_id"]) else None
        if tid is None:
            continue
        rank = ranked_ids.index(tid) + 1 if tid in ranked_ids else None
        score_val = float(ranked_df.loc[ranked_df["tmdb_id"] == tid, "final_score"].iloc[0]) if tid in ranked_ids else None
        sim_val   = float(ranked_df.loc[ranked_df["tmdb_id"] == tid, "sim_score"].iloc[0]) if tid in ranked_ids else None
        test_item_details.append({
            "tmdb_id": tid,
            "title": row["Name"],
            "rating": float(row["Rating"]),
            "rank": rank,
            "final_score": score_val,
            "sim_score": sim_val,
            "in_top_10": rank is not None and rank <= 10,
            "in_top_50": rank is not None and rank <= 50,
            "in_top_100": rank is not None and rank <= 100,
        })

    elapsed = time.time() - start_time

    config = {
        "like_threshold": like_threshold,
        "train_frac": train_frac,
        "min_vote_count": min_vote_count,
        "genre_weight": feat_mod.GENRE_WEIGHT,
        "cast_weight": feat_mod.CAST_WEIGHT,
        "director_weight": feat_mod.DIRECTOR_WEIGHT,
        "keyword_weight": feat_mod.KEYWORD_WEIGHT,
        "tfidf_weight": feat_mod.TFIDF_WEIGHT,
        "top_n_liked": TOP_N_LIKED,
        "popularity_weight": POPULARITY_WEIGHT,
        "n_train_liked": len(train_liked_ids),
        "n_test_relevant": len(test_relevant_ids),
        "n_test_in_pool": len(test_in_pool),
        "n_ranked": total_ranked,
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "config": config,
        "metrics": metrics,
        "test_items": test_item_details,
    }

    # save
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"eval_{ts}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        _print_report(metrics, config, test_item_details, candidate_id_set, elapsed, output_path)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PARAMETER SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep(
    cast_weights: list[float]      = [0.0, 0.05],
    director_weights: list[float]  = [6.0, 8.0, 10.0],
    keyword_weights: list[float]   = [2.0, 3.0, 5.0, 8.0, 12.0],
    train_fracs: list[float]       = [0.70, 0.75, 0.80, 0.85],
    genre_weight: float | None     = None,
    tfidf_weight: float | None     = None,
    output_dir: str = "data/eval_results",
    **kwargs,
) -> pd.DataFrame:
    """
    Runs a grid sweep over cast_weight, director_weight, keyword_weight,
    and train_frac. Returns a DataFrame of all results and saves to CSV.

    Sweep 2 rationale (based on sweep 1 findings):
      - director_weight is the dominant signal → push higher (4-10)
      - cast_weight barely matters and 23k features drown signal → push lower (0.01-0.3)
      - keyword_weight has 15k features, may also drown signal → sweep (0.1-2.0)
      - train_frac=0.90 had zero recall → drop it
      - genre_weight and tfidf_weight can be fixed or overridden
    """
    all_results = []
    combos = list(itertools.product(cast_weights, director_weights, keyword_weights, train_fracs))
    total = len(combos)

    print(f"\n{'=' * 65}")
    print(f"  PARAMETER SWEEP v2: {total} configurations")
    print(f"  cast_weights     = {cast_weights}")
    print(f"  director_weights = {director_weights}")
    print(f"  keyword_weights  = {keyword_weights}")
    print(f"  train_fracs      = {train_fracs}")
    if genre_weight is not None:
        print(f"  genre_weight     = {genre_weight} (fixed)")
    if tfidf_weight is not None:
        print(f"  tfidf_weight     = {tfidf_weight} (fixed)")
    print(f"{'=' * 65}\n")

    for i, (cw, dw, kw, tf) in enumerate(combos):
        print(f"--- Run {i + 1}/{total}: cast={cw} dir={dw} kw={kw} tf={tf} ---", end="  ")

        try:
            result = run_eval(
                cast_weight=cw,
                director_weight=dw,
                keyword_weight=kw,
                genre_weight=genre_weight,
                tfidf_weight=tfidf_weight,
                train_frac=tf,
                output_dir=output_dir,
                verbose=False,
                **kwargs,
            )

            row = {
                "cast_weight": cw,
                "director_weight": dw,
                "keyword_weight": kw,
                "train_frac": tf,
                "ndcg@5": result["metrics"]["ndcg@5"],
                "ndcg@10": result["metrics"]["ndcg@10"],
                "recall@10": result["metrics"]["recall@10"],
                "recall@50": result["metrics"]["recall@50"],
                "hit_rate@10": result["metrics"]["hit_rate@10"],
                "coverage@100": result["metrics"]["coverage@100"],
                "novelty@10": result["metrics"]["novelty@10"],
                "elapsed": result["elapsed_seconds"],
            }
            all_results.append(row)

            print(f"nDCG@10={row['ndcg@10']:.4f}  R@10={row['recall@10']:.4f}  R@50={row['recall@50']:.4f}")

        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                "cast_weight": cw,
                "director_weight": dw,
                "keyword_weight": kw,
                "train_frac": tf,
                "ndcg@5": None, "ndcg@10": None,
                "recall@10": None, "recall@50": None,
                "hit_rate@10": None, "coverage@100": None,
                "novelty@10": None, "elapsed": None,
            })

    sweep_df = pd.DataFrame(all_results)

    # save CSV
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"sweep_{ts}.csv")
    sweep_df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 65}")
    print(f"  SWEEP v2 COMPLETE: {len(sweep_df)} runs")
    print(f"  Results saved to: {csv_path}")
    print(f"{'=' * 65}")

    # print best configs
    for metric in ["ndcg@5", "ndcg@10", "recall@10", "recall@50"]:
        valid = sweep_df.dropna(subset=[metric])
        if not valid.empty:
            best = valid.loc[valid[metric].idxmax()]
            print(f"\n  Best {metric}: {best[metric]:.4f}")
            print(f"    cast={best['cast_weight']}  dir={best['director_weight']}  "
                  f"kw={best['keyword_weight']}  tf={best['train_frac']}")

    # show top-10 overall (by ndcg@10)
    top10 = sweep_df.dropna(subset=["ndcg@10"]).nlargest(10, "ndcg@10")
    if not top10.empty:
        print(f"\n  Top 10 by nDCG@10:")
        for _, r in top10.iterrows():
            print(f"    cast={r['cast_weight']:<5} dir={r['director_weight']:<5} "
                  f"kw={r['keyword_weight']:<5} tf={r['train_frac']:<5} "
                  f"nDCG@10={r['ndcg@10']:.4f}  R@50={r['recall@50']:.4f}")

    return sweep_df


# ═══════════════════════════════════════════════════════════════════════════════
# 8. REPORT PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

def _print_report(metrics, config, test_item_details, candidate_id_set, elapsed, output_path):

    print(f"\n{'=' * 65}")
    print(f"  EVALUATION RESULTS  ({elapsed:.1f}s)")
    print(f"{'=' * 65}")

    print(f"\n  Config:")
    print(f"    train_frac={config['train_frac']}  like_threshold={config['like_threshold']}")
    print(f"    genre_w={config['genre_weight']}  cast_w={config['cast_weight']}  "
          f"dir_w={config['director_weight']}  kw_w={config['keyword_weight']}  "
          f"tfidf_w={config['tfidf_weight']}")
    print(f"    train liked={config['n_train_liked']}  test positives={config['n_test_relevant']}")

    print(f"\n  Core Metrics:")
    print(f"    {'Metric':<25} {'Value':>8}")
    print(f"    {'─' * 35}")
    for prefix in ["ndcg", "recall", "hit_rate"]:
        for k_val in sorted(set(
            int(key.split('@')[1]) for key in metrics
            if key.startswith(f'{prefix}@') and key.split('@')[1].isdigit()
        )):
            val = metrics[f"{prefix}@{k_val}"]
            label = f"{prefix.replace('_', ' ').title()}@{k_val}"
            print(f"    {label:<25} {val:>8.4f}")

    print(f"\n  Diversity:")
    cov_count = int(metrics['coverage@100'] * len(candidate_id_set))
    print(f"    Coverage@100 = {metrics['coverage@100']:.4f}  ({cov_count}/{len(candidate_id_set)})")
    print(f"    Novelty@10   = {metrics['novelty@10']:.2f} bits")

    found_top10  = sum(1 for t in test_item_details if t["in_top_10"])
    found_top50  = sum(1 for t in test_item_details if t["in_top_50"])
    found_top100 = sum(1 for t in test_item_details if t["in_top_100"])
    total_test   = len(test_item_details)

    print(f"\n  Test Items: {found_top10}/{total_test} in top-10 | "
          f"{found_top50}/{total_test} in top-50 | "
          f"{found_top100}/{total_test} in top-100")

    print(f"\n  Per-Item Ranks:")
    sorted_items = sorted(test_item_details, key=lambda x: x["rank"] if x["rank"] else 99999)
    for item in sorted_items:
        if item["rank"]:
            rank_str = f"rank {item['rank']:>5}  (sim={item['sim_score']:.4f})"
        else:
            rank_str = "NOT RANKED"
        marker = " ★" if item["rank"] and item["rank"] <= 10 else ""
        print(f"    {item['title']:<40} {rank_str}{marker}")

    print(f"\n  Saved: {output_path}")
    print(f"{'=' * 65}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation for movie recommender")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep instead of single eval")
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--like_threshold", type=float, default=4.0)
    parser.add_argument("--min_votes", type=int, default=300)
    parser.add_argument("--cast_w", type=float, default=None)
    parser.add_argument("--dir_w", type=float, default=None)
    parser.add_argument("--genre_w", type=float, default=None)
    parser.add_argument("--kw_w", type=float, default=None)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--output_dir", type=str, default="data/eval_results")

    args = parser.parse_args()

    if args.sweep:
        sweep_df = run_sweep(output_dir=args.output_dir)
    else:
        results = run_eval(
            train_frac=args.train_frac,
            like_threshold=args.like_threshold,
            min_vote_count=args.min_votes,
            cast_weight=args.cast_w,
            director_weight=args.dir_w,
            genre_weight=args.genre_w,
            keyword_weight=args.kw_w,
            ks=args.ks,
            output_dir=args.output_dir,
        )