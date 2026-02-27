import pandas as pd
import tmdbsimple as tmdb
import os, time, json

from app.api_keys.tmdb_api import API_KEY

CACHE_PATH = "data/tmdb_enrich_cache.json"
SLEEP_SECONDS = 0.08                  # stay well under TMDB rate limit
TOP_CAST = 5                          # how many cast members to keep per movie

def load_cache() -> dict:
    try:
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache: dict) -> None:
    os.makedirs("data", exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
        
def fetch_credits_keywords(movie_id: int) -> dict:
    m = tmdb.Movies(int(movie_id))
    
    credits = m.credits()
    time.sleep(SLEEP_SECONDS)
    
    cast = [
        member['name'] for member in credits.get('cast',[])[:TOP_CAST]
        if member.get('name')
    ]
    
    director = None
    for member in credits.get('crew'):
        if member.get("job") == "Director" and member.get("name"):
            director = member["name"]
            break
        
    kw_resp = m.keywords()
    time.sleep(SLEEP_SECONDS)
    
    keywords = [
        kw["name"]
        for kw in kw_resp.get("keywords", [])
        if kw.get("name")
    ]

    return {
        "cast": cast,
        "director": director,
        "keywords": keywords,
    }
    
def enrich_movies(movie_ids: list[int], label: str = "") -> dict[int, dict]:
    """
    Fetches cast, director, and keywords for each movie_id.
    Caches results so re-runs are free.

    Returns:
      enriched: tmdb_id -> {"cast": [...], "director": str|None, "keywords": [...]}
    """
    cache = load_cache()
    enriched = {}
    ids_to_fetch = [mid for mid in movie_ids if str(mid) not in cache]

    print(f"[enrich] {label} — {len(movie_ids)} total | "
          f"{len(movie_ids) - len(ids_to_fetch)} cached | {len(ids_to_fetch)} to fetch")

    for i, mid in enumerate(ids_to_fetch):
        try:
            result = fetch_credits_keywords(mid)
            cache[str(mid)] = result
        except Exception as e:
            print(f"  [warn] tmdb_id={mid} failed: {e}")
            cache[str(mid)] = {"cast": [], "director": None, "keywords": []}

        # save periodically so a crash doesn't lose all progress
        if (i + 1) % 100 == 0:
            save_cache(cache)
            print(f"  checkpoint: {i + 1}/{len(ids_to_fetch)} fetched")

    save_cache(cache)

    # build output from cache (covers both freshly fetched + previously cached)
    for mid in movie_ids:
        entry = cache.get(str(mid))
        if entry:
            enriched[mid] = entry
        else:
            enriched[mid] = {"cast": [], "director": None, "keywords": []}

    return enriched

def build_enriched_df(movie_ids: list[int], label: str = "") -> pd.DataFrame:
    """
    Convenience wrapper: returns a flat DataFrame with one row per movie_id.

    Columns:
      tmdb_id | director | cast_0..cast_N | cast_list | keywords_list
    """
    enriched = enrich_movies(movie_ids, label=label)

    rows = []
    for mid, data in enriched.items():
        row = {
            "tmdb_id": mid,
            "director": data.get("director"),
            # pipe-delimited strings are easy to read back and join on
            "cast_list": "|".join(data.get("cast", [])),
            "keywords_list": "|".join(data.get("keywords", [])),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    tmdb.API_KEY = API_KEY

    candidates_df = pd.read_parquet("data/candidates.parquet")
    rated_df      = pd.read_csv("data/letterboxd_ratings_mapped.csv")

    # collect all IDs that need enrichment: candidates + liked set (for scoring later)
    liked_ids = (
        rated_df
        .loc[rated_df["Rating"] >= 4.0, "tmdb_id"]
        .dropna()
        .astype(int)
        .tolist()
    )
    candidate_ids = candidates_df["tmdb_id"].dropna().astype(int).tolist()

    all_ids = list(dict.fromkeys(candidate_ids + liked_ids))  # dedup, preserve order

    print(f"Total unique IDs to enrich: {len(all_ids)}")

    enriched_df = build_enriched_df(all_ids, label="candidates+liked")

    enriched_df.to_parquet("data/tmdb_enriched.parquet", index=False)
    print(f"Saved enriched data: {len(enriched_df)} rows -> data/tmdb_enriched.parquet")

    # quick sanity check
    filled_director  = enriched_df["director"].notna().sum()
    filled_cast      = (enriched_df["cast_list"] != "").sum()
    filled_keywords  = (enriched_df["keywords_list"] != "").sum()
    print(f"  director filled:  {filled_director}/{len(enriched_df)}")
    print(f"  cast filled:      {filled_cast}/{len(enriched_df)}")
    print(f"  keywords filled:  {filled_keywords}/{len(enriched_df)}")






