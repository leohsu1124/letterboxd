import pandas as pd
import tmdbsimple as tmdb
import os, time, json
from collections import defaultdict
from .state import build_state

from app.api_keys.tmdb_api import API_KEY

# Configs 
CACHE_PATH = "data/tmdb_candidates_cache.json"  
SLEEP_SECONDS = 0.08                 # for TMDB

def load_cache():
    try:
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache):
    os.makedirs("data", exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
        
def fetch_recs_and_similar(movie_id: int, page: int = 1):
    m = tmdb.Movies(int(movie_id))
    recs = m.recommendations(page=page).get('results', [])
    time.sleep(SLEEP_SECONDS)
    sim = m.similar_movies(page=page).get('results',[])
    time.sleep(SLEEP_SECONDS)
    return recs, sim

def fetch_popular(page: int = 1):
    m = tmdb.Movies()
    popular = m.popular(page=page).get('results', [])
    time.sleep(SLEEP_SECONDS)
    return popular

def fetch_discover(spec: dict, page: int):
    """
    spec keys:
      - sort_by (str)
      - vote_count_gte (int)
      - optional extra discover params later
    """
    d = tmdb.Discover()
    res = d.movie(
        sort_by=spec["sort_by"],
        vote_count_gte=spec["vote_count_gte"],
        page=page,
    )
    time.sleep(SLEEP_SECONDS)
    return res.get("results", [])
    

def build_candidates(liked_ids: set[int], seen_ids: set[int], max_seed: int = 200, 
                     pages_per_seed: int = 1, popular_pages: int = 3, 
                     discover_specs: list[dict] | None = None):
    """
    Returns:
      candidates_df: one row per candidate movie with metadata + provenance
    """
    cache = load_cache()

    # candidate accumulator
    # id -> dict with counts and provenance (rec and sim hits, seed_ids)
    cand = {}
    provenance = defaultdict(lambda: {"rec_hits": 0, "sim_hits": 0, "discover_hits": 0, 
                                      "seed_ids": set(), "discover_sources": set()})

    # Limit seeds so you don’t blow up calls (you can raise later)
    seed_list = list(liked_ids)[:max_seed]

    for seed_id in seed_list:
        for page in range(1, pages_per_seed + 1):
            cache_key = f"{seed_id}|{page}"
            if cache_key in cache:
                recs, sim = cache[cache_key]["recs"], cache[cache_key]["sim"]
            else:
                recs, sim = fetch_recs_and_similar(seed_id, page=page)
                cache[cache_key] = {"recs": recs, "sim": sim}

            for item in recs:
                cid = item.get("id")
                if cid is None:
                    continue
                cid = int(cid)
                if cid in seen_ids:
                    continue
                cand.setdefault(cid, item)
                provenance[cid]["rec_hits"] += 1
                provenance[cid]["seed_ids"].add(seed_id)

            for item in sim:
                cid = item.get("id")
                if cid is None:
                    continue
                cid = int(cid)
                if cid in seen_ids:
                    continue
                cand.setdefault(cid, item)
                provenance[cid]["sim_hits"] += 1
                provenance[cid]["seed_ids"].add(seed_id)

    # popular fallback addition
    for p in range(1, popular_pages + 1):
        cache_key = f"popular|{p}"
        if cache_key in cache:
            pops = cache[cache_key]["popular"]
        else:
            pops = fetch_popular(page=p)
            cache[cache_key] = {"popular": pops}

        for item in pops:
            cid = item.get("id")
            if cid is None:
                continue
            cid = int(cid)
            if cid in seen_ids:
                continue
            cand.setdefault(cid, item)
            # popular doesn't add rec/sim hits, but we can tag its source implicitly
            
    # discover diversification addition
    if discover_specs:
        before = len(cand)

        for spec in discover_specs:
            name = spec.get("name", f'{spec["sort_by"]}|{spec["vote_count_gte"]}')
            page_start = spec["page_start"]
            pages = spec["pages"]

            for p in range(page_start, page_start + pages):
                cache_key = f'discover|{name}|{spec["sort_by"]}|{spec["vote_count_gte"]}|{p}'
                if cache_key in cache:
                    disc = cache[cache_key]["discover"]
                else:
                    disc = fetch_discover(spec, page=p)
                    cache[cache_key] = {"discover": disc}

                for item in disc:
                    cid = item.get("id")
                    if cid is None:
                        continue
                    cid = int(cid)
                    if cid in seen_ids:
                        continue

                    cand.setdefault(cid, item)
                    provenance[cid]["discover_hits"] += 1
                    provenance[cid]["discover_sources"].add(name)

        after = len(cand)
        print("Discover added uniques:", after - before)
    
    save_cache(cache)

    # dataframe build with provenance + lightweight metadata
    rows = []
    for cid, item in cand.items():
        prov = provenance[cid]
        rows.append({
            "tmdb_id": cid,
            "title": item.get("title"),
            "release_date": item.get("release_date"),
            "popularity": item.get("popularity"),
            "vote_average": item.get("vote_average"),
            "vote_count": item.get("vote_count"),
            "overview": item.get("overview"),
            "poster_path": item.get("poster_path"),
            "rec_hits": prov["rec_hits"],
            "sim_hits": prov["sim_hits"],
            "discover_hits": prov["discover_hits"],
            "discover_source_count": len(prov["discover_sources"]),
            "discover_sources": "|".join(sorted(prov["discover_sources"])),
            "seed_count": len(prov["seed_ids"]),
        })

    candidates_df = pd.DataFrame(rows)
    
    # simple initial score weighting:
    if not candidates_df.empty:
        candidates_df["score_seed"] = (
            2.0 * candidates_df["rec_hits"]
            + 1.0 * candidates_df["sim_hits"]
            + 0.5 * candidates_df["seed_count"]
            + 0.2 * candidates_df["discover_hits"]
            + 0.3 * candidates_df["discover_source_count"]
            + 0.01 * candidates_df["popularity"].fillna(0)
        )
        candidates_df = candidates_df.sort_values("score_seed", ascending=False)

    return candidates_df
        

if __name__ == '__main__':
    tmdb.API_KEY = API_KEY
    
    rated_df = pd.read_csv('data/letterboxd_ratings_mapped.csv')
    watched_df = pd.read_csv('data/letterboxd_watched_mapped.csv')
    
    state = build_state(rated_df=rated_df,watched_df=watched_df, like_threshold= 4.0)
    liked_ids = set(state.liked_ids)
    seen_ids = set(state.seen_ids)
    
    print(len(state.liked_ids),len(state.seen_ids))
    
    # config.py
    DISCOVER_SPECS = [
        {"name": "mid_depth_votes", "sort_by": "vote_count.desc", "vote_count_gte": 200, "page_start": 30, "pages": 75},
        {"name": "high_quality", "sort_by": "vote_average.desc", "vote_count_gte": 2000, "page_start": 1, "pages": 100},
        {"name": "recent_releases", "sort_by": "primary_release_date.desc", "vote_count_gte": 50, "page_start": 1, "pages": 25},
    ]
    
    candidates_df = build_candidates(liked_ids=liked_ids, seen_ids=seen_ids, max_seed=200, 
                                     pages_per_seed=3, popular_pages=10, discover_specs=DISCOVER_SPECS)
    
    candidates_df.to_csv("data/candidates.csv", index=False)
    print("Candidates:", len(candidates_df))