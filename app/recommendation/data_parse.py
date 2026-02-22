from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import tmdbsimple as tmdb
from app.api_keys.tmdb_api import API_KEY
from tmdbsimple import Search

from data.data_mapper import Movie, data_loadmap

# Configs 
CACHE_PATH = "data/tmdb_cache.json"   # title|year -> best match payload
SLEEP_SECONDS = 0.08                 # for TMDB
MIN_SCORE_TO_ACCEPT = 6.0            # flagging threshold


# Helpers
def normalize_title(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # punctuation removal 
    return s

@dataclass
class TMDBMatch:
    tmdb_id: int
    title: str
    year: Optional[int]
    score: float


def load_cache(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_cache(path: str, cache: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)

def score_candidate(lb_title: str, lb_year: Optional[int], cand: Dict[str, Any]) -> TMDBMatch:
    tmdb_title = cand.get("title") or ""
    tmdb_id = int(cand["id"])

    tmdb_year = None
    rd = cand.get("release_date") or ""
    if len(rd) >= 4 and rd[:4].isdigit():
        tmdb_year = int(rd[:4])

    # base score: popularity is helpful but should not dominate
    score = float(cand.get("popularity") or 0.0) * 0.01

    # title agreement
    a = normalize_title(lb_title)
    b = normalize_title(tmdb_title)
    if a == b:
        score += 10.0
    elif a in b or b in a:
        score += 3.0

    # year agreement
    if lb_year is not None and tmdb_year is not None:
        if tmdb_year == lb_year:
            score += 6.0
        elif abs(tmdb_year - lb_year) == 1:
            score += 2.0
        else:
            score -= 2.0

    return TMDBMatch(tmdb_id=tmdb_id, title=tmdb_title, year=tmdb_year, score=score)

def tmdb_search_best(title: str, year: Optional[int]) -> Optional[TMDBMatch]:
    s = tmdb.Search()
    resp = s.movie(query=title, year=year) if year else s.movie(query=title)
    results = resp.get("results", [])
    if not results:
        return None

    best: Optional[TMDBMatch] = None
    for cand in results[:10]:  # top results are enough
        m = score_candidate(title, year, cand)
        if best is None or m.score > best.score:
            best = m
    return best

# mapping letterboxd to tmdb data
def map_ratings_to_tmdb(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Name", "Year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cache = load_cache(CACHE_PATH)

    out = df.copy()
    out["tmdb_id"] = pd.NA
    out["tmdb_match_title"] = pd.NA
    out["tmdb_match_year"] = pd.NA
    out["tmdb_match_score"] = pd.NA
    out["tmdb_status"] = "unmapped"  # mapped|low_conf|not_found

    for i, row in out.iterrows():
        title = str(row["Name"]).strip()
        year = int(row["Year"]) if pd.notna(row["Year"]) and str(row["Year"]).strip() != "" else None

        key = f"{normalize_title(title)}|{year or ''}"

        if key in cache:
            payload = cache[key]
            if payload is None:
                out.at[i, "tmdb_status"] = "not_found"
                continue

            out.at[i, "tmdb_id"] = payload["tmdb_id"]
            out.at[i, "tmdb_match_title"] = payload["title"]
            out.at[i, "tmdb_match_year"] = payload.get("year")
            out.at[i, "tmdb_match_score"] = payload["score"]
            out.at[i, "tmdb_status"] = "mapped" if payload["score"] >= MIN_SCORE_TO_ACCEPT else "low_conf"
            continue

        # not cached -> hit API
        match = tmdb_search_best(title, year)
        time.sleep(SLEEP_SECONDS)

        if match is None:
            cache[key] = None
            out.at[i, "tmdb_status"] = "not_found"
            continue

        cache[key] = {"tmdb_id": match.tmdb_id, "title": match.title, "year": match.year, "score": match.score}

        out.at[i, "tmdb_id"] = match.tmdb_id
        out.at[i, "tmdb_match_title"] = match.title
        out.at[i, "tmdb_match_year"] = match.year
        out.at[i, "tmdb_match_score"] = match.score
        out.at[i, "tmdb_status"] = "mapped" if match.score >= MIN_SCORE_TO_ACCEPT else "low_conf"

    save_cache(CACHE_PATH, cache)
    return out


if __name__ == '__main__':
    tmdb.API_KEY = API_KEY
    
    df = pd.read_csv('data/letterboxd_data/ratings.csv')
    mapped = map_ratings_to_tmdb(df)
    mapped.to_csv('data/letterboxd_ratings_mapped.csv',index=False)
    
    print(mapped["tmdb_status"].value_counts(dropna=False))
    print("Mapped %:", (mapped["tmdb_status"] == "mapped").mean())
    print("Low conf %:", (mapped["tmdb_status"] == "low_conf").mean())
    print("Not found %:", (mapped["tmdb_status"] == "not_found").mean())

    # write review file for low confidence matches
    review = mapped[mapped["tmdb_status"] == "low_conf"][["Name", "Year", "tmdb_id", "tmdb_match_title", "tmdb_match_year", "tmdb_match_score", "Letterboxd URI"]]
    if len(review) > 0:
        review.to_csv("data/tmdb_needs_review.csv", index=False)
        print(f"Wrote {len(review)} low-confidence matches to data/tmdb_needs_review.csv")