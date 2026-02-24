## Next Watch (w/ Leo's Letterboxd)

Since the process for getting a [Letterboxd API](https://api-docs.letterboxd.com/) rather tedious, I opted to use a _(new-ish??)_ extract personal data feature.

To ensure I had a grasp on the data extracted, I made a time-lapse animation based on my own ratings (423 movies / 1,104 watched at time of writing).

![animation](/animation/animation.gif)

---

1. **Data Foundations**

   Started off by importing my `ratings.csv` into a table in order to build an ID resolution step & map each rated movie to its `tmdb_id` in the [TMDB](https://www.themoviedb.org/) dataset. We can utilize the [TMDB API](https://developer.themoviedb.org/docs/getting-started) to implement the mapping with a `title + year` search (TMDB `/search/movie`) and if needed, can distinguish through features like movie release year, runtime, etc.

   We also want to do the same thing to the `watched.csv` for future cross-reference purposes and eliminate any recommendations that are movies seen but not rated.

   After generating a mapped ratings and a mapped watch dataframes, we had a matched percentage of 98.35% and 95.83%, with 0% ambiguous on both, and 1.65% and 4.17% failed mappings respectively. Upon closer inspection, this is due to the inclusion of rated TV shows that aren't part of the movie dataset, allowing to simply ignore them for this project :)

---

2. **Candidates Generation**

   The goal of generating a candidates pool is to define what movies I _could_ recommend from. Because of the size of the TMDB dataset, we won't score "all movies" but rather build a pool that we can score efficiently.

   For each movie rated ≥4.0 on my Letterboxd, I'll pull from (Set A) `/movie/{id}/recommendations` and `/movie/{id}/similar`. Additionally, I'll also pull from (Set B) `/movie/popular` as a fallback pool to round out our candidate selections. Then, I have `(A ∪ B) - Watched`, knowing that rated is subset of the watched set.

---

3. **MVP Recommender**

---

4. **Offline Recs**

---

5. **Learned Reranker**

---

6. **Systemization**
