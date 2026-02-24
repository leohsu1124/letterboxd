## Next Watch (w/ Leo's Letterboxd)

Since the process for getting a [Letterboxd API](https://api-docs.letterboxd.com/) rather tedious, I opted to use a _(new-ish??)_ extract personal data feature.

To ensure I had a grasp on the data extracted, I made a time-lapse animation based on my own ratings (423 movies / 1,104 watched at time of writing).

![animation](/animation/animation.gif)

---

1. Data Foundations \

   Started off by importing my `ratings.csv` into a table in order to build an ID resolution step & map each rated movie to its `tmdb_id` in the [TMDB](https://www.themoviedb.org/) dataset. We can utilize the [TMDB API](https://developer.themoviedb.org/docs/getting-started) to implement the mapping with a `title + year` search (TMDB `/search/movie`) and if needed, can distinguish through features like movie release year, runtime, etc.

---

- PHASE 2: MOVIE CANDIDATES
- PHASE 3: MVP RECOMMENDER
- PHASE 4: OFFLINE RECCS
- PHASE 5: LEARNED RERANKER
- PHASE 6: SYSTEMIZATION
