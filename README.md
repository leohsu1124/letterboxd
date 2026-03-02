## Next Watch (w/ Letterboxd)

This project was inspired by my curiousity of recommendation systems ever since I learned about them in a machine learning course at CMU. It's that and another great excuse to watch more movies!

Since the process for getting a [Letterboxd API](https://api-docs.letterboxd.com/) rather tedious, I opted to use a _(new-ish??)_ extract personal data feature.

To ensure I had a grasp on the data extracted, I made a time-lapse animation based on my own ratings (423 movies / 1,104 watched at time of writing) inspired by [@dado3212](https://github.com/dado3212/letterboxd-scripts/tree/main).

![animation](/animation/animation.gif)

---

1. **Data Foundations**

   Started off by importing my `ratings.csv` into a table in order to build an ID resolution step & map each rated movie to its `tmdb_id` in the [TMDB](https://www.themoviedb.org/) dataset. We can utilize the [TMDB API](https://developer.themoviedb.org/docs/getting-started) to implement the mapping with a `title + year` search (TMDB `/search/movie`) and if needed, can distinguish through features like movie release year, runtime, etc. I personally opted for the [TMDBSimple](https://github.com/celiao/tmdbsimple) wrapper for convenience.

   We also want to do the same thing to the `watched.csv` for future cross-reference purposes and eliminate any recommendations that are movies seen but not rated.

   After generating a mapped ratings and a mapped watch dataframes, we had a matched percentage of 98.35% and 95.83%, with 0% ambiguous on both, and 1.65% and 4.17% failed mappings respectively. Upon closer inspection, this is due to the inclusion of rated TV shows that aren't part of the movie dataset, allowing to simply ignore them for this project :)

---

2.  **Candidates Generation**

    The goal of generating a candidates pool is to define what movies I _could_ recommend from. Because of the size of the TMDB dataset, we won't score "all movies" but rather build a pool that we can score efficiently.

    For each movie rated ≥4.0 on my Letterboxd, I'll pull from (Set A) `/movie/{id}/recommendations` and `/movie/{id}/similar`. Additionally, I'll also pull from (Set B) `/movie/popular` as a fallback pool to round out our candidate selections. Then, I have `(A ∪ B) - Watched`, knowing that rated is subset of the watched set.

    However, after fine-tuning some parameters, I could only get a candidate pool of about ~3200 movies. Hoping for more, I implemented a discover function, where it should introduce slight entropy in the system. The following are the different discovery avenues included:

    >

        {"name": "mid_depth_votes", "sort_by": "vote_count.desc", "vote_count_gte": 200, "page_start": 30, "pages": 75},
        {"name": "high_quality", "sort_by": "vote_average.desc", "vote_count_gte": 2000, "page_start": 1, "pages": 100},
        {"name": "recent_releases", "sort_by": "primary_release_date.desc", "vote_count_gte": 50, "page_start": 1, "pages": 25},

    This ensures that we discover movies with the most recent release dates, the most highly voted movies on average, and also movies with most vote counts. However, you'll notice that for the vote_counts, we started at page 30 as we realized the pages before that often had tons of overlap.

    Now, we have a candidates dataframe that has about ~8850 movies. A good sanity check is that the movies aren't all from the same genre or just franchise sequels/remakes.

---

3. **MVP Recommender (No Training)**

   The goal for the MVP Recommender is to produce _immediate_ good Top-K (`k = 10`) recommendations with explainability. In order to do so, we can fetch the top 5 cast members, director, and keywords for all candidates and liked movies set (movies rated ≥4.0) from TMDB. Then, create a preference sparse feature matrix by stacking features genres (3x weight), director (2x), keywords (1x), cast (0.5x), and [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) on overview (0.3x).

   From each candidate, we'll score via cosine similarity against liked set (mean over top-20), add a log-popularity prior (5%), and [MMR](https://arxiv.org/html/2503.13881v1) reranks to ensure diversity. Personally, I threw in a genre cap of 3 also for diversity and a superhero cap discriminator to downplay my bias for more comic-related movies within the liked set. Filters out candidates with fewer than 300 votes. Adding explanation generations by finding which movies in liked set drove the recommendations and what features they might share (ie. _"Because you liked X and Y / directed by Z / features A, B"_)

---

4. **Offline Recs**
   ![plot](/pics/sweep_progress.png)
   ![plot](/pics/keyword_weight.png)
   ![plot](/pics/director_keyword_heatmap.png)

---

5. **Learned Reranker**

---

6. **Systemization**
