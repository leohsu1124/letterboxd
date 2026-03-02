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

   **Generated Top-10 Recommendations (w/ MVP)**

   ![recs](/pics/mvp_recommender_top10recs.png)

---

4.  **Offline Recs**

    The goal is to build a rigorous offline evaluation to compare recommender configs and fine-tune feature weights to be confident recommendations are _quantitatively_ better in this model.

    Since ratings are sorted chronologically, our train/test split is a 80%/20% split of the liked set. By construction, tests postives aren't built into the candidate pool as it excludes all seen movies by default. The scoring builds a liked feature matrix from train-only liked movies, scoring all candidates (including injected test movies) using the scoring pipeline.

    We're judging on the metrics of:
    - **nDCG@k:** Normalized Discounted Cumulative Gain: Are test positives ranked high?
    - **Recall@k:** What fraction of test positives in the top-k?
    - **Hit Rate@k:** Did at least one test positive make the top-k?
    - **Catalog Coverage@k:** What fraction of the candidate pool appears in the top 100?
    - **Novelty@k:** Mean self-information of recommendations (higher = less popular)

    ![plot](/pics/sweep_progress.png)

    Fine-tuning are done through progressive sweeps, a grid search over feature weights and train fractions. Supports `--sweep` from CLI. Testing roughly 500 configs, we found that `director_weight` played a significant role in breakthroughs for our metrics whereas `cast_weight` was irrelevant, thus defaulting at 0.0. `keyword_weight` ≥ 2.0 tripled our Recall@50 compared to when `keyword_weight = 1.0`. Further investigation revealed a tradeoff, nDCG saturates at `keyword_weight = 5.0` while Recall peaks at `keyword_weight = 3.0`.

    ![plot](/pics/keyword_weight.png)

    Because of the trade-off, we performed an additional sweep altering director and keyword weighting while `cast_weight = 0.0`.

    ![plot](/pics/director_keyword_heatmap.png)

    The final configs:

    >

          cast_weight = 0.0, director_weight = 10.0, keyword_weight = 3.0, genre_weight = 3.0, tfidf_weight = 0.3

    **How Does It Translate for Our Recommendations**

    Our evaluation reveals that preferences are driven by thematic content (keywords) and who directed the film, regardless of the actors appearing. The original (Step 3) recommendations biases are often determined along the lines of _"You liked Marvel so here's more Marvel."_

    With tuned weights, recommendations and the overall model favors thinking like _"You liked Parasite (2019) so here's more films about class struggle."_ over _"You liked a film starring Brad Pitt so here's more films with Brad Pitt."_

    However, a practical aspect worth noting is that our final configs produces Recall@50 = 0.161, meaning the system would surface roughly 5 out of 31 future favorites in a top-50 list. Certain movies, classics like _Jaws_ and _Lord of the Rings_ or less-known discoveries like _Godland_, are inherently hard ot predict from prior viewing patterns. The final configs was also determined with this in mind as we didn't want to overfit.

    **Generated Top-10 Recommendations (w/ Final Configs)**

    ![recs](/pics/offline_val_top10recs.png)

---

5. **Learned Reranker**
