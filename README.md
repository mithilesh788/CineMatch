# CineMatch — Movie Recommendation System

CineMatch is a Streamlit movie-discovery app built around the MovieLens 100K dataset and TMDb movie posters/catalogue.

## Existing recommendation features

- Popularity-based recommendations
- Content-based filtering using TF-IDF and cosine similarity
- User-based collaborative filtering
- SVD-based collaborative filtering
- Mood-based discovery and description search
- Group movie-night recommendations and hidden-gem discovery
- Cinema Journal and live Indian/international catalogue
- Model Analytics with ranking-metric tables/charts and rating-prediction errors

## Machine Learning Lab algorithms integrated into the recommender

The six algorithms are part of the existing recommendation workflow; there is no separate Lab Models tab:

- **K-NN:** estimates personalized movie scores using similar rating patterns.
- **K-Means:** groups users by genre-rating preferences and contributes cluster-based rankings.
- **Decision Tree:** predicts whether a user is likely to like a movie.
- **Random Forest:** combines trees to estimate like likelihood.
- **SVM:** contributes a liked/not-liked preference ranking.
- **Linear Regression:** predicts expected ratings on the 1–5 scale.

CineMatch combines the six algorithms by converting each model's scores to percentile ranks, then averaging those ranks. Discover, Mood, and Describe It blend this shared ML ranking with their existing content, mood, or text relevance. The existing Model Analytics tab compares all ten models individually; rating predictors are also compared by MAE and RMSE. TMDb catalogue features (Watch Party, Hidden Treasures, and Indian & New) continue to use TMDb because those live catalogue items do not have MovieLens rating rows for these models to predict.

These are initial practical baselines, not exhaustively tuned models. Run and document tuning experiments before describing them as tuned or state-of-the-art.

## Setup on macOS

```bash
cd ~/Desktop/CineMatch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/download_data.py
python src/evaluate_models.py
streamlit run app.py
```

The evaluation script creates/updates:

- `results/model_comparison.csv` — ten-model ranking comparison
- `results/rating_prediction_metrics.csv` — MAE and RMSE for rating predictors

## TMDb configuration

Do not put API keys directly in source code. Configure `TMDB_API_KEY` through an environment variable or `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "YOUR_OWN_TMDB_KEY"
```

Keep `.streamlit/secrets.toml`, `.env`, local datasets, `.venv`, `cinema_journal.json`, and backup files out of public repositories. The provided ZIP intentionally omits the dataset and local secrets; run `python src/download_data.py` to download MovieLens 100K again.

## Evaluation protocol

- Split each eligible user's ratings chronologically, using the latest 20% as test ratings and earlier ratings as training data.
- Treat test ratings of 4 or higher as relevant for ranking metrics.
- Exclude movies already rated in that user's training data from recommendation lists.
- Fit rating-derived models on training data only during evaluation.
- Use the rating-error output for actual rating estimates; classification scores are not mislabeled as predicted ratings.

Review the MovieLens dataset notes before redistribution. Do not invent or pre-fill model scores in the report; regenerate metrics from the dataset before submission.
