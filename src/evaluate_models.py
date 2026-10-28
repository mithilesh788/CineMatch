"""Evaluate CineMatch recommendation baselines and ML Lab algorithms."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_utils import load_movies, load_ratings
from lab_models import LabModelSuite, MODEL_NAMES
from recommenders import CineMatchEngine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model_comparison.csv"
RATING_OUT = ROOT / "results" / "rating_prediction_metrics.csv"
K = 10
RELEVANT_RATING = 4.0


def metrics_at_k(recommended, relevant, k=K):
    """Return Precision@K, Recall@K and NDCG@K."""
    recommended = list(recommended)[:k]
    relevant = set(relevant)
    hits = [1 if movie_id in relevant else 0 for movie_id in recommended]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant) if relevant else 0.0
    dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
    idcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return precision, recall, dcg / idcg if idcg else 0.0


def chronological_holdout(ratings, test_fraction=0.20, min_ratings=5):
    """Hold out each eligible user's latest ratings without shuffling time."""
    train_parts, test_parts = [], []
    for _, group in ratings.groupby("user_id"):
        group = group.sort_values("timestamp")
        if len(group) < min_ratings:
            train_parts.append(group)
            continue
        n_test = max(1, round(len(group) * test_fraction))
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])

    train = pd.concat(train_parts, ignore_index=True)
    test = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts
        else pd.DataFrame(columns=ratings.columns)
    )
    return train, test


def main():
    movies = load_movies()
    ratings = load_ratings()
    train, test = chronological_holdout(ratings)

    if test.empty:
        raise ValueError("No test ratings available for evaluation.")

    print("Training the existing recommendation engine...", flush=True)
    engine = CineMatchEngine(movies, train)
    print("Training the six ML Lab algorithms...", flush=True)
    lab = LabModelSuite(movies, train)
    candidate_count = len(movies)

    methods = {
        "Popularity": lambda uid: engine.popularity_recommendations(candidate_count),
        "User-based CF": lambda uid: engine.user_based_recommendations(uid, candidate_count),
        "SVD CF": lambda uid: engine.svd_recommendations(uid, candidate_count),
        "Content-based": lambda uid: (
            engine.content_recommendations(
                int(train.loc[train.user_id == uid].sort_values("timestamp").iloc[-1].movie_id),
                candidate_count,
            )
            if not train.loc[train.user_id == uid].empty
            else engine.popularity_recommendations(candidate_count)
        ),
    }
    for model_name in MODEL_NAMES:
        methods[model_name] = (
            lambda uid, selected=model_name: lab.recommendations(
                selected, uid, candidate_count
            )
        )

    test_users = sorted(test.user_id.unique())
    rows = []
    print("Evaluating ranking quality for 10 models...", flush=True)

    for name, recommend in methods.items():
        user_metrics = []
        for uid in test_users:
            user_test = test[test.user_id == uid]
            relevant = set(
                user_test.loc[user_test.rating >= RELEVANT_RATING, "movie_id"].astype(int)
            )
            if not relevant:
                continue

            seen = set(train.loc[train.user_id == uid, "movie_id"].astype(int))
            recs = recommend(int(uid))
            if recs.empty:
                recommended_ids = []
            else:
                recommended_ids = (
                    recs.loc[~recs.movie_id.astype(int).isin(seen)]
                    .movie_id.astype(int)
                    .drop_duplicates()
                    .head(K)
                    .tolist()
                )

            user_metrics.append(metrics_at_k(recommended_ids, relevant, K))

        if user_metrics:
            average = np.mean(user_metrics, axis=0)
            rows.append({
                "model": name,
                "precision_at_10": average[0],
                "recall_at_10": average[1],
                "ndcg_at_10": average[2],
            })
        print(f"  Finished: {name}", flush=True)

    results = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)

    # Rating error is meaningful for methods that produce estimated ratings.
    # Classification scores are deliberately not mislabeled as predicted ratings.
    rating_models = ["User-based CF", "SVD CF", "K-NN", "Linear Regression"]
    actual_values = []
    prediction_values = {name: [] for name in rating_models}
    movie_pos = {int(mid): pos for pos, mid in enumerate(lab.movie_ids)}
    test_groups = {int(uid): group for uid, group in test.groupby("user_id")}
    print("Calculating rating-prediction errors...", flush=True)

    for uid, user_test in test_groups.items():
        # Produce each baseline prediction vector once per user.
        user_cf = engine.user_based_recommendations(uid, candidate_count)
        svd_cf = engine.svd_recommendations(uid, candidate_count)
        user_cf_scores = dict(zip(user_cf.movie_id.astype(int), user_cf.recommendation_score.astype(float)))
        svd_cf_scores = dict(zip(svd_cf.movie_id.astype(int), svd_cf.recommendation_score.astype(float)))
        knn_scores = lab.scores_for_user("K-NN", uid)
        lr_scores = lab.scores_for_user("Linear Regression", uid)

        for row in user_test.itertuples(index=False):
            mid = int(row.movie_id)
            pos = movie_pos.get(mid)
            if pos is None:
                continue
            actual_values.append(float(row.rating))
            prediction_values["User-based CF"].append(
                float(np.clip(user_cf_scores.get(mid, engine.global_mean), 1.0, 5.0))
            )
            prediction_values["SVD CF"].append(
                float(np.clip(svd_cf_scores.get(mid, engine.global_mean), 1.0, 5.0))
            )
            prediction_values["K-NN"].append(float(knn_scores[pos]))
            prediction_values["Linear Regression"].append(float(lr_scores[pos]))

    rating_rows = []
    actual = np.asarray(actual_values, dtype=float)
    for name in rating_models:
        predicted = np.asarray(prediction_values[name], dtype=float)
        if len(predicted) == len(actual) and len(actual):
            rating_rows.append({
                "model": name,
                "mae": mean_absolute_error(actual, predicted),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "test_ratings": len(actual),
            })
    rating_results = pd.DataFrame(rating_rows)
    rating_results.to_csv(RATING_OUT, index=False)

    print("\nEvaluation protocol")
    print(f"Training ratings: {len(train):,}")
    print(f"Test ratings: {len(test):,}")
    print(f"Users with test ratings: {test.user_id.nunique():,}")
    print(f"Relevant rating threshold: >= {RELEVANT_RATING}")
    print(f"Ranking cutoff: {K}")
    print("\nRanking model comparison")
    print(results.to_string(index=False))
    print(f"\nSaved: {OUT}")
    print("\nRating prediction errors (lower is better)")
    print(rating_results.to_string(index=False))
    print(f"\nSaved: {RATING_OUT}")
    print(
        "\nNote: model settings are initial practical baselines; perform and document "
        "additional tuning experiments before describing them as tuned models."
    )


if __name__ == "__main__":
    main()
