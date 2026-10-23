"""Additional Machine Learning Lab algorithms for CineMatch.

The six practical algorithms are exposed through one small recommendation
suite. All rating-derived features are built from the ratings passed to the
constructor, so evaluation can fit this suite on training data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


MODEL_NAMES = [
    "K-NN",
    "K-Means Clustering",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "Linear Regression",
]


class LabModelSuite:
    """Train and use the six ML Lab algorithms for movie recommendations."""

    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame, random_state: int = 42):
        required_movie_cols = {"movie_id", "title", "genres"}
        required_rating_cols = {"user_id", "movie_id", "rating"}
        if not required_movie_cols.issubset(movies.columns):
            raise ValueError(f"Movies must contain columns: {sorted(required_movie_cols)}")
        if not required_rating_cols.issubset(ratings.columns):
            raise ValueError(f"Ratings must contain columns: {sorted(required_rating_cols)}")

        self.movies = movies.copy().drop_duplicates("movie_id").reset_index(drop=True)
        self.ratings = ratings.copy()
        self.ratings = self.ratings[
            self.ratings.movie_id.isin(self.movies.movie_id)
        ].copy().reset_index(drop=True)
        if self.ratings.empty:
            raise ValueError("No ratings match the provided movie dataset.")

        self.random_state = int(random_state)
        self.movie_ids = self.movies.movie_id.astype(int).to_numpy()
        self.movie_to_pos = {int(mid): i for i, mid in enumerate(self.movie_ids)}
        self.user_ids = np.sort(self.ratings.user_id.astype(int).unique())
        self.user_to_pos = {int(uid): i for i, uid in enumerate(self.user_ids)}
        self.global_mean = float(self.ratings.rating.mean())

        # MovieLens genres are represented as whitespace-separated tokens by data_utils.
        genre_tokens = set()
        for value in self.movies.genres.fillna("").astype(str):
            genre_tokens.update(value.split())
        genre_tokens.discard("Unknown")
        self.genres = sorted(genre_tokens) or ["Unknown"]
        self.genre_to_pos = {genre: i for i, genre in enumerate(self.genres)}
        self.movie_genres = np.zeros((len(self.movies), len(self.genres)), dtype=np.float32)
        for row_pos, value in enumerate(self.movies.genres.fillna("").astype(str)):
            for genre in value.split():
                genre_pos = self.genre_to_pos.get(genre)
                if genre_pos is not None:
                    self.movie_genres[row_pos, genre_pos] = 1.0

        rating_user_pos = self.ratings.user_id.astype(int).map(self.user_to_pos).to_numpy()
        rating_movie_pos = self.ratings.movie_id.astype(int).map(self.movie_to_pos).to_numpy()
        rating_values = self.ratings.rating.to_numpy(dtype=np.float32)
        rated_genres = self.movie_genres[rating_movie_pos]

        self.user_rating_sum = np.bincount(
            rating_user_pos, weights=rating_values, minlength=len(self.user_ids)
        ).astype(np.float32)
        self.user_rating_count = np.bincount(
            rating_user_pos, minlength=len(self.user_ids)
        ).astype(np.float32)
        self.user_genre_sum = np.zeros((len(self.user_ids), len(self.genres)), dtype=np.float32)
        self.user_genre_count = np.zeros_like(self.user_genre_sum)
        np.add.at(self.user_genre_sum, rating_user_pos, rated_genres * rating_values[:, None])
        np.add.at(self.user_genre_count, rating_user_pos, rated_genres)

        # A user's genre preference relative to the global mean is the clustering profile.
        user_genre_means = self._genre_means(
            self.user_genre_sum, self.user_genre_count, fill_value=self.global_mean
        )
        self.user_genre_profile = np.where(
            self.user_genre_count > 0,
            user_genre_means - self.global_mean,
            0.0,
        ).astype(np.float32)
        n_clusters = max(1, min(8, len(self.user_ids)))
        self.kmeans_scaler = StandardScaler()
        scaled_profiles = self.kmeans_scaler.fit_transform(self.user_genre_profile)
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.user_cluster_labels = self.kmeans.fit_predict(scaled_profiles)
        self._cluster_score_cache = {}

        # Item-item K-NN: represent each movie by centered ratings from its users.
        user_means = np.divide(
            self.user_rating_sum,
            self.user_rating_count,
            out=np.full_like(self.user_rating_sum, self.global_mean),
            where=self.user_rating_count > 0,
        )
        user_item = self.ratings.pivot_table(
            index="user_id", columns="movie_id", values="rating", aggfunc="mean"
        ).reindex(index=self.user_ids, columns=self.movie_ids)
        raw_user_item = user_item.to_numpy(dtype=np.float32)
        observed = ~np.isnan(raw_user_item)
        centered = np.where(observed, raw_user_item - user_means[:, None], 0.0)
        self.movie_similarity = cosine_similarity(csr_matrix(centered.T)).astype(np.float32)

        # Movie-level prior for cold/low-evidence cluster recommendations.
        item_stats = self.ratings.groupby("movie_id").rating.agg(["count", "mean"])
        self.movie_prior = np.full(len(self.movie_ids), self.global_mean, dtype=np.float32)
        for mid, stat in item_stats.iterrows():
            pos = self.movie_to_pos.get(int(mid))
            if pos is not None:
                count = float(stat["count"])
                self.movie_prior[pos] = (
                    count / (count + 10.0) * float(stat["mean"])
                    + 10.0 / (count + 10.0) * self.global_mean
                )

        # Training features are leave-one-interaction-out to avoid each row's
        # target rating appearing in that row's own user profile.
        train_x = self._pair_features(
            self.ratings.user_id.to_numpy(dtype=int),
            self.ratings.movie_id.to_numpy(dtype=int),
            exclude_ratings=rating_values,
        )
        liked = (rating_values >= 4.0).astype(np.int8)
        if len(np.unique(liked)) < 2:
            raise ValueError("The training ratings need both liked (>=4) and lower-rated examples.")

        self.decision_tree = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.random_forest = RandomForestClassifier(
            n_estimators=60,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.svm = make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=0.5,
                class_weight="balanced",
                dual=False,
                max_iter=4000,
                random_state=self.random_state,
            ),
        )
        self.linear_regression = make_pipeline(StandardScaler(), LinearRegression())

        self.decision_tree.fit(train_x, liked)
        self.random_forest.fit(train_x, liked)
        self.svm.fit(train_x, liked)
        self.linear_regression.fit(train_x, rating_values.astype(np.float32))

        self.fitted = True

    @staticmethod
    def _genre_means(sums, counts, fill_value):
        return np.divide(
            sums,
            counts,
            out=np.full_like(sums, float(fill_value), dtype=np.float32),
            where=counts > 0,
        )

    def _pair_features(self, user_ids, movie_ids, exclude_ratings=None):
        """Build user/movie features; optionally remove each row's target rating."""
        user_ids = np.asarray(user_ids, dtype=int)
        movie_ids = np.asarray(movie_ids, dtype=int)
        n_rows = len(user_ids)
        user_pos = np.asarray([self.user_to_pos.get(int(uid), -1) for uid in user_ids])
        movie_pos = np.asarray([self.movie_to_pos.get(int(mid), -1) for mid in movie_ids])

        movie_genres = np.zeros((n_rows, len(self.genres)), dtype=np.float32)
        known_movies = movie_pos >= 0
        movie_genres[known_movies] = self.movie_genres[movie_pos[known_movies]]

        user_genre_sums = np.full((n_rows, len(self.genres)), 0.0, dtype=np.float32)
        user_genre_counts = np.zeros_like(user_genre_sums)
        user_rating_sums = np.full(n_rows, self.global_mean, dtype=np.float32)
        user_rating_counts = np.zeros(n_rows, dtype=np.float32)

        known_users = user_pos >= 0
        if known_users.any():
            profile_rows = np.flatnonzero(known_users)
            user_genre_sums[profile_rows] = self.user_genre_sum[user_pos[known_users]]
            user_genre_counts[profile_rows] = self.user_genre_count[user_pos[known_users]]
            user_rating_sums[profile_rows] = self.user_rating_sum[user_pos[known_users]]
            user_rating_counts[profile_rows] = self.user_rating_count[user_pos[known_users]]

        if exclude_ratings is not None:
            row_ratings = np.asarray(exclude_ratings, dtype=np.float32)
            valid = known_users & known_movies
            rows = np.flatnonzero(valid)
            user_genre_sums[rows] -= row_ratings[rows, None] * movie_genres[rows]
            user_genre_counts[rows] -= movie_genres[rows]
            user_rating_sums[rows] -= row_ratings[rows]
            user_rating_counts[rows] -= 1.0

        genre_means = self._genre_means(
            user_genre_sums, np.maximum(user_genre_counts, 0.0), self.global_mean
        )
        user_means = np.divide(
            user_rating_sums,
            user_rating_counts,
            out=np.full(n_rows, self.global_mean, dtype=np.float32),
            where=user_rating_counts > 0,
        )
        preference_features = genre_means - self.global_mean
        count_features = np.log1p(np.maximum(user_rating_counts, 0.0))[:, None]
        return np.concatenate(
            [movie_genres, preference_features, (user_means - self.global_mean)[:, None], count_features],
            axis=1,
        ).astype(np.float32)

    def _known_user_position(self, user_id):
        return self.user_to_pos.get(int(user_id))

    def cluster_for_user(self, user_id):
        """Return the one-based cluster number for display, or None for an unknown user."""
        pos = self._known_user_position(user_id)
        if pos is None:
            return None
        return int(self.user_cluster_labels[pos]) + 1

    def _knn_rating_scores(self, user_id, n_neighbors=20):
        """Predict ratings with item-item K-NN using cosine-similar movie vectors."""
        user_rows = self.ratings[self.ratings.user_id == int(user_id)]
        if user_rows.empty:
            return self.movie_prior.copy()

        user_rows = user_rows[user_rows.movie_id.astype(int).isin(self.movie_to_pos)].copy()
        rated_positions = np.asarray(
            [self.movie_to_pos[int(mid)] for mid in user_rows.movie_id],
            dtype=int,
        )
        rated_values = user_rows.rating.to_numpy(dtype=np.float32)
        if len(rated_positions) == 0:
            return self.movie_prior.copy()
        user_mean = float(np.mean(rated_values))
        similarities = self.movie_similarity[:, rated_positions]
        k = min(int(n_neighbors), similarities.shape[1])
        if k <= 0:
            return self.movie_prior.copy()
        top = np.argpartition(similarities, -k, axis=1)[:, -k:]
        top_sim = np.take_along_axis(similarities, top, axis=1)
        weights = np.clip(top_sim, 0.0, None)
        neighbor_deviations = rated_values[top] - user_mean
        denominator = weights.sum(axis=1)
        numerator = (weights * neighbor_deviations).sum(axis=1)
        scores = np.where(
            denominator > 1e-8,
            user_mean + numerator / np.maximum(denominator, 1e-8),
            self.movie_prior,
        )
        return np.clip(scores, 1.0, 5.0).astype(np.float32)

    def _cluster_rating_scores(self, user_id):
        user_pos = self._known_user_position(user_id)
        if user_pos is None:
            return self.movie_prior.copy()
        label = int(self.user_cluster_labels[user_pos])
        if label in self._cluster_score_cache:
            return self._cluster_score_cache[label].copy()
        member_ids = self.user_ids[self.user_cluster_labels == label]
        member_ratings = self.ratings[self.ratings.user_id.isin(member_ids)]
        stats = member_ratings.groupby("movie_id").rating.agg(["count", "mean"])
        scores = self.movie_prior.copy()
        for mid, row in stats.iterrows():
            pos = self.movie_to_pos.get(int(mid))
            if pos is None:
                continue
            count = float(row["count"])
            scores[pos] = (
                count / (count + 5.0) * float(row["mean"])
                + 5.0 / (count + 5.0) * float(self.movie_prior[pos])
            )
        scores = np.clip(scores, 1.0, 5.0).astype(np.float32)
        self._cluster_score_cache[label] = scores.copy()
        return scores

    def scores_for_user(self, model_name, user_id):
        """Return a score for every known movie for one user."""
        user_id = int(user_id)
        if model_name == "K-NN":
            return self._knn_rating_scores(user_id)
        if model_name == "K-Means Clustering":
            return self._cluster_rating_scores(user_id)

        pair_x = self._pair_features(
            np.full(len(self.movie_ids), user_id, dtype=int), self.movie_ids
        )
        if model_name == "Decision Tree":
            class_pos = int(np.where(self.decision_tree.classes_ == 1)[0][0])
            return self.decision_tree.predict_proba(pair_x)[:, class_pos]
        if model_name == "Random Forest":
            class_pos = int(np.where(self.random_forest.classes_ == 1)[0][0])
            return self.random_forest.predict_proba(pair_x)[:, class_pos]
        if model_name == "SVM":
            return self.svm.decision_function(pair_x)
        if model_name == "Linear Regression":
            return np.clip(self.linear_regression.predict(pair_x), 1.0, 5.0)
        raise ValueError(f"Unknown lab model: {model_name}")

    def hybrid_scores_for_user(self, user_id):
        """Combine all six lab models into a common 0–1 ranking score.

        Each model's raw scores have different meanings (rating, probability,
        or SVM margin). Converting each score vector to percentile ranks makes
        the models comparable without pretending their raw scales match.
        """
        model_ranks = []
        for model_name in MODEL_NAMES:
            values = np.asarray(self.scores_for_user(model_name, user_id), dtype=float)
            if values.ndim != 1 or len(values) != len(self.movie_ids):
                raise ValueError(f"Unexpected score shape from {model_name}.")
            finite = np.isfinite(values)
            if not finite.any():
                ranks = np.full(len(values), 0.5, dtype=float)
            else:
                fill_value = float(np.nanmedian(values[finite]))
                safe_values = np.where(finite, values, fill_value)
                ranks = pd.Series(safe_values).rank(method="average", pct=True).to_numpy(dtype=float)
            model_ranks.append(ranks)
        return np.mean(np.vstack(model_ranks), axis=0)

    def predict_rating(self, model_name, user_id, movie_id):
        """Predict a 1–5 rating with K-NN or Linear Regression."""
        if int(movie_id) not in self.movie_to_pos:
            return float(self.global_mean)
        if model_name == "K-NN":
            scores = self._knn_rating_scores(int(user_id))
            return float(scores[self.movie_to_pos[int(movie_id)]])
        if model_name == "Linear Regression":
            features = self._pair_features([int(user_id)], [int(movie_id)])
            prediction = self.linear_regression.predict(features)[0]
            return float(np.clip(prediction, 1.0, 5.0))
        raise ValueError("Rating predictions are provided for K-NN and Linear Regression.")

    def recommendations(self, model_name, user_id, n=10):
        """Rank unseen movies for a user and return a movie DataFrame."""
        scores = np.asarray(self.scores_for_user(model_name, user_id), dtype=float)
        seen = set(
            self.ratings.loc[self.ratings.user_id == int(user_id), "movie_id"].astype(int)
        )
        eligible = np.asarray([int(mid) not in seen for mid in self.movie_ids])
        eligible_positions = np.flatnonzero(eligible)
        if len(eligible_positions) == 0:
            return pd.DataFrame(columns=list(self.movies.columns) + ["recommendation_score"])

        order = eligible_positions[np.argsort(scores[eligible_positions])[::-1][:max(int(n), 1)]]
        score_map = {int(self.movie_ids[pos]): float(scores[pos]) for pos in order}
        result = self.movies[self.movies.movie_id.astype(int).isin(score_map)].copy()
        result["recommendation_score"] = result.movie_id.astype(int).map(score_map)
        return result.sort_values("recommendation_score", ascending=False).reset_index(drop=True)
