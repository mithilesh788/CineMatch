import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

MOOD_GENRES = {
    "Happy": ["Comedy", "Animation", "Musical"],
    "Relaxed": ["Comedy", "Documentary", "Romance"],
    "Romantic": ["Romance", "Drama", "Musical"],
    "Adventurous": ["Adventure", "Action", "Fantasy", "Sci-Fi"],
    "Thriller / Suspense": ["Thriller", "Mystery", "Crime"],
    "Thought-provoking": ["Drama", "Documentary", "Sci-Fi"],
    "Nostalgic": ["Drama", "Comedy", "Romance"],
}

class CineMatchEngine:
    def __init__(self, movies, ratings):
        self.movies = movies.copy().reset_index(drop=True)
        self.ratings = ratings.copy()
        self.movie_ids = self.movies.movie_id.astype(int).tolist()
        self.popularity = ratings.groupby("movie_id").rating.agg(["count", "mean"])
        self.global_mean = float(ratings.rating.mean())
        self.popularity["score"] = (self.popularity["count"] / (self.popularity["count"] + 10) * self.popularity["mean"] + 10 / (self.popularity["count"] + 10) * self.global_mean)
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.content = self.tfidf.fit_transform((self.movies.title.fillna("") + " " + self.movies.genres.fillna("")).tolist())
        self.user_item = ratings.pivot_table(index="user_id", columns="movie_id", values="rating", aggfunc="mean").reindex(columns=self.movie_ids).fillna(0)
        self.user_ids = self.user_item.index.tolist()
        self.user_similarity = None
        self.svd_scores = self._fit_svd()

    def _fit_svd(self):
        a = self.user_item.to_numpy(dtype=float)
        counts = (a != 0).sum(axis=1)
        means = np.divide(a.sum(axis=1), counts, out=np.zeros(a.shape[0]), where=counts != 0)
        centered = a.copy()
        for i in range(len(centered)):
            centered[i, centered[i] != 0] -= means[i]
        if min(centered.shape) < 3:
            return a
        k = min(30, min(centered.shape) - 1)
        u, s, vt = svds(csr_matrix(centered), k=k)
        return u @ np.diag(s) @ vt + means[:, None]

    def _format(self, scores, n=10, exclude=None):
        exclude = set(int(x) for x in (exclude or []))
        pairs = sorted(((int(mid), float(score)) for mid, score in scores.items() if int(mid) not in exclude), key=lambda x: x[1], reverse=True)
        chosen = pairs[:n]
        score_map = dict(chosen)
        out = self.movies[self.movies.movie_id.isin(score_map)].copy()
        out["recommendation_score"] = out.movie_id.map(score_map)
        return out.sort_values("recommendation_score", ascending=False).reset_index(drop=True)

    def popularity_recommendations(self, n=10):
        return self._format(self.popularity.score.to_dict(), n)

    def content_recommendations(self, movie_id, n=10):
        idxs = self.movies.index[self.movies.movie_id == int(movie_id)]
        if len(idxs) == 0:
            return pd.DataFrame()
        sims = cosine_similarity(self.content[idxs[0]], self.content).ravel()
        return self._format(dict(zip(self.movie_ids, sims)), n, [int(movie_id)])

    def user_based_recommendations(self, user_id, n=10):
        if int(user_id) not in self.user_item.index:
            return self.popularity_recommendations(n)
        matrix = self.user_item.to_numpy(dtype=float)
        sims = cosine_similarity(self.user_item.loc[[int(user_id)]].to_numpy(), matrix).ravel()
        pos = self.user_item.index.get_loc(int(user_id))
        sims[pos] = 0
        neighbors = np.argsort(sims)[::-1][:30]
        nr = matrix[neighbors]
        weights = sims[neighbors]
        denom = (nr != 0).T @ np.abs(weights)
        numer = nr.T @ weights
        pred = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)
        seen = self.ratings.loc[self.ratings.user_id == int(user_id), "movie_id"].astype(int).tolist()
        return self._format(dict(zip(self.movie_ids, pred)), n, seen)

    def svd_recommendations(self, user_id, n=10):
        if int(user_id) not in self.user_item.index:
            return self.popularity_recommendations(n)
        pos = self.user_item.index.get_loc(int(user_id))
        seen = self.ratings.loc[self.ratings.user_id == int(user_id), "movie_id"].astype(int).tolist()
        return self._format(dict(zip(self.movie_ids, self.svd_scores[pos])), n, seen)

    def mood_recommendations(self, mood, genres=None, n=10):
        target = set(MOOD_GENRES.get(mood, [])) | set(genres or [])
        df = self.movies.copy()
        df["mood_score"] = df.genres.apply(lambda x: len(set(str(x).split()) & target) / max(1, len(target)))
        pop = self.popularity["score"]
        df["pop_score"] = df.movie_id.map(pop).fillna(self.global_mean) / 5
        df["recommendation_score"] = .7 * df.mood_score + .3 * df.pop_score
        return df.sort_values("recommendation_score", ascending=False).head(n).reset_index(drop=True)

    def description_search(self, query, n=10):
        if not str(query).strip():
            return pd.DataFrame()
        q = self.tfidf.transform([query])
        sims = cosine_similarity(q, self.content).ravel()
        return self._format(dict(zip(self.movie_ids, sims)), n)

    @staticmethod
    def _percentile_scores(values):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).any():
            return np.full(len(values), 0.5, dtype=float)
        safe = np.where(np.isfinite(values), values, np.nanmedian(values[np.isfinite(values)]))
        return pd.Series(safe).rank(method="average", pct=True).to_numpy(dtype=float)

    def ml_hybrid_recommendations(
        self,
        user_id,
        lab_suite,
        n=10,
        seed_movie_id=None,
        query=None,
        mood=None,
        genres=None,
    ):
        """Recommend from the six-algorithm ensemble, optionally adding context.

        K-NN, K-Means, Decision Tree, Random Forest, SVM, and Linear
        Regression are combined using their percentile rankings. For the
        existing Discover, Describe It, and Mood workflows, contextual
        similarity contributes to the final ranking rather than bypassing
        the ML ensemble.
        """
        user_id = int(user_id)
        if user_id not in getattr(lab_suite, "user_to_pos", {}):
            return self.popularity_recommendations(n)

        suite_scores = lab_suite.hybrid_scores_for_user(user_id)
        suite_ids = [int(mid) for mid in lab_suite.movie_ids]
        score_map = dict(zip(suite_ids, np.asarray(suite_scores, dtype=float)))
        ml_scores = np.asarray([score_map.get(int(mid), 0.5) for mid in self.movie_ids], dtype=float)
        ml_rank = self._percentile_scores(ml_scores)
        final_scores = ml_rank.copy()

        # Discover: balance the six-model ensemble with similarity to the selected movie.
        if seed_movie_id is not None:
            positions = self.movies.index[self.movies.movie_id.astype(int) == int(seed_movie_id)]
            if len(positions):
                similarities = cosine_similarity(self.content[positions[0]], self.content).ravel()
                final_scores = 0.60 * ml_rank + 0.40 * self._percentile_scores(similarities)

        # Describe It: use the existing metadata text match alongside personalized ML ranking.
        if query is not None and str(query).strip():
            query_vector = self.tfidf.transform([str(query)])
            similarities = cosine_similarity(query_vector, self.content).ravel()
            final_scores = 0.60 * ml_rank + 0.40 * self._percentile_scores(similarities)

        # Mood: the ensemble remains the main signal; genre/mood fit steers the result.
        if mood is not None:
            target = set(MOOD_GENRES.get(str(mood), [])) | set(genres or [])
            if target:
                mood_fit = np.asarray([
                    len(set(str(value).split()) & target) / max(1, len(target))
                    for value in self.movies.genres.fillna("")
                ], dtype=float)
                final_scores = 0.60 * ml_rank + 0.40 * self._percentile_scores(mood_fit)

        seen = set(
            self.ratings.loc[self.ratings.user_id.astype(int) == user_id, "movie_id"].astype(int).tolist()
        )
        if seed_movie_id is not None:
            seen.add(int(seed_movie_id))
        return self._format(dict(zip(self.movie_ids, final_scores)), n, seen)

    def hidden_gems(self, n=10, min_votes=5):
        df = self.movies.copy()
        df["rating_count"] = df.movie_id.map(self.popularity["count"]).fillna(0)
        df["bayesian_score"] = df.movie_id.map(self.popularity["score"]).fillna(0)
        df = df[df.rating_count >= min_votes].copy()
        if df.empty:
            return self.popularity_recommendations(n)
        df["recommendation_score"] = df.bayesian_score - .12 * np.log1p(df.rating_count)
        return df.sort_values("recommendation_score", ascending=False).head(n).reset_index(drop=True)

    def group_recommendations(self, user_ids, n=10):
        valid = [int(u) for u in user_ids if int(u) in self.user_item.index]
        if not valid:
            return self.popularity_recommendations(n)
        a = self.user_item.loc[valid].to_numpy(dtype=float)
        counts = (a != 0).sum(axis=0)
        avg = np.divide(a.sum(axis=0), counts, out=np.zeros(a.shape[1]), where=counts != 0)
        pop = pd.Series(self.popularity.score).reindex(self.movie_ids).fillna(self.global_mean).to_numpy()
        scores = np.where(counts > 0, avg, pop)
        seen = set()
        for u in valid:
            seen |= set(self.ratings.loc[self.ratings.user_id == u, "movie_id"].astype(int))
        return self._format(dict(zip(self.movie_ids, scores)), n, seen)
