from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "ml-100k"

GENRES = [
    "Unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

def load_movies():
    path = DATA_DIR / "u.item"
    if not path.exists():
        raise FileNotFoundError("Dataset missing. Run: python src/download_data.py")
    cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + [f"genre_{i}" for i in range(19)]
    movies = pd.read_csv(path, sep="|", encoding="latin-1", header=None, names=cols)
    genre_cols = [f"genre_{i}" for i in range(19)]
    movies["genres"] = movies.apply(
        lambda r: " ".join(GENRES[i] for i, col in enumerate(genre_cols) if r[col] == 1),
        axis=1
    )
    movies["genres"] = movies["genres"].replace("", "Unknown")
    movies["title"] = movies["title"].fillna("Unknown title").astype(str)
    return movies.drop_duplicates("movie_id")[["movie_id", "title", "release_date", "genres"]].reset_index(drop=True)

def load_ratings():
    path = DATA_DIR / "u.data"
    if not path.exists():
        raise FileNotFoundError("Dataset missing. Run: python src/download_data.py")
    ratings = pd.read_csv(
    path,
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
    engine="python",
)
    ratings = ratings.dropna(subset=["user_id", "movie_id", "rating"])
    ratings = ratings.drop_duplicates(["user_id", "movie_id"], keep="last")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating"])
    ratings = ratings[ratings["rating"].between(1, 5)]
    ratings[["user_id", "movie_id"]] = ratings[["user_id", "movie_id"]].astype(int)
    return ratings.reset_index(drop=True)
