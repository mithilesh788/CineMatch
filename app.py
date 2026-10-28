
from pathlib import Path
import os
import re
import sys
import html
import logging
from difflib import SequenceMatcher

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_movies, load_ratings
from recommenders import CineMatchEngine, MOOD_GENRES
from lab_models import LabModelSuite, MODEL_NAMES

st.set_page_config(
    page_title="CineMatch | Discover Cinema",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

OMDB_API_KEY = os.getenv("OMDB_API_KEY", "").strip()

def load_tmdb_api_key_default():
    """Load the TMDb key from environment variables or Streamlit secrets.

    The app can still accept a key manually in the Indian & New tab when
    no configured key is available. Never hardcode credentials in source.
    """
    key = os.getenv("TMDB_API_KEY", "").strip()
    if key:
        return key

    try:
        return str(st.secrets.get("TMDB_API_KEY", "")).strip()
    except Exception:
        # Streamlit secrets may not be configured during local development.
        return ""


TMDB_API_KEY_DEFAULT = load_tmdb_api_key_default()

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
POSTER_LOGGER = logging.getLogger("cinematch.poster")


# ==================================================
# 2. CINEMATIC THEME
# ==================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #09090d;
        --panel: #15151c;
        --panel2: #1c1c25;
        --border: #30303b;
        --text: #f5f5f7;
        --muted: #aaaab7;
        --red: #e50914;
        --red-light: #ff424c;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 50% -12%,
                rgba(112, 12, 25, .35),
                transparent 48%
            ),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, p, label {
        color: var(--text);
    }

    h1 {
        font-size: clamp(2.2rem, 5vw, 4rem) !important;
        font-weight: 850 !important;
        letter-spacing: -1.8px;
    }

    h2 {
        font-size: 1.65rem !important;
        font-weight: 780 !important;
        letter-spacing: -.5px;
    }

    h3 {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
    }

    .cine-hero {
        position: relative;
        overflow: hidden;
        padding: clamp(28px, 5vw, 66px);
        margin: 8px 0 30px 0;
        border: 1px solid #51222b;
        border-radius: 24px;
        background:
            radial-gradient(
                ellipse at 90% 15%,
                rgba(229, 9, 20, .40),
                transparent 42%
            ),
            linear-gradient(
                115deg,
                #281016 0%,
                #191016 52%,
                #101015 100%
            );
        box-shadow: 0 18px 60px rgba(0, 0, 0, .28);
    }

    .cine-eyebrow {
        color: #ff6972;
        font-size: .75rem;
        font-weight: 850;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .cine-hero h1 {
        margin: 0;
        line-height: 1.08;
    }

    .cine-hero p {
        max-width: 680px;
        color: #d0c5ca;
        font-size: 1rem;
        line-height: 1.8;
        margin-top: 16px;
    }

    .hero-pill {
        display: inline-block;
        margin: 16px 7px 0 0;
        padding: 7px 12px;
        border: 1px solid #60303a;
        border-radius: 30px;
        color: #f5dfe2;
        background: rgba(255, 255, 255, .04);
        font-size: .78rem;
    }

    .stButton > button {
        min-height: 44px;
        border: 1px solid #ed252e;
        border-radius: 10px;
        background: var(--red);
        color: white;
        font-weight: 750;
        transition: all .18s ease;
    }

    .stButton > button:hover {
        background: var(--red-light);
        border-color: var(--red-light);
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 5px 20px rgba(229, 9, 20, .22);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: #17171f;
        border-color: #353541;
        border-radius: 10px;
    }

    textarea, input {
        color: var(--text) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 14px;
        background: transparent;
        border-radius: 9px 9px 0 0;
        color: #b7b7c2;
        font-weight: 650;
    }

    .stTabs [aria-selected="true"] {
        color: white !important;
        border-bottom: 3px solid var(--red) !important;
    }

    .movie-card {
        min-height: 125px;
        padding: 15px;
        margin: 0 0 12px 0;
        border: 1px solid var(--border);
        border-radius: 13px;
        background: linear-gradient(145deg, #1b1b23, #111116);
        transition: transform .18s ease, border-color .18s ease;
    }

    .movie-card:hover {
        transform: translateY(-2px);
        border-color: #8a3039;
    }

    .movie-title {
        color: #fff;
        font-size: 1rem;
        line-height: 1.4;
        font-weight: 750;
        margin-bottom: 8px;
        overflow-wrap: anywhere;
    }

    .movie-genres {
        color: #b5b5c2;
        font-size: .78rem;
        line-height: 1.6;
    }

    .movie-score {
        display: inline-block;
        margin-top: 12px;
        color: #ff727a;
        font-size: .78rem;
        font-weight: 750;
    }

    .poster-fallback {
        display: flex;
        align-items: center;
        justify-content: center;
        aspect-ratio: 2 / 3;
        min-height: 160px;
        padding: 12px;
        border: 1px solid #47232b;
        border-radius: 10px;
        background:
            radial-gradient(
                ellipse at 50% 20%,
                #66202c,
                transparent 65%
            ),
            linear-gradient(145deg, #24202a, #101015);
        text-align: center;
        color: #f7dfe2;
        font-size: .9rem;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .section-eyebrow {
        color: #ff6972;
        font-size: .72rem;
        font-weight: 850;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    [data-testid="stMetric"] {
        padding: 16px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--panel);
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: var(--border);
    }

    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 3rem 1rem;
        }

        .cine-hero {
            padding: 26px 20px;
            border-radius: 17px;
        }

        h1 {
            letter-spacing: -1px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0 8px;
            font-size: .8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 3. LOAD MOVIELENS DATA
# ==================================================

@st.cache_data
def load_data():
    movies = load_movies()
    ratings = load_ratings()
    return movies, ratings


@st.cache_resource
def get_engine():
    movies, ratings = load_data()
    return CineMatchEngine(movies, ratings)


@st.cache_resource(show_spinner=False)
def get_lab_model_suite():
    """Train and cache the six-model ensemble used by core recommendation tabs."""
    local_movies, local_ratings = load_data()
    return LabModelSuite(local_movies, local_ratings)


# ==================================================
# 4. MOVIELENS POSTERS AND MOVIE CARDS
# ==================================================

def extract_year(title):
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return match.group(1) if match else None




@st.cache_data(ttl=86400, show_spinner=False)
def get_poster(title, year=None, api_key=""):
    """Find a movie poster using TMDb first and OMDb as fallback."""

    clean_title = re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        str(title),
    ).strip()

    if not clean_title:
        return None

    # Search TMDb using the movie title.
    if api_key:
        search_params = {
            "query": clean_title,
            "include_adult": "false",
            "language": "en-US",
            "page": 1,
            "api_key": api_key,
        }

        # First search with the release year.
        if year:
            try:
                response = requests.get(
                    f"{TMDB_BASE_URL}/search/movie",
                    params={
                        **search_params,
                        "primary_release_year": str(year),
                    },
                    timeout=8,
                )

                response.raise_for_status()
                results = response.json().get("results", [])

                poster = find_best_tmdb_poster(
                    results,
                    clean_title,
                )

                if poster:
                    return f"{TMDB_IMAGE_BASE}{poster}"

                POSTER_LOGGER.warning(
                    "TMDb poster not matched: title=%r year=%r query=%r candidates=%s",
                    clean_title,
                    year,
                    clean_title,
                    [
                        {
                            "id": item.get("id"),
                            "title": item.get("title"),
                            "original_title": item.get("original_title"),
                            "has_poster": bool(item.get("poster_path")),
                        }
                        for item in results[:10]
                    ],
                )

            except (requests.RequestException, ValueError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                POSTER_LOGGER.warning(
                    "TMDb poster request failed: title=%r year=%r query=%r error=%s status=%s",
                    clean_title,
                    year,
                    clean_title,
                    type(exc).__name__,
                    status,
                )

        # Retry without the release year.
        try:
            response = requests.get(
                f"{TMDB_BASE_URL}/search/movie",
                params=search_params,
                timeout=8,
            )

            response.raise_for_status()
            results = response.json().get("results", [])

            poster = find_best_tmdb_poster(
                results,
                clean_title,
            )

            if poster:
                return f"{TMDB_IMAGE_BASE}{poster}"

            POSTER_LOGGER.warning(
                "TMDb poster not matched: title=%r year=%r query=%r candidates=%s",
                clean_title,
                year,
                clean_title,
                [
                    {
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "original_title": item.get("original_title"),
                        "has_poster": bool(item.get("poster_path")),
                    }
                    for item in results[:10]
                ],
            )

        except (requests.RequestException, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            POSTER_LOGGER.warning(
                "TMDb poster request failed: title=%r year=%r query=%r error=%s status=%s",
                clean_title,
                year,
                clean_title,
                type(exc).__name__,
                status,
            )

    # Try OMDb if TMDb did not return a poster.
    if OMDB_API_KEY:
        params = {
            "apikey": OMDB_API_KEY,
            "t": clean_title,
        }

        if year:
            params["y"] = str(year)

        try:
            response = requests.get(
                "https://www.omdbapi.com/",
                params=params,
                timeout=5,
            )

            response.raise_for_status()
            data = response.json()

            poster = data.get("Poster")

            if (
                data.get("Response") == "True"
                and poster
                and poster != "N/A"
            ):
                return poster

        except (requests.RequestException, ValueError):
            pass

    POSTER_LOGGER.warning(
        "No poster found: title=%r year=%r tmdb_key_configured=%s omdb_key_configured=%s",
        clean_title,
        year,
        bool(api_key),
        bool(OMDB_API_KEY),
    )
    return None


def find_best_tmdb_poster(results, requested_title):
    """Choose a close title match, including punctuation/article variations."""

    def normalize(value):
        value = str(value or "").casefold()
        # MovieLens may put articles at the end: "Alien, The".
        value = re.sub(r",\s*(the|a|an)$", "", value)
        value = re.sub(r"[^a-z0-9]+", " ", value)
        value = " ".join(value.split())
        return re.sub(r"^(the|a|an)\s+", "", value)

    wanted = normalize(requested_title)
    if not wanted:
        return None

    candidates = []
    for movie in results:
        poster = movie.get("poster_path")
        if not poster:
            continue
        titles = {
            normalize(movie.get("title")),
            normalize(movie.get("original_title")),
        } - {""}
        if not titles:
            continue

        # Exact matches win. Similarity catches minor punctuation, subtitle,
        # and transliteration differences without accepting unrelated titles.
        similarity = max(SequenceMatcher(None, wanted, title).ratio() for title in titles)
        if wanted in titles:
            similarity = 1.0
        elif any(wanted in title or title in wanted for title in titles):
            similarity = max(similarity, 0.88)
        if similarity >= 0.72:
            candidates.append((similarity, movie))

    if not candidates:
        return None

    candidates.sort(
        key=lambda pair: (
            pair[0],
            int(bool(pair[1].get("release_date"))),
            int(pair[1].get("vote_count") or 0),
        ),
        reverse=True,
    )
    return candidates[0][1]["poster_path"]




def show_movies(df, heading=None, limit=12, score_label="Match score"):
    if heading:
        st.markdown(
            f'<div class="section-eyebrow">'
            f'{html.escape(str(heading))}</div>',
            unsafe_allow_html=True,
        )

    if df is None or df.empty:
        st.info("No recommendations found. Try another preference.")
        return

    results = df.head(limit)
    cols = st.columns(4, gap="medium")

    for index, (_, row) in enumerate(results.iterrows()):
        title = str(row.get("title", "Unknown title"))
        genres = str(row.get("genres", "Genres unavailable"))
        year = extract_year(title)
        poster_url = row.get("poster_url")

        if not isinstance(poster_url, str) or not poster_url.startswith("http"):
            poster_url = get_poster(title, year, active_tmdb_key())

        with cols[index % len(cols)]:
            if poster_url:
                try:
                    st.image(poster_url, width="stretch")
                except Exception:
                    st.markdown(
                        '<div class="poster-fallback">🎬</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f'<div class="poster-fallback">'
                    f'🎬<br><br>{html.escape(title)}</div>',
                    unsafe_allow_html=True,
                )

            score = row.get("recommendation_score")
            score_html = ""

            if score is not None and pd.notna(score):
                score_html = (
                    f'<div class="movie-score">✦ {html.escape(str(score_label))}: '
                    f'{float(score):.3f}</div>'
                )

            # Keep the complete card in one HTML block.
            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">
                        {html.escape(title)}
                    </div>
                    <div class="movie-genres">
                        {html.escape(genres)}
                    </div>
                    {score_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==================================================
# 5. TMDB API FUNCTIONS
# ==================================================

@st.cache_data(ttl=3600, show_spinner=False)
def tmdb_get(endpoint, params_tuple, api_key):
    """
    Request TMDb data using the API key.

    params_tuple is a tuple of (key, value) pairs so Streamlit
    can cache requests consistently.
    """
    params = dict(params_tuple)
    params["api_key"] = api_key

    response = requests.get(
        f"{TMDB_BASE_URL}{endpoint}",
        params=params,
        timeout=15,
    )

    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=86400, show_spinner=False)
def tmdb_genres(api_key):
    data = tmdb_get(
        "/genre/movie/list",
        (("language", "en-US"),),
        api_key,
    )

    return {
        item["id"]: item["name"]
        for item in data.get("genres", [])
    }


def normalize_tmdb_movie(movie, genre_map):
    genre_ids = movie.get("genre_ids", [])

    genre_names = [
        genre_map[genre_id]
        for genre_id in genre_ids
        if genre_id in genre_map
    ]

    poster_path = movie.get("poster_path")

    return {
        "title": movie.get("title") or movie.get("name") or "Untitled",
        "overview": movie.get("overview") or "No overview available.",
        "release_date": movie.get("release_date") or "Release date unavailable",
        "rating": movie.get("vote_average", 0),
        "votes": movie.get("vote_count", 0),
        "genres": ", ".join(genre_names) if genre_names else "Genre unavailable",
        "poster": (
            f"{TMDB_IMAGE_BASE}{poster_path}"
            if poster_path
            else None
        ),
        "id": movie.get("id"),
    }


def show_tmdb_movies(items, genre_map, limit=20):
    if not items:
        st.info("No movies found for these filters.")
        return

    cols = st.columns(4, gap="medium")

    for index, movie in enumerate(items[:limit]):
        item = normalize_tmdb_movie(movie, genre_map)

        with cols[index % len(cols)]:
            if item["poster"]:
                st.image(item["poster"], width="stretch")
            else:
                st.markdown(
                    f'<div class="poster-fallback">'
                    f'🎬<br><br>{html.escape(item["title"])}</div>',
                    unsafe_allow_html=True,
                )

            rating = item["rating"]
            release = item["release_date"]

            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">
                        {html.escape(item["title"])}
                    </div>
                    <div class="movie-genres">
                        {html.escape(item["genres"])}
                    </div>
                    <div class="movie-score">
                        ⭐ {rating:.1f}/10
                        &nbsp;·&nbsp;
                        {html.escape(str(release))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Movie details"):
                st.write(item["overview"])
                st.caption(f"TMDb votes: {item['votes']:,}")
                st.markdown(
                    f"[View on TMDb](https://www.themoviedb.org/movie/{item['id']})"
                )



# ==================================================
# 5A. SMART MATCH SCORE + CINEMA JOURNAL HELPERS
# ==================================================

JOURNAL_PATH = ROOT / "cinema_journal.json"

def active_tmdb_key():
    return str(st.session_state.get("tmdb_api_key_input", TMDB_API_KEY_DEFAULT)).strip()


def load_journal():
    """Load the user's journal from a small local JSON file."""
    try:
        if JOURNAL_PATH.exists():
            data = pd.read_json(JOURNAL_PATH, orient="records")
            return data.to_dict("records")
    except (ValueError, OSError):
        pass
    return []


def save_journal(entries):
    """Save journal entries locally on this computer."""
    try:
        JOURNAL_PATH.write_text(
            __import__("json").dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
    except OSError:
        return False


def get_journal_entries():
    if "cinematch_journal" not in st.session_state:
        st.session_state["cinematch_journal"] = load_journal()
    return st.session_state["cinematch_journal"]


def movie_match_score(movie, preferred_genres=None, mood_genres=None, runtime_limit=None):
    """
    Explainable heuristic score (0-100), not a TMDb rating or ML prediction.
    Combines genre fit, audience rating, popularity, and optional runtime fit.
    """
    preferred_genres = set(preferred_genres or [])
    mood_genres = set(mood_genres or [])
    genre_ids = set(movie.get("genre_ids") or [])
    score_parts = []

    if preferred_genres:
        genre_fit = len(genre_ids & preferred_genres) / max(1, len(preferred_genres))
        score_parts.append((0.50, min(1.0, genre_fit)))
    if mood_genres:
        mood_fit = len(genre_ids & mood_genres) / max(1, len(mood_genres))
        score_parts.append((0.20, min(1.0, mood_fit)))

    rating = float(movie.get("vote_average") or 0)
    votes = int(movie.get("vote_count") or 0)
    rating_component = min(1.0, rating / 10.0)
    popularity = float(movie.get("popularity") or 0)
    popularity_component = min(1.0, popularity / 100.0)
    score_parts.extend([(0.20, rating_component), (0.10, popularity_component)])

    if runtime_limit and movie.get("runtime"):
        runtime = int(movie["runtime"])
        runtime_component = 1.0 if runtime <= runtime_limit else max(
            0.0, 1.0 - ((runtime - runtime_limit) / max(runtime_limit, 1))
        )
        score_parts.append((0.15, runtime_component))

    weight_total = sum(weight for weight, _ in score_parts)
    if not weight_total:
        return 0
    score = 100 * sum(weight * value for weight, value in score_parts) / weight_total

    # Small confidence adjustment for titles with very few audience votes.
    confidence = min(1.0, votes / 100.0)
    score = score * (0.85 + 0.15 * confidence)
    return round(max(0, min(100, score)))


def show_smart_tmdb_movies(items, genre_map, preferred_genres=None,
                           mood_genres=None, runtime_limit=None, limit=12,
                           reason_label="Preference match", preserve_order=False):
    """Render TMDb results with an explainable CineMatch score and save option."""
    if not items:
        st.info("No movies matched those filters. Try broadening your preferences.")
        return

    enriched = []
    for movie in items:
        movie = dict(movie)
        movie["_cine_score"] = movie_match_score(
            movie, preferred_genres, mood_genres, runtime_limit
        )
        enriched.append(movie)

    if not preserve_order:
        enriched.sort(
            key=lambda x: (x["_cine_score"], x.get("vote_average", 0)),
            reverse=True,
        )
    st.session_state["last_tmdb_results"] = [
        {
            "id": m.get("id"),
            "title": m.get("title") or m.get("name") or "Untitled",
            "release_date": m.get("release_date") or "",
            "rating": float(m.get("vote_average") or 0),
            "votes": int(m.get("vote_count") or 0),
            "overview": m.get("overview") or "",
            "poster": (
                f"{TMDB_IMAGE_BASE}{m['poster_path']}"
                if m.get("poster_path") else None
            ),
        }
        for m in enriched
    ]

    st.caption("CineMatch Match Score is a transparent preference-fit estimate, not an official rating.")
    cols = st.columns(4, gap="medium")
    for index, movie in enumerate(enriched[:limit]):
        item = normalize_tmdb_movie(movie, genre_map)
        score = movie["_cine_score"]
        title = item["title"]
        with cols[index % len(cols)]:
            if item["poster"]:
                st.image(item["poster"], width="stretch")
            else:
                st.markdown(
                    f'<div class="poster-fallback">🎬<br><br>{html.escape(title)}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="movie-card">'
                f'<div class="movie-title">{html.escape(title)}</div>'
                f'<div class="movie-genres">{html.escape(item["genres"])}</div>'
                f'<div class="movie-score">✦ Match Score: {score}/100</div>'
                f'<div class="movie-genres">⭐ {item["rating"]:.1f}/10 · '
                f'{item["votes"]:,} TMDb votes</div></div>',
                unsafe_allow_html=True,
            )
            with st.expander("Why this match?"):
                reasons = []
                ids = set(movie.get("genre_ids") or [])
                pref = set(preferred_genres or [])
                mood = set(mood_genres or [])
                if pref and ids & pref:
                    reasons.append("Matches your selected genres.")
                if mood and ids & mood:
                    reasons.append("Fits the selected group mood.")
                if item["rating"] >= 7:
                    reasons.append("Has a strong TMDb audience rating.")
                if runtime_limit and movie.get("runtime"):
                    if int(movie["runtime"]) <= runtime_limit:
                        reasons.append(f"Fits the {runtime_limit}-minute runtime limit.")
                if not reasons:
                    reasons.append(reason_label + " using rating, popularity, and available metadata.")
                st.write(" ".join(reasons))
                st.write(item["overview"])
                st.caption(f"Release: {item['release_date']} · TMDb votes: {item['votes']:,}")
                st.markdown(f"[View on TMDb](https://www.themoviedb.org/movie/{item['id']})")


def tmdb_discover(api_key, extra_params):
    """Fetch movie candidates from TMDb Discover endpoint."""
    params = {"language": "en-US", "include_adult": "false", "page": 1}
    params.update(extra_params)
    data = tmdb_get("/discover/movie", tuple(sorted(params.items())), api_key)
    return data.get("results", [])


# ==================================================
# 6. HERO
# ==================================================

st.markdown(
    """
    <div class="cine-hero">
        <div class="cine-eyebrow">Your personal cinema companion</div>
        <h1>Find your next<br>favorite movie.</h1>
        <p>
            Discover films that fit your mood, explore movies similar
            to your favorites, and find hidden gems beyond the usual hits.
            Browse Indian and international releases too.
        </p>
        <span class="hero-pill">✦ Personalized discovery</span>
        <span class="hero-pill">◈ Smart recommendations</span>
        <span class="hero-pill">🎬 Movie night made easy</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 7. INITIALIZE LOCAL RECOMMENDER
# ==================================================

try:
    with st.spinner("Preparing your cinema experience..."):
        movies, ratings = load_data()
        rec = get_engine()

except FileNotFoundError:
    st.error("MovieLens dataset was not found.")
    st.code("python src/download_data.py")
    st.stop()

except Exception as exc:
    st.error("CineMatch could not initialize.")
    st.exception(exc)
    st.stop()


# ==================================================
# 8. APP TABS
# ==================================================

tabs = st.tabs(
    [
        " Discover",
        "♥ Mood",
        "⌕ Describe It",
        "♟ Group Night",
        "✦ Hidden Gems",
        "▥ Model Analytics",
        "🇮🇳 Indian & New",
        "▤ Cinema Journal",
    ]
)


# ==================================================
# TAB 1: DISCOVER
# ==================================================

with tabs[0]:
    st.markdown("## Discover movies")
    st.caption("Your local recommendations combine all six Machine Learning Lab algorithms.")

    # MovieLens contains anonymized sample profiles, not registered app accounts.
    available_profile_ids = sorted(ratings.user_id.astype(int).unique().tolist())
    profile_counts = ratings.groupby("user_id").size()
    default_profile_id = int(profile_counts.idxmax()) if not profile_counts.empty else available_profile_ids[0]
    recommendation_profile_id = st.selectbox(
        "Recommendation profile (MovieLens user ID)",
        available_profile_ids,
        index=available_profile_ids.index(default_profile_id),
        key="recommendation_profile_id",
        help="Choose an anonymized MovieLens rating profile to demonstrate personalized recommendations. This is not an account or your personal data.",
    )

    opts = movies.sort_values("title").reset_index(drop=True)
    title_options = opts["title"].astype(str).tolist()

    if title_options:
        selected_title = st.selectbox(
            "Choose a movie you like",
            title_options,
            key="discover_movie",
        )

        if st.button(
            "Find similar movies",
            type="primary",
            width="stretch",
            key="discover_button",
        ):
            mid = int(
                opts.loc[
                    opts["title"] == selected_title,
                    "movie_id"
                ].iloc[0]
            )
            try:
                with st.spinner("Combining six recommendation algorithms..."):
                    lab_suite = get_lab_model_suite()
                    recommendations = rec.ml_hybrid_recommendations(
                        recommendation_profile_id,
                        lab_suite,
                        n=12,
                        seed_movie_id=mid,
                    )
                show_movies(
                    recommendations,
                    heading="Because you like this movie · six-model hybrid",
                )
            except Exception as exc:
                st.error("Could not generate personalized recommendations.")
                st.caption(str(exc))

    st.divider()
    st.markdown("## Recommended for your profile")
    st.caption("A blended ranking from K-NN, K-Means, Decision Tree, Random Forest, SVM, and Linear Regression.")
    try:
        with st.spinner("Preparing personalized picks from the six-model ensemble..."):
            lab_suite = get_lab_model_suite()
            profile_recommendations = rec.ml_hybrid_recommendations(
                recommendation_profile_id,
                lab_suite,
                n=8,
            )
        show_movies(profile_recommendations, limit=8, score_label="ML ensemble score")
    except Exception as exc:
        st.warning("The personalized ensemble could not be loaded; showing popular recommendations instead.")
        st.caption(str(exc))
        show_movies(rec.popularity_recommendations(8), limit=8)


# ==================================================
# TAB 2: MOOD
# ==================================================

with tabs[1]:
    st.markdown("## What are you in the mood for?")
    st.caption("Find something to watch based on the feeling you're after.")

    mood_options = list(MOOD_GENRES.keys())

    mood = st.selectbox(
        "Choose your mood",
        mood_options,
        key="mood_choice",
    )

    all_genres = sorted(
        {
            genre
            for values in MOOD_GENRES.values()
            for genre in values
        }
    )

    genres = st.multiselect(
        "Optional genre preferences",
        all_genres,
        key="mood_genres",
    )

    if st.button(
        "Recommend for my mood",
        type="primary",
        width="stretch",
        key="mood_button",
    ):
        try:
            with st.spinner("Blending your mood with six ML models..."):
                lab_suite = get_lab_model_suite()
                mood_recommendations = rec.ml_hybrid_recommendations(
                    recommendation_profile_id,
                    lab_suite,
                    n=12,
                    mood=mood,
                    genres=genres,
                )
            show_movies(
                mood_recommendations,
                heading=f"Selected mood: {mood} · ML-personalized",
            )
        except Exception as exc:
            st.error("Could not generate mood-based recommendations.")
            st.caption(str(exc))


# ==================================================
# TAB 3: DESCRIBE IT
# ==================================================

with tabs[2]:
    st.markdown("## Describe your perfect movie")
    st.caption("Describe the genre, atmosphere, or kind of story you want.")

    query = st.text_area(
        "What would you like to watch?",
        placeholder=(
            "A suspenseful mystery with an unexpected ending, "
            "a clever detective, and a dark atmosphere..."
        ),
        height=130,
        key="description_query",
    )

    if st.button(
        "Find matching movies",
        type="primary",
        width="stretch",
        key="description_button",
    ):
        if not query.strip():
            st.warning("Please describe the kind of movie you're looking for.")
        else:
            try:
                with st.spinner("Matching your description with the six-model ensemble..."):
                    lab_suite = get_lab_model_suite()
                    text_recommendations = rec.ml_hybrid_recommendations(
                        recommendation_profile_id,
                        lab_suite,
                        n=12,
                        query=query,
                    )
                show_movies(
                    text_recommendations,
                    heading="Your description matches · ML-personalized",
                )
            except Exception as exc:
                st.error("Could not generate description-based recommendations.")
                st.caption(str(exc))

    st.caption(
        "Description matching uses MovieLens metadata, primarily titles "
        "and genres. It does not search full plot summaries."
    )


# ==================================================
# TAB 4: GROUP NIGHT / WATCH PARTY
# ==================================================

with tabs[3]:
    st.markdown("## 🍿 Watch Party")
    st.caption("Build a shared movie-night shortlist from everyone's preferred genres, mood, and runtime.")

    api_key = active_tmdb_key()
    if not api_key:
        st.warning("Add your TMDb API key in the Indian & New tab first.")
    else:
        try:
            genre_map = tmdb_genres(api_key)
            genre_names = list(genre_map.values())
            name_to_id = {name: gid for gid, name in genre_map.items()}

            left, right = st.columns(2)
            with left:
                party_genres = st.multiselect(
                    "Genres everyone can agree on",
                    genre_names,
                    default=[g for g in ["Comedy", "Adventure"] if g in genre_names],
                    key="party_genres",
                )
                party_language_options = {
                    "Any language": None, "Hindi": "hi", "Marathi": "mr",
                    "Tamil": "ta", "Telugu": "te", "Malayalam": "ml",
                    "Kannada": "kn", "Bengali": "bn", "English": "en",
                }
                party_language = st.selectbox(
                    "Preferred language", list(party_language_options.keys()),
                    key="party_language",
                )
            with right:
                party_mood_options = {
                    "Anything fun": [], "Feel-good": [35, 10751],
                    "Action-packed": [28, 12], "Mystery night": [9648, 53],
                    "Emotional": [18, 10749], "Mind-bending": [878, 9648],
                }
                party_mood = st.selectbox(
                    "Group mood", list(party_mood_options.keys()), key="party_mood"
                )
                runtime_limit = st.select_slider(
                    "Maximum runtime",
                    options=[90, 105, 120, 135, 150, 180, 210, 240, 270, 300],
                    value=135,
                    format_func=lambda value: f"{value} minutes",
                    key="party_runtime",
                )

            if st.button("Find our movie-night shortlist", type="primary",
                         width="stretch", key="party_tmdb_button"):
                params = {"sort_by": "popularity.desc", "vote_count.gte": 20, "with_runtime.lte": runtime_limit}
                if party_genres:
                    params["with_genres"] = ",".join(
                        str(name_to_id[g]) for g in party_genres
                    )
                mood_ids = party_mood_options[party_mood]
                if mood_ids:
                    params["with_genres"] = "|".join(map(str, mood_ids))
                lang = party_language_options[party_language]
                if lang:
                    params["with_original_language"] = lang
                try:
                    with st.spinner("Finding movies for your group..."):
                        candidates = tmdb_discover(api_key, params)
                        # Runtime is available through TMDb movie details; fetch a small
                        # candidate set and filter only when runtime metadata is returned.
                        if candidates:
                            show_smart_tmdb_movies(
                                candidates, genre_map,
                                preferred_genres=[name_to_id[g] for g in party_genres],
                                mood_genres=mood_ids,
                                runtime_limit=runtime_limit,
                                limit=12,
                                reason_label="Watch Party preference match",
                            )
                        else:
                            st.info("No matches yet. Try removing a genre or choosing Any language.")
                except requests.RequestException:
                    st.error("Could not reach TMDb. Check your connection and API key.")
        except requests.RequestException:
            st.error("Could not load TMDb genres. Check the API key in Indian & New.")

    st.info("How the score works: selected-genre fit, mood fit, TMDb rating, and popularity. It is a heuristic score, not a machine-learning prediction.")


# ==================================================
# TAB 5: HIDDEN TREASURES
# ==================================================

with tabs[4]:
    st.markdown("## ✦ Hidden Treasures")
    st.caption("Find lesser-known movies with promising audience ratings—not just the biggest blockbusters.")

    api_key = active_tmdb_key()
    if not api_key:
        st.warning("Add your TMDb API key in the Indian & New tab first.")
    else:
        try:
            genre_map = tmdb_genres(api_key)
            genre_names = list(genre_map.values())
            name_to_id = {name: gid for gid, name in genre_map.items()}

            c1, c2, c3 = st.columns(3)
            with c1:
                hidden_genre = st.selectbox(
                    "Genre", ["Any genre"] + genre_names, key="hidden_genre"
                )
                hidden_language_options = {
                    "Any language": None, "Hindi": "hi", "Marathi": "mr",
                    "Tamil": "ta", "Telugu": "te", "Malayalam": "ml",
                    "Kannada": "kn", "Bengali": "bn", "English": "en",
                }
                hidden_language = st.selectbox(
                    "Language", list(hidden_language_options.keys()), key="hidden_language"
                )
            with c2:
                minimum_rating = st.slider(
                    "Minimum TMDb rating", 5.0, 9.0, 6.5, 0.5, key="hidden_rating"
                )
                max_votes = st.select_slider(
                    "Maximum vote count (lesser-known filter)",
                    options=[500, 1000, 2500, 5000, 10000, 25000],
                    value=5000,
                    format_func=lambda value: f"{value:,} votes",
                    key="hidden_max_votes",
                )
            with c3:
                release_period = st.selectbox(
                    "Release period",
                    ["Any year", "Since 2020", "2010–2019", "2000–2009", "Before 2000"],
                    key="hidden_period",
                )
                treasure_sort = st.selectbox(
                    "Sort discoveries",
                    ["Best match", "Highest rated", "Least known"],
                    key="hidden_sort",
                )

            if st.button("Discover hidden treasures", type="primary",
                         width="stretch", key="hidden_tmdb_button"):
                params = {
                    "sort_by": (
                        "vote_count.asc"
                        if treasure_sort == "Least known"
                        else "vote_average.desc"
                    ),
                    "vote_average.gte": minimum_rating,
                    "vote_count.gte": 50,
                    "vote_count.lte": max_votes,
                    "vote_average.lte": 10,
                }
                if hidden_genre != "Any genre":
                    params["with_genres"] = name_to_id[hidden_genre]
                lang = hidden_language_options[hidden_language]
                if lang:
                    params["with_original_language"] = lang
                if release_period == "Since 2020":
                    params["primary_release_date.gte"] = "2020-01-01"
                elif release_period == "2010–2019":
                    params["primary_release_date.gte"] = "2010-01-01"
                    params["primary_release_date.lte"] = "2019-12-31"
                elif release_period == "2000–2009":
                    params["primary_release_date.gte"] = "2000-01-01"
                    params["primary_release_date.lte"] = "2009-12-31"
                elif release_period == "Before 2000":
                    params["primary_release_date.lte"] = "1999-12-31"

                try:
                    with st.spinner("Searching for lesser-known movies..."):
                        candidates = tmdb_discover(api_key, params)
                        if treasure_sort == "Highest rated":
                            candidates.sort(key=lambda m: float(m.get("vote_average") or 0), reverse=True)
                        elif treasure_sort == "Least known":
                            candidates.sort(key=lambda m: int(m.get("vote_count") or 0))
                        else:
                            # Balance rating and relative obscurity for a discovery-oriented ranking.
                            candidates.sort(
                                key=lambda m: (
                                    float(m.get("vote_average") or 0) -
                                    min(2.0, int(m.get("vote_count") or 0) / 25000)
                                ),
                                reverse=True,
                            )
                        if candidates:
                            show_smart_tmdb_movies(
                                candidates, genre_map,
                                preferred_genres=[name_to_id[hidden_genre]]
                                if hidden_genre != "Any genre" else [],
                                limit=12,
                                reason_label="Hidden Treasure ranking",
                                preserve_order=True,
                            )
                        else:
                            st.info("No movies match all these filters. Try increasing the vote limit or lowering the rating.")
                except requests.RequestException:
                    st.error("Could not reach TMDb. Check your connection and API key.")
        except requests.RequestException:
            st.error("Could not load TMDb genres. Check the API key in Indian & New.")

    st.caption("Hidden Treasure is a discovery label based on your selected rating and vote-count limits; it does not mean a film is objectively underrated.")


# TAB 6: MODEL ANALYTICS
# ==================================================

with tabs[5]:
    st.markdown("## Model performance analytics")
    st.caption(
        "Compare recommendation models using metrics generated "
        "by the evaluation script."
    )

    results_path = ROOT / "results" / "model_comparison.csv"

    if results_path.exists():
        try:
            scores = pd.read_csv(results_path)

            if "model" in scores.columns:
                required_lab_models = set(MODEL_NAMES)
                present_models = set(scores["model"].astype(str))
                if not required_lab_models.issubset(present_models):
                    st.info(
                        "These results may be from the earlier four-model baseline. "
                        "Run `python src/evaluate_models.py` to generate the new "
                        "10-model comparison."
                    )

            if scores.empty:
                st.info("The model comparison file is empty.")
            else:
                metric_cols = scores.select_dtypes(
                    include="number"
                ).columns.tolist()

                first, second, third = st.columns(3)

                with first:
                    st.metric("Models evaluated", len(scores))

                with second:
                    st.metric("Metrics available", len(metric_cols))

                with third:
                    st.metric("Results file", "Loaded")

                st.markdown("### Comparison table")

                st.dataframe(
                    scores,
                    width="stretch",
                    hide_index=True,
                )

                if "model" in scores.columns and metric_cols:
                    metric = st.selectbox(
                        "Metric to visualize",
                        metric_cols,
                        key="analytics_metric",
                    )

                    chart_data = scores[["model", metric]].copy()
                    chart_data = chart_data.dropna()
                    chart_data = chart_data.set_index("model")

                    st.markdown("### Model comparison")
                    st.bar_chart(chart_data)

                st.caption(
                    "Higher is better for Precision@K, Recall@K, and NDCG@K. "
                    "Lower is better for RMSE and MAE. Interpret each metric "
                    "according to its definition."
                )

        except Exception as exc:
            st.error("Could not read the model comparison results.")
            st.exception(exc)

    else:
        st.info(
            "No evaluation results are available yet. Run the evaluation "
            "script to generate the comparison file."
        )
        st.code("python src/evaluate_models.py")

    rating_results_path = ROOT / "results" / "rating_prediction_metrics.csv"
    if rating_results_path.exists():
        try:
            rating_scores = pd.read_csv(rating_results_path)
            if not rating_scores.empty:
                st.divider()
                st.markdown("### Rating prediction errors")
                st.caption("MAE and RMSE measure rating-prediction error; lower values are better.")
                st.dataframe(rating_scores, width="stretch", hide_index=True)
                rating_metric_cols = [
                    col for col in ["mae", "rmse"] if col in rating_scores.columns
                ]
                if "model" in rating_scores.columns and rating_metric_cols:
                    selected_error_metric = st.selectbox(
                        "Rating error metric to visualize",
                        rating_metric_cols,
                        key="rating_error_metric",
                    )
                    error_chart = rating_scores[["model", selected_error_metric]].copy()
                    error_chart = error_chart.dropna().set_index("model")
                    st.bar_chart(error_chart)
        except Exception as exc:
            st.warning("Could not read rating prediction metrics.")
            st.caption(str(exc))


# ==================================================
# TAB 7: INDIAN & INTERNATIONAL MOVIES
# ==================================================

with tabs[6]:
    st.markdown("## Indian & international cinema")
    st.caption(
        "Browse popular films, new releases, upcoming movies, "
        "and trending cinema using TMDb."
    )

    api_key = st.text_input(
        "TMDb API key",
        value=TMDB_API_KEY_DEFAULT,
        type="password",
        help="Your key is used to retrieve movie data from TMDb.",
        key="tmdb_api_key_input",
    ).strip()

    region_choice = st.radio(
        "Movie region",
        ["India", "Worldwide"],
        horizontal=True,
        key="tmdb_region",
    )

    language_options = {
        "All languages": None,
        "Hindi": "hi",
        "Marathi": "mr",
        "Tamil": "ta",
        "Telugu": "te",
        "Malayalam": "ml",
        "Kannada": "kn",
        "Bengali": "bn",
        "English": "en",
    }

    selected_language = st.selectbox(
        "Language",
        list(language_options.keys()),
        key="tmdb_language",
    )

    categories = [
        "Popular",
        "Now playing",
        "Upcoming",
        "Top rated",
        "Trending this week",
        "Search by title",
    ]

    category = st.selectbox(
        "Browse category",
        categories,
        key="tmdb_category",
    )

    search_title = ""

    if category == "Search by title":
        search_title = st.text_input(
            "Movie title",
            placeholder="Enter a movie name...",
            key="tmdb_search_title",
        )

    page_number = st.selectbox(
        "Page",
        [1, 2, 3, 4, 5],
        key="tmdb_page",
    )

    if st.button(
        "Find movies",
        type="primary",
        width="stretch",
        key="tmdb_search_button",
    ):
        if not api_key:
            st.warning("Enter your TMDb API key to continue.")
        elif category == "Search by title" and not search_title.strip():
            st.warning("Enter a movie title to search.")
        else:
            try:
                with st.spinner("Loading movies from TMDb..."):
                    genre_map = tmdb_genres(api_key)

                    params: dict[str, str | int | bool] = {
                        "page": page_number,
                    }

                    if region_choice == "India":
                        params["region"] = "IN"

                    language_code = language_options[selected_language]

                    if language_code:
                        params["with_original_language"] = language_code

                    if category == "Popular":
                        endpoint = "/discover/movie"
                        params["sort_by"] = "popularity.desc"

                    elif category == "Now playing":
                        endpoint = "/movie/now_playing"
                        params["language"] = "en-US"

                    elif category == "Upcoming":
                        endpoint = "/movie/upcoming"
                        params["language"] = "en-US"

                    elif category == "Top rated":
                        endpoint = "/discover/movie"
                        params["sort_by"] = "vote_average.desc"
                        params["vote_count.gte"] = 100

                    elif category == "Trending this week":
                        endpoint = "/trending/movie/week"
                        params.pop("page", None)

                    else:
                        endpoint = "/search/movie"
                        params["query"] = search_title.strip()
                        params["include_adult"] = "false"

                    # Language filtering is supported by discover and search.
                    # Apply it only to endpoints that support the parameter.
                    if category in [
                        "Popular",
                        "Top rated",
                        "Search by title",
                    ] and language_code:
                        params["with_original_language"] = language_code

                    result = tmdb_get(
                        endpoint,
                        tuple(sorted(params.items())),
                        api_key,
                    )

                    movie_results = result.get("results", [])

                    if not movie_results:
                        st.info(
                            "No movies matched those filters. "
                            "Try another language, region, or category."
                        )
                    else:
                        st.success(
                            f"Found {len(movie_results)} movies "
                            f"from TMDb · {category}"
                        )

                        show_tmdb_movies(
                            movie_results,
                            genre_map,
                            limit=20,
                        )

            except requests.HTTPError as exc:
                status_code = (
                    exc.response.status_code
                    if exc.response is not None
                    else "unknown"
                )

                if status_code == 401:
                    st.error(
                        "TMDb rejected the API key. Check the key in "
                        "your TMDb account settings."
                    )
                elif status_code == 403:
                    st.error(
                        "TMDb denied the request. Check your API access "
                        "and try again."
                    )
                else:
                    st.error(
                        f"TMDb returned HTTP {status_code}. "
                        "Please try again."
                    )

            except requests.Timeout:
                st.error(
                    "TMDb took too long to respond. Check your internet "
                    "connection and try again."
                )

            except requests.RequestException as exc:
                st.error(
                    "Could not connect to TMDb. Check your internet "
                    "connection and try again."
                )
                st.caption(str(exc))

            except (ValueError, KeyError) as exc:
                st.error("The movie data returned by TMDb could not be read.")
                st.caption(str(exc))


# ==================================================
# TAB 8: CINEMA JOURNAL
# ==================================================

with tabs[7]:
    st.markdown("## ▤ My Cinema Journal")
    st.caption("Keep your own watchlist, watched history, favourites, and personal ratings. Entries are saved locally on this computer.")

    entries = get_journal_entries()

    with st.form("journal_add_form", clear_on_submit=True):
        st.markdown("### Add a movie")
        movie_title = st.text_input("Movie title", placeholder="e.g. Interstellar")
        journal_status = st.selectbox(
            "Status", ["Want to watch", "Watched", "Favourite"], key="journal_status_input"
        )
        personal_rating = st.slider(
            "Your rating", 0, 10, 0, help="Choose 0 if you do not want to rate it yet."
        )
        journal_notes = st.text_area("Your notes (optional)", placeholder="What did you think?")
        submitted = st.form_submit_button("Save to my journal", type="primary", width="stretch")

    if submitted:
        if not movie_title.strip():
            st.warning("Enter a movie title first.")
        elif any(
            str(entry.get("title", "")).strip().casefold() == movie_title.strip().casefold()
            for entry in entries
        ):
            st.warning("That movie is already in your journal. Edit or remove the existing entry below.")
        else:
            entries.append({
                "title": movie_title.strip(),
                "status": journal_status,
                "rating": int(personal_rating),
                "notes": journal_notes.strip(),
            })
            st.session_state["cinematch_journal"] = entries
            if save_journal(entries):
                st.success(f"Saved “{movie_title.strip()}” to your journal.")
            else:
                st.warning("Added for this session, but could not write the local journal file.")

    entries = get_journal_entries()
    st.divider()
    st.markdown("### Your collection")
    if entries:
        total = len(entries)
        watched = sum(1 for entry in entries if entry.get("status") == "Watched")
        favourites = sum(1 for entry in entries if entry.get("status") == "Favourite")
        rated = [int(entry.get("rating", 0)) for entry in entries if int(entry.get("rating", 0)) > 0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Movies saved", total)
        m2.metric("Watched", watched)
        m3.metric("Favourites", favourites)
        m4.metric("Average personal rating", f"{sum(rated) / len(rated):.1f}/10" if rated else "—")

        journal_df = pd.DataFrame(entries)
        st.dataframe(journal_df, width="stretch", hide_index=True)
        remove_title = st.selectbox(
            "Remove an entry", ["Choose a movie..."] + [str(e.get("title", "")) for e in entries],
            key="journal_remove_title",
        )
        if st.button("Remove selected entry", key="journal_remove_button"):
            if remove_title != "Choose a movie...":
                entries = [e for e in entries if str(e.get("title", "")) != remove_title]
                st.session_state["cinematch_journal"] = entries
                if save_journal(entries):
                    st.success(f"Removed {remove_title}.")
                    st.rerun()
                else:
                    st.error("Could not update the local journal file.")
    else:
        st.info("Your journal is empty. Add your first movie above.")
        last_results = st.session_state.get("last_tmdb_results", [])
        if last_results:
            st.markdown("### Recently discovered on CineMatch")
            suggested_titles = [item["title"] for item in last_results]
            suggested = st.selectbox("Pick a recent discovery", suggested_titles, key="journal_recent_pick")
            if st.button("Add selected discovery to watchlist", key="journal_add_discovery"):
                selected = next(item for item in last_results if item["title"] == suggested)
                entries.append({
                    "title": selected["title"], "status": "Want to watch",
                    "rating": 0, "notes": f"Discovered on TMDb. TMDb rating: {selected['rating']}/10",
                })
                st.session_state["cinematch_journal"] = entries
                if save_journal(entries):
                    st.success("Added to your watchlist.")
                    st.rerun()


# The six lab algorithms are integrated into Discover, Mood, and Describe It.
# The Model Analytics tab compares individual models; there is no separate Lab Models tab.


# ==================================================
# 9. FOOTER
# ==================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:18px 0 4px 0;">
        <div style="font-size:1rem; font-weight:800; color:#ff6972;">
            🎬 CineMatch
        </div>
        <div style="font-size:.78rem; color:#aaaab7; margin-top:6px;">
            Discover differently. Watch something unforgettable.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
