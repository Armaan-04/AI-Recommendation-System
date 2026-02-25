import os
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Get token from Streamlit secrets or environment
def get_tmdb_token():
    try:
        import streamlit as st
        return st.secrets["TMDB_READ_TOKEN"]
    except Exception:
        token = os.getenv("TMDB_READ_TOKEN")
        if not token:
            raise RuntimeError("TMDB_READ_TOKEN not found. Add it to Streamlit secrets or environment variables.")
        return token

TMDB_READ_TOKEN = get_tmdb_token()

HEADERS = {
    "Authorization": f"Bearer {TMDB_READ_TOKEN}",
    "accept": "application/json"
}

# Fetch genre map
def fetch_genre_map():
    url = f"{TMDB_BASE_URL}/genre/movie/list?language=en-US"
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()
    data = res.json()["genres"]
    return {g["id"]: g["name"] for g in data}

# Fetch movies from 2000–2025
def fetch_movies_2000_2025(pages=5):
    all_movies = []
    for page in range(1, pages + 1):
        url = f"{TMDB_BASE_URL}/discover/movie"
        params = {
            "sort_by": "vote_average.desc",
            "vote_count.gte": 300,
            "primary_release_date.gte": "2000-01-01",
            "primary_release_date.lte": "2025-12-31",
            "language": "en-US",
            "page": page
        }
        res = requests.get(url, headers=HEADERS, params=params)
        res.raise_for_status()
        all_movies.extend(res.json()["results"])

    df = pd.DataFrame(all_movies)
    return df

# Build AI similarity model
def build_similarity_model(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Make content richer
    df["content"] = (
        df["title"].fillna("") + " " +
        df["overview"].fillna("") + " " +
        df["genres_text"].fillna("") * 3  # boost genre importance
    )

    embeddings = model.encode(df["content"].tolist(), show_progress_bar=True)
    sim_matrix = cosine_similarity(embeddings, embeddings)

    return sim_matrix

# Recommend similar movies
def recommend_similar_movies(title, df, sim_matrix, top_n=10, selected_genres=None):
    if title not in df["title"].values:
        return []

    idx = df.index[df["title"] == title][0]
    scores = list(enumerate(sim_matrix[idx]))

    # Remove itself
    scores = scores[1:]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []
    for i, score in scores:
        movie = df.iloc[i]

        # Genre filter
        if selected_genres:
            movie_genres = set(movie["genres_text"].split(", "))
            if not movie_genres.intersection(set(selected_genres)):
                continue

        hybrid_score = (
            0.6 * score +
            0.3 * (movie["vote_average"] / 10) +
            0.1 * (min(movie["vote_count"], 5000) / 5000)
        )

        results.append({
            "title": movie["title"],
            "rating": movie["vote_average"],
            "genres": movie["genres_text"],
            "overview": movie["overview"],
            "score": round(hybrid_score, 3)
        })

        if len(results) >= top_n:
            break

    return results