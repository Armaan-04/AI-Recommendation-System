import requests
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_BASE_URL = "https://api.themoviedb.org/3"


def get_tmdb_headers():
    try:
        token = st.secrets["TMDB_READ_TOKEN"]  # Streamlit Cloud
    except Exception:
        token = "PASTE_YOUR_TMDB_READ_TOKEN_HERE"  # Local dev only

    return {
        "Authorization": f"Bearer {token}",
        "accept": "application/json"
    }


@st.cache_data(show_spinner=False)
def fetch_genre_map():
    url = f"{TMDB_BASE_URL}/genre/movie/list"
    res = requests.get(url, headers=get_tmdb_headers(), timeout=30)
    res.raise_for_status()
    data = res.json()
    return {g["id"]: g["name"] for g in data["genres"]}


@st.cache_data(show_spinner=False)
def fetch_movies_2000_2025(pages=5):
    genre_map = fetch_genre_map()
    all_movies = []

    for page in range(1, pages + 1):
        url = f"{TMDB_BASE_URL}/discover/movie"
        params = {
            "sort_by": "vote_average.desc",
            "vote_count.gte": 500,
            "primary_release_date.gte": "2000-01-01",
            "primary_release_date.lte": "2025-12-31",
            "page": page,
            "language": "en-US"
        }

        res = requests.get(url, headers=get_tmdb_headers(), params=params, timeout=30)
        res.raise_for_status()
        data = res.json()

        for m in data["results"]:
            genres = [genre_map.get(gid, "") for gid in m.get("genre_ids", [])]
            all_movies.append({
                "title": m["title"],
                "overview": m.get("overview", ""),
                "genres": genres,
                "rating": m.get("vote_average", 0),
                "year": m.get("release_date", "")[:4]
            })

    df = pd.DataFrame(all_movies)
    df["genres_text"] = df["genres"].apply(lambda x: " ".join(x))
    df["text"] = df["overview"].fillna("") + " " + df["genres_text"] * 3
    return df


@st.cache_resource(show_spinner=False)
def build_similarity_model(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=False)
    return model, embeddings


def recommend_similar_movies(movie_title, df, embeddings, selected_genres=None, top_k=10):
    if movie_title not in df["title"].values:
        return pd.DataFrame()

    idx = df[df["title"] == movie_title].index[0]
    sim_scores = cosine_similarity([embeddings[idx]], embeddings)[0]

    df = df.copy()
    df["similarity"] = sim_scores

    if selected_genres:
        df = df[df["genres"].apply(lambda g: any(genre in g for genre in selected_genres))]

    recommendations = df.sort_values(by=["similarity", "rating"], ascending=False)
    return recommendations.head(top_k)[["title", "year", "genres", "rating"]]