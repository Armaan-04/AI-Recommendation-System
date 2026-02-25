import requests
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_BASE_URL = "https://api.themoviedb.org/3"

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_tmdb_headers():
    token = st.secrets["TMDB_READ_TOKEN"]
    return {
        "Authorization": f"Bearer {token}",
        "accept": "application/json"
    }

@st.cache_data(show_spinner=False)
def fetch_genre_map():
    url = f"{TMDB_BASE_URL}/genre/movie/list?language=en-US"
    res = requests.get(url, headers=get_tmdb_headers(), timeout=30)
    res.raise_for_status()
    return {g["id"]: g["name"] for g in res.json()["genres"]}

@st.cache_data(show_spinner=True)
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

        for m in res.json()["results"]:
            genres = [genre_map.get(gid, "") for gid in m.get("genre_ids", [])]
            all_movies.append({
                "title": m["title"],
                "overview": m.get("overview", ""),
                "genres": genres,
                "year": (m.get("release_date", "") or "")[:4],
                "rating": m.get("vote_average", 0)
            })

    return pd.DataFrame(all_movies)

@st.cache_resource(show_spinner=True)
def build_similarity_model(df):
    model = load_embedding_model()
    texts = (df["overview"].fillna("") + " " + df["genres"].astype(str)).tolist()
    embeddings = model.encode(texts, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix

def recommend_similar_movies(title, df, sim_matrix, top_n=5, genre_filter=None):
    if title not in df["title"].values:
        return pd.DataFrame()

    idx = df.index[df["title"] == title][0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]

    results = []
    for i, _ in scores:
        row = df.iloc[i]
        if genre_filter and genre_filter not in row["genres"]:
            continue
        results.append(row)
        if len(results) >= top_n:
            break

    return pd.DataFrame(results)