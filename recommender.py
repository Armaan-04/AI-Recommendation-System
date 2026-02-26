import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_BASE_URL = "https://api.themoviedb.org/3"

def get_headers():
    token = st.secrets["TMDB_READ_TOKEN"]
    return {
        "Authorization": f"Bearer {token}",
        "accept": "application/json"
    }

@st.cache_data(show_spinner=False)
def fetch_genre_map():
    url = f"{TMDB_BASE_URL}/genre/movie/list"
    res = requests.get(url, headers=get_headers(), timeout=30)
    res.raise_for_status()
    genres = res.json()["genres"]
    return {g["id"]: g["name"] for g in genres}

@st.cache_data(show_spinner=True)
def fetch_movies_2000_2025(pages=5):
    genre_map = fetch_genre_map()
    all_movies = []

    for page in range(1, pages + 1):
        url = f"{TMDB_BASE_URL}/discover/movie"
        params = {
            "sort_by": "popularity.desc",
            "primary_release_date.gte": "2000-01-01",
            "primary_release_date.lte": "2025-12-31",
            "vote_count.gte": 300,
            "language": "en-US",
            "page": page
        }
        res = requests.get(url, headers=get_headers(), params=params, timeout=30)
        res.raise_for_status()

        for m in res.json()["results"]:
            genres = [genre_map.get(g, "") for g in m.get("genre_ids", [])]
            all_movies.append({
                "title": m["title"],
                "overview": m.get("overview", ""),
                "genres": ", ".join(genres),
                "rating": m.get("vote_average", 0),
                "year": m.get("release_date", "")[:4]
            })

    return pd.DataFrame(all_movies)

@st.cache_resource(show_spinner=True)
def build_similarity_model(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    text = (
        df["title"].fillna("") + " " +
        df["overview"].fillna("") + " " +
        df["genres"].fillna("") * 3
    ).tolist()

    embeddings = model.encode(text, show_progress_bar=False)
    return embeddings

def recommend_similar_movies(title, df, embeddings, top_n=8, genre_filter=None):
    if title not in df["title"].values:
        return []

    idx = df.index[df["title"] == title][0]
    scores = cosine_similarity([embeddings[idx]], embeddings)[0]

    df_scores = df.copy()
    df_scores["score"] = scores

    if genre_filter:
        df_scores = df_scores[df_scores["genres"].str.contains(genre_filter, case=False)]

    df_scores = df_scores.sort_values("score", ascending=False)

    return df_scores.iloc[1: top_n + 1]