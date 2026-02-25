import os
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st

TMDB_BASE_URL = "https://api.themoviedb.org/3"

def get_tmdb_headers():
    token = st.secrets.get("TMDB_READ_TOKEN", None)
    if not token:
        raise RuntimeError("TMDB_READ_TOKEN not found in Streamlit secrets.")
    return {
        "Authorization": f"Bearer {token}",
        "accept": "application/json"
    }

def fetch_genre_map():
    url = f"{TMDB_BASE_URL}/genre/movie/list"
    res = requests.get(url, headers=get_tmdb_headers())
    res.raise_for_status()
    data = res.json()["genres"]
    return {g["id"]: g["name"] for g in data}

def fetch_movies_2000_2025(pages=5):
    genre_map = fetch_genre_map()
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
        res = requests.get(url, headers=get_tmdb_headers(), params=params)
        res.raise_for_status()

        for m in res.json()["results"]:
            all_movies.append({
                "title": m.get("title", ""),
                "overview": m.get("overview", ""),
                "year": m.get("release_date", "")[:4],
                "rating": m.get("vote_average", 0),
                "genres_text": ", ".join([genre_map.get(gid, "") for gid in m.get("genre_ids", [])])
            })

    df = pd.DataFrame(all_movies)
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    return df

def build_similarity_model(df):
    df["overview"] = df["overview"].fillna("")
    df["genres_text"] = df["genres_text"].fillna("")

    df["content"] = (
        df["title"] + " " +
        df["overview"] + " " +
        (df["genres_text"] + " ") * 3  # boost genre importance
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["content"].tolist(), show_progress_bar=True)

    return embeddings

def recommend_similar_movies(movie_title, df, embeddings, selected_genres=None, top_n=10):
    if movie_title not in df["title"].values:
        return pd.DataFrame()

    idx = df.index[df["title"] == movie_title][0]
    sim_scores = cosine_similarity([embeddings[idx]], embeddings)[0]

    df_scores = df.copy()
    df_scores["score"] = sim_scores

    if selected_genres:
        mask = df_scores["genres_text"].str.contains("|".join(selected_genres), case=False, na=False)
        df_scores = df_scores[mask]

    df_scores = df_scores.sort_values("score", ascending=False)
    return df_scores.iloc[1: top_n + 1][["title", "year", "rating", "genres_text"]]