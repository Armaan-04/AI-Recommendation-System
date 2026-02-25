import os
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_READ_TOKEN = os.getenv("TMDB_READ_TOKEN")

if not TMDB_READ_TOKEN:
    raise RuntimeError("TMDB_READ_TOKEN not set. Add it in Streamlit Secrets.")

TMDB_BASE_URL = "https://api.themoviedb.org/3"

HEADERS = {
    "Authorization": f"Bearer {TMDB_READ_TOKEN}",
    "accept": "application/json"
}

def fetch_movies_2020_2025(pages=5):
    movies = []

    for page in range(1, pages + 1):
        url = f"{TMDB_BASE_URL}/discover/movie"
        params = {
            "sort_by": "vote_average.desc",
            "vote_count.gte": 500,
            "primary_release_date.gte": "2020-01-01",
            "primary_release_date.lte": "2025-12-31",
            "language": "en-US",
            "page": page
        }

        res = requests.get(url, headers=HEADERS, params=params)
        res.raise_for_status()
        data = res.json()["results"]

        for m in data:
            movies.append({
                "title": m["title"],
                "overview": m["overview"],
                "rating": m["vote_average"],
                "release_date": m["release_date"]
            })

    return pd.DataFrame(movies)


def build_similarity_model(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["overview"].fillna("").tolist(), show_progress_bar=True)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


def recommend_similar_movies(title, df, sim_matrix, top_n=5):
    if title not in df["title"].values:
        return pd.DataFrame()

    idx = df.index[df["title"] == title][0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    results = []
    for i, score in scores:
        results.append({
            "Title": df.iloc[i]["title"],
            "Rating": df.iloc[i]["rating"],
            "Release Date": df.iloc[i]["release_date"],
            "Similarity": round(score, 3)
        })

    return pd.DataFrame(results)



