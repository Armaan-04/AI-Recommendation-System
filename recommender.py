import os
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Read token from environment or Streamlit secrets
TMDB_READ_TOKEN = os.getenv("TMDB_READ_TOKEN")

if TMDB_READ_TOKEN is None:
    raise RuntimeError("TMDB_READ_TOKEN not found. Add it to Streamlit secrets or environment variables.")

HEADERS = {
    "Authorization": f"Bearer {TMDB_READ_TOKEN}",
    "accept": "application/json"
}

@staticmethod
def _tmdb_get(url, params=None):
    res = requests.get(url, headers=HEADERS, params=params)
    res.raise_for_status()
    return res.json()

def fetch_movies_2020_2025(pages=5):
    all_movies = []

    for page in range(1, pages + 1):
        data = _tmdb_get(
            f"{TMDB_BASE_URL}/discover/movie",
            params={
                "sort_by": "vote_average.desc",
                "vote_count.gte": 300,
                "primary_release_date.gte": "2020-01-01",
                "primary_release_date.lte": "2025-12-31",
                "page": page,
                "language": "en-US"
            }
        )

        for m in data["results"]:
            all_movies.append({
                "title": m["title"],
                "overview": m["overview"] or "",
                "rating": m["vote_average"],
                "genre_ids": m["genre_ids"]
            })

    return pd.DataFrame(all_movies)

def fetch_genre_map():
    data = _tmdb_get(f"{TMDB_BASE_URL}/genre/movie/list")
    return {g["id"]: g["name"] for g in data["genres"]}

def build_similarity_model(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["overview"].tolist(), show_progress_bar=True)
    return model, embeddings

def recommend_similar_movies(input_title, df, embeddings, genre_map, top_n=5):
    if input_title not in df["title"].values:
        return []

    idx = df.index[df["title"] == input_title][0]
    input_genres = set(df.iloc[idx]["genre_ids"])

    # 🔥 GENRE FILTERING FIRST
    filtered_df = df[df["genre_ids"].apply(lambda gids: len(set(gids) & input_genres) > 0)].reset_index(drop=True)
    filtered_embeddings = embeddings[filtered_df.index]

    input_embedding = embeddings[idx].reshape(1, -1)
    sims = cosine_similarity(input_embedding, filtered_embeddings)[0]

    filtered_df["similarity"] = sims
    filtered_df = filtered_df.sort_values(by=["similarity", "rating"], ascending=False)

    results = filtered_df[filtered_df["title"] != input_title].head(top_n)

    results["genres"] = results["genre_ids"].apply(lambda gids: ", ".join([genre_map[g] for g in gids if g in genre_map]))

    return results[["title", "rating", "genres"]]