import os
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

TMDB_API_KEY = os.getenv("3aa5fc1a0e128493de599c8bba423273")  # Use your existing env var
TMDB_BASE_URL = "https://api.themoviedb.org/3"


def fetch_movies_2020_2025(pages=5):
    """
    Fetch popular & top-rated movies from TMDB between 2020 and 2025.
    """
    all_movies = []
    for page in range(1, pages + 1):
        url = f"{TMDB_BASE_URL}/discover/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "sort_by": "vote_average.desc",
            "vote_count.gte": 500,
            "primary_release_date.gte": "2020-01-01",
            "primary_release_date.lte": "2025-12-31",
            "page": page,
            "language": "en-US",
        }
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json().get("results", [])
        all_movies.extend(data)

    df = pd.DataFrame(all_movies)
    df = df[["id", "title", "overview", "release_date", "vote_average", "genre_ids"]]
    df["overview"] = df["overview"].fillna("")
    return df


def build_similarity_model(df):
    """
    Build sentence embeddings and cosine similarity matrix.
    """
    texts = (df["title"] + ". " + df["overview"]).tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix


def recommend_similar_movies(title, df, sim_matrix, top_n=10):
    """
    Recommend movies similar to the given title.
    """
    if title not in df["title"].values:
        return []

    idx = df.index[df["title"] == title][0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

    recs = df.iloc[[i[0] for i in scores]][
        ["title", "release_date", "vote_average"]
    ]
    return recs.reset_index(drop=True)



