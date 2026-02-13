import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_READ_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzYWE1ZmMxYTBlMTI4NDkzZGU1OTljOGJiYTQyMzI3MyIsIm5iZiI6MTc3MDk5OTAyNy45MDcsInN1YiI6IjY5OGY0Y2YzY2Q3M2U5YTgzZjgzZWVlYyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.dutUEGivjhshrW-ye1XWw85Zv1SKWLC3nSyJV4czY1Q"  

HEADERS = {
    "Authorization": f"Bearer {TMDB_READ_TOKEN}",
    "accept": "application/json"
}

def fetch_movies_upto_2025(start_year=2000, end_year=2025, pages_per_year=1):
    movies = []

    for year in range(start_year, end_year + 1):
        for page in range(1, pages_per_year + 1):
            params = {
                "primary_release_year": year,
                "sort_by": "vote_average.desc",
                "vote_count.gte": 500,
                "page": page,
            }
            r = requests.get(
                f"{TMDB_BASE_URL}/discover/movie",
                headers=HEADERS,
                params=params,
                timeout=30
            )
            r.raise_for_status()
            movies.extend(r.json().get("results", []))

    df = pd.DataFrame(movies)
    df = df[["id", "title", "overview", "release_date", "vote_average", "vote_count"]]
    df["overview"] = df["overview"].fillna("")
    df = df.drop_duplicates("title").reset_index(drop=True)
    return df

def build_similarity_model(df):
    df["content"] = df["title"].astype(str) + " " + df["overview"].astype(str)
    tfidf = TfidfVectorizer(stop_words="english", max_features=15000)
    tfidf_matrix = tfidf.fit_transform(df["content"])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity

def recommend_similar_movies(selected_title, df, similarity, top_n=10):
    if selected_title not in df["title"].values:
        return pd.DataFrame(columns=["title", "release_date", "vote_average", "vote_count"])

    idx = df.index[df["title"] == selected_title][0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1: top_n + 1]
    rec_indices = [i[0] for i in scores]

    return df.iloc[rec_indices][["title", "release_date", "vote_average", "vote_count"]]




