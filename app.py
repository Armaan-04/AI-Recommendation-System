import streamlit as st
from recommender import fetch_movies_2000_2025, build_similarity_model, recommend_similar_movies

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI-Powered Movie Recommendation System (2000–2025)")
st.caption("AI-based recommendations using sentence embeddings + genre filtering")

@st.cache_data
def load_data():
    return fetch_movies_2000_2025(pages=5)

@st.cache_resource
def load_model(df):
    return build_similarity_model(df)

df = load_data()
embeddings = load_model(df)

st.subheader("🔍 Choose a Movie")
movie = st.selectbox("Select a movie you like", df["title"].sort_values().tolist())

all_genres = sorted(set(", ".join(df["genres_text"]).split(", ")))
all_genres = [g for g in all_genres if g.strip()]

selected_genres = st.multiselect(
    "Optional: Filter by genres",
    options=all_genres,
    default=[]
)

if st.button("🎯 Recommend Movies"):
    results = recommend_similar_movies(movie, df, embeddings, selected_genres, top_n=10)

    if results.empty:
        st.warning("No recommendations found. Try removing genre filters.")
    else:
        st.subheader("🎬 Recommended Movies")
        for _, row in results.iterrows():
            st.markdown(f"**{row['title']}** ({row['year']}) ⭐ {row['rating']}")
            st.caption(f"Genres: {row['genres_text']}")
            st.divider()
        
