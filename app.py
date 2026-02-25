import streamlit as st
from recommender import fetch_movies_2000_2025, build_similarity_model, recommend_similar_movies

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")

st.title("🎬 AI-Powered Movie Recommendation System (2000–2025)")
st.caption("Content-based AI recommendations using embeddings + cosine similarity")

@st.cache_data(show_spinner=True)
def load_data():
    return fetch_movies_2000_2025(pages=5)

@st.cache_resource(show_spinner=True)
def load_model(df):
    return build_similarity_model(df)

df = load_data()
model, embeddings = load_model(df)

movie_list = sorted(df["title"].tolist())
genre_list = sorted({g for sub in df["genres"] for g in sub})

col1, col2 = st.columns(2)

with col1:
    selected_movie = st.selectbox("🎥 Select a movie", movie_list)

with col2:
    selected_genres = st.multiselect("🎭 Filter by genres (optional)", genre_list, default=["Action"] if "Action" in genre_list else [])

if st.button("✨ Recommend Similar Movies"):
    results = recommend_similar_movies(selected_movie, df, embeddings, selected_genres)

    if results.empty:
        st.warning("No similar movies found. Try changing genre filters.")
    else:
        st.subheader("🔥 Recommended Movies")
        for _, row in results.iterrows():
            st.markdown(f"""
**🎬 {row['title']} ({row['year']})**  
⭐ Rating: {row['rating']}  
🎭 Genres: {', '.join(row['genres'])}  
---
""")
