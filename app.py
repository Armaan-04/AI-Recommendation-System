import streamlit as st
from recommender import fetch_movies_2000_2025, build_similarity_model, recommend_similar_movies

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")

st.title("🎬 AI-Powered Movie Recommendation System (2000–2025)")
st.caption("Recommendations powered by NLP embeddings (Sentence Transformers)")

@st.cache_data
def load_data():
    return fetch_movies_2000_2025(pages=6)

df = load_data()

@st.cache_resource
def load_model(df):
    return build_similarity_model(df)

embeddings = load_model(df)

movie = st.selectbox("🎥 Choose a movie", sorted(df["title"].unique()))

genres = ["All", "Action", "Comedy", "Drama", "Thriller", "Romance", "Sci-Fi", "Fantasy", "Horror"]
genre_filter = st.selectbox("🎭 Filter by Genre", genres)

if st.button("Recommend"):
    results = recommend_similar_movies(
        movie,
        df,
        embeddings,
        top_n=10,
        genre_filter=None if genre_filter == "All" else genre_filter
    )

    for _, row in results.iterrows():
        st.markdown(f"### {row['title']} ({row['year']})")
        st.markdown(f"⭐ Rating: {row['rating']}")
        st.markdown(f"🎭 Genres: {row['genres']}")
        st.markdown(f"<p style='font-size:14px'>{row['overview']}</p>", unsafe_allow_html=True)
        st.divider()
