import streamlit as st
from recommender import fetch_movies_2020_2025, fetch_genre_map, build_similarity_model, recommend_similar_movies

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI-Powered Movie Recommendation System (2020–2025)")

@st.cache_data(show_spinner=True)
def load_data():
    df = fetch_movies_2020_2025(pages=5)
    genre_map = fetch_genre_map()
    return df, genre_map

@st.cache_resource(show_spinner=True)
def load_model(df):
    model, embeddings = build_similarity_model(df)
    return embeddings

with st.spinner("Fetching movies from TMDB..."):
    df, genre_map = load_data()

with st.spinner("Building AI similarity model..."):
    embeddings = load_model(df)

movie_list = sorted(df["title"].tolist())
selected_movie = st.selectbox("🎥 Select a movie you like:", movie_list)

if st.button("🔮 Recommend Similar Movies"):
    recommendations = recommend_similar_movies(selected_movie, df, embeddings, genre_map, top_n=6)

    if len(recommendations) == 0:
        st.warning("No similar movies found.")
    else:
        st.subheader("✨ You may also like:")
        for _, row in recommendations.iterrows():
            st.markdown(f"**🎬 {row['title']}**")
            st.write(f"⭐ Rating: {row['rating']}")
            st.write(f"🎭 Genres: {row['genres']}")
            st.markdown("---")
        
