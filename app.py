import streamlit as st
from recommender import fetch_movies_2020_2025, build_similarity_model, recommend_similar_movies

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI-Powered Movie Recommendation System (2020–2025)")

@st.cache_data
def load_data():
    return fetch_movies_2020_2025(pages=5)

@st.cache_resource
def load_model(df):
    return build_similarity_model(df)

with st.spinner("Loading movies from TMDB..."):
    df = load_data()

with st.spinner("Building AI similarity model..."):
    sim_matrix = load_model(df)

movie_list = sorted(df["title"].unique().tolist())

selected_movie = st.selectbox("🎥 Select a movie you liked:", movie_list)

if st.button("Recommend Similar Movies"):
    recommendations = recommend_similar_movies(selected_movie, df, sim_matrix)

    if recommendations.empty:
        st.warning("Movie not found in dataset.")
    else:
        st.subheader("🍿 Recommended Movies")
        st.dataframe(recommendations, use_container_width=True)
        
