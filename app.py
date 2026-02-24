import streamlit as st
from recommender import fetch_movies_2020_2025, build_similarity_model, recommend_similar_movies

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI-Powered Movie Recommendation System (2020–2025)")
st.write("Select a movie and get semantically similar recommendations using AI embeddings.")

@st.cache_data
def load_data():
    return fetch_movies_2020_2025(pages=5)

@st.cache_resource
def build_model(df):
    return build_similarity_model(df)

with st.spinner("Loading movies..."):
    df = load_data()

with st.spinner("Building AI similarity model... (first run may take a minute)"):
    sim_matrix = build_model(df)

movie_list = df["title"].sort_values().tolist()
selected_movie = st.selectbox("🎥 Select a movie", movie_list)

if st.button("Recommend 🎯"):
    recs = recommend_similar_movies(selected_movie, df, sim_matrix, top_n=10)

    if len(recs) == 0:
        st.warning("Movie not found.")
    else:
        st.subheader("✨ Recommended Movies")
        for _, row in recs.iterrows():
            st.markdown(
                f"**{row['title']}**  \n"
                f"📅 {row['release_date']} | ⭐ {row['vote_average']}"
            )


        
