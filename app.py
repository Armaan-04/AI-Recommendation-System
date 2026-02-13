import streamlit as st
from recommender import fetch_movies_upto_2025, build_similarity_model, recommend_similar_movies

st.set_page_config(page_title="🎬 AI Movie Recommendation System (Best Movies up to 2025)", layout="wide")

st.title("🎬 AI Movie Recommendation System (Best Movies up to 2025)")
st.write("AI-powered recommendations using NLP (TF-IDF + Cosine Similarity) on top-rated movies till 2025.")

@st.cache_data
def load_data():
    df = fetch_movies_upto_2025(start_year=2000, end_year=2025, pages_per_year=1)
    sim = build_similarity_model(df)
    return df, sim

with st.spinner("Loading movies and building AI model..."):
    df, similarity = load_data()

movie_list = sorted(df["title"].tolist())
selected_movie = st.selectbox("Pick a movie you like:", movie_list)

top_n = st.slider("How many recommendations?", 5, 20, 10)

if st.button("Recommend 🎯"):
    recs = recommend_similar_movies(selected_movie, df, similarity, top_n=top_n)

    if recs.empty:
        st.warning("No recommendations found.")
    else:
        st.subheader("AI-Powered Recommendations")
        for _, row in recs.iterrows():
            st.markdown(
                f"""
                **🎬 {row['title']}**  
                ⭐ Rating: {row['vote_average']} ({int(row['vote_count'])} votes)  
                📅 Release Date: {row['release_date']}
                """
            )



        
