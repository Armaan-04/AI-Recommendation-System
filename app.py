import streamlit as st
from recommender import fetch_movies_2000_2025, build_similarity_model, recommend_similar_movies

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="centered")
st.title("🎬 AI-Powered Movie Recommendation System (2000–2025)")

@st.cache_data
def load_data():
    return fetch_movies_2000_2025(pages=5)

df = load_data()
sim_matrix = build_similarity_model(df)

genres = sorted({g for sublist in df["genres"] for g in sublist if g})
selected_genre = st.selectbox("Filter by Genre (optional)", ["All"] + genres)

movie = st.selectbox("Choose a movie you like:", sorted(df["title"].tolist()))

if st.button("Recommend"):
    genre_filter = None if selected_genre == "All" else selected_genre
    recs = recommend_similar_movies(movie, df, sim_matrix, top_n=6, genre_filter=genre_filter)

    if recs.empty:
        st.warning("No recommendations found. Try another movie or genre.")
    else:
        st.subheader("Recommended Movies")
        for _, row in recs.iterrows():
            st.markdown(f"**🎥 {row['title']} ({row['year']})**")
            st.caption(f"⭐ Rating: {row['rating']:.1f}")
            st.markdown(f"<small>{row['overview']}</small>", unsafe_allow_html=True)
            st.markdown("---")
