import streamlit as st
from recommender import (
    fetch_movies_2000_2025,
    fetch_genre_map,
    build_similarity_model,
    recommend_similar_movies
)

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")

st.markdown("""
<style>
.movie-title {
    font-size: 18px;
    font-weight: 600;
}
.movie-meta {
    font-size: 13px;
    color: #888;
}
.movie-overview {
    font-size: 13px;
    line-height: 1.5;
}
.block {
    padding: 12px;
    border-radius: 12px;
    background: #111;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 AI-Powered Movie Recommendation System")
st.caption("Content-based recommendations using AI embeddings (2000–2025)")

@st.cache_data(show_spinner="Fetching movies from TMDB...")
def load_data():
    return fetch_movies_2000_2025(pages=8)

@st.cache_resource(show_spinner="Building AI similarity model...")
def load_model(df):
    return build_similarity_model(df)

# Load data
df = load_data()
model = load_model(df)

# Genre map (ensures Action always appears)
genre_map = fetch_genre_map()
all_genres = ["All"] + sorted(genre_map.values())

# UI controls
col1, col2 = st.columns([2, 1])

with col1:
    movie_title = st.selectbox(
        "🎥 Select a movie you like",
        options=sorted(df["title"].dropna().unique().tolist())
    )

with col2:
    selected_genre = st.selectbox(
        "🎭 Filter by genre (optional)",
        options=all_genres
    )

if st.button("✨ Recommend Movies"):
    with st.spinner("Finding similar movies..."):
        genre_filter = None if selected_genre == "All" else selected_genre
        recs = recommend_similar_movies(
            movie_title,
            df,
            model,
            genre_filter=genre_filter,
            top_n=10
        )

    if not recs:
        st.warning("No similar movies found. Try another movie or remove the genre filter.")
    else:
        st.subheader("🍿 Recommended for you")

        for i, row in enumerate(recs, start=1):
            st.markdown(f"""
            <div class="block">
                <div class="movie-title">{i}. {row['title']} ({int(row.get('year', 0))})</div>
                <div class="movie-meta">⭐ Rating: {row.get('rating', 'N/A')} | 🎭 Genres: {row.get('genres_text', 'N/A')}</div>
                <div class="movie-overview">{row.get('overview', 'No description available.')}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + AI Embeddings")
        
