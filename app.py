import streamlit as st
from recommender import fetch_movies_2000_2025, fetch_genre_map, build_similarity_model, recommend_similar_movies

st.set_page_config(
    page_title="🎬 AI Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.movie-card {
    background: #0f1117;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    border: 1px solid #222;
}
.movie-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 4px;
}
.movie-meta {
    font-size: 13px;
    color: #aaa;
    margin-bottom: 8px;
}
.movie-overview {
    font-size: 14px;
    line-height: 1.5;
    color: #ddd;
}
.badge {
    display: inline-block;
    background: #1f2933;
    color: #9ca3af;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 11px;
    margin-right: 6px;
}
.poster {
    border-radius: 10px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 AI Movie Recommendation System")
st.caption("Smarter recommendations using semantic AI embeddings (2000–2025)")

# ---------- Data ----------
@st.cache_data
def load_data():
    df = fetch_movies_2000_2025(pages=5)
    genre_map = fetch_genre_map()
    df["genres_text"] = df["genre_ids"].apply(
        lambda ids: ", ".join([genre_map.get(i, "Unknown") for i in ids])
    )
    return df

@st.cache_resource
def build_model(df):
    return build_similarity_model(df)

df = load_data()
sim_matrix = build_model(df)

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("🎯 Filters")
    movie_list = sorted(df["title"].dropna().unique().tolist())
    selected_movie = st.selectbox("Select a movie", movie_list)

    all_genres = sorted({g for gs in df["genres_text"] for g in gs.split(", ") if g})
    selected_genres = st.multiselect("Filter by genres (optional)", all_genres)

    recommend_btn = st.button("🔥 Recommend")

# ---------- Main Results ----------
if recommend_btn:
    results = recommend_similar_movies(
        selected_movie, df, sim_matrix, top_n=10, selected_genres=selected_genres
    )

    st.subheader("✨ Recommended Movies")

    for r in results:
        col1, col2 = st.columns([1, 3])

        with col1:
            poster_url = f"https://image.tmdb.org/t/p/w500{r.get('poster_path', '')}"
            if r.get("poster_path"):
                st.image(poster_url, use_container_width=True)
            else:
                st.image("https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)

        with col2:
            st.markdown(f"""
<div class="movie-card">
    <div class="movie-title">{r['title']}</div>
    <div class="movie-meta">
        ⭐ {r['rating']} &nbsp; • &nbsp; 🧠 Match: {r['score']}
    </div>
    <div class="movie-meta">
        {''.join([f"<span class='badge'>{g}</span>" for g in r['genres'].split(", ")])}
    </div>
    <div class="movie-overview">
        {r['overview']}
    </div>
</div>
""", unsafe_allow_html=True)

else:
    st.info("👈 Select a movie and click Recommend to get AI-powered suggestions.")
        
