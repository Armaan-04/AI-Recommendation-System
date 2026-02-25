import streamlit as st
from recommender import fetch_movies_2000_2025, fetch_genre_map, build_similarity_model, recommend_similar_movies

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI-Powered Movie Recommendation System (2000–2025)")

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

movie_list = sorted(df["title"].dropna().unique().tolist())
selected_movie = st.selectbox("🎥 Select a movie:", movie_list)

all_genres = sorted({g for gs in df["genres_text"] for g in gs.split(", ") if g})
selected_genres = st.multiselect("🎭 Filter by genres (optional):", all_genres)

if st.button("🔥 Recommend"):
    results = recommend_similar_movies(
        selected_movie, df, sim_matrix, top_n=10, selected_genres=selected_genres
    )

    if results:
        st.subheader("✨ Recommended Movies")
        for r in results:
            st.markdown(f"""
**🎬 {r['title']}**  
⭐ Rating: {r['rating']}  
🎭 Genres: {r['genres']}  
🧠 Match Score: {r['score']}  

_{r['overview']}_  
---
""")
    else:
        st.warning("No good matches found. Try changing genres.")
        
