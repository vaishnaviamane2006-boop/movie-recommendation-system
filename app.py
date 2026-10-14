
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


@st.cache_data
def load_movies():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = "/tmp/movies.zip"
    extract_path = "/tmp/movie_data"

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_path)

    folder = os.path.join(extract_path, "ml-latest-small")

    movies = pd.read_csv(os.path.join(folder, "movies.csv"))
    ratings = pd.read_csv(os.path.join(folder, "ratings.csv"))

    movies = movies.drop_duplicates(subset="movieId").copy()
    movies["genres"] = movies["genres"].fillna("")
    movies = movies[
        (movies["genres"] != "(no genres listed)") &
        (movies["genres"] != "")
    ].reset_index(drop=True)

    return movies, ratings


@st.cache_resource
def build_model(genres):
    vectorizer = TfidfVectorizer(
        token_pattern=r"[^|]+",
        lowercase=True
    )
    matrix = vectorizer.fit_transform(genres)
    similarity = cosine_similarity(matrix, dense_output=False)
    return similarity


st.title("🎬 Movie Recommendation System")
st.write(
    "Discover movies similar to your favorite films "
    "using machine learning."
)
st.caption("College Mini Project | Content-Based Filtering")

with st.spinner("Loading movies and building the recommendation model..."):
    movies, ratings = load_movies()
    similarity_matrix = build_model(tuple(movies["genres"].tolist()))

st.success(f"Model ready! {len(movies):,} movies loaded.")

selected_movie = st.selectbox(
    "Choose a movie",
    movies["title"].tolist(),
    index=(
        movies["title"].tolist().index("Toy Story (1995)")
        if "Toy Story (1995)" in movies["title"].tolist()
        else 0
    )
)

number = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10,
    step=5
)


def recommend(movie_title, count):
    movie_index = movies.index[
        movies["title"] == movie_title
    ][0]

    scores = similarity_matrix[movie_index].toarray().ravel()
    scores[movie_index] = -1

    ranked = np.argsort(scores)[::-1][:count]

    results = movies.iloc[ranked][["title", "genres"]].copy()
    results["Similarity Score"] = np.round(scores[ranked], 3)

    return results.reset_index(drop=True)


if st.button("Get Recommendations", type="primary"):
    results = recommend(selected_movie, number)

    st.subheader(f"Movies similar to {selected_movie}")
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download recommendations as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="movie_recommendations.csv",
        mime="text/csv"
    )

st.divider()
st.subheader("Model Evaluation")

evaluation = []
movie_index = movies.index[
    movies["title"] == selected_movie
][0]

scores = similarity_matrix[movie_index].toarray().ravel()
scores[movie_index] = -1

for k in [5, 10, 15, 20]:
    top_indices = np.argsort(scores)[::-1][:k]

    evaluation.append({
        "Number of Recommendations": k,
        "Mean Similarity Score": round(
            float(scores[top_indices].mean()), 3
        )
    })

evaluation_df = pd.DataFrame(evaluation)
st.dataframe(evaluation_df, use_container_width=True)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(
    evaluation_df["Number of Recommendations"],
    evaluation_df["Mean Similarity Score"],
    marker="o"
)
ax.set_title("Recommendation System Evaluation")
ax.set_xlabel("Number of Recommendations")
ax.set_ylabel("Mean Similarity Score")
ax.grid(True, alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Note: Similarity scores measure genre similarity, "
    "not prediction accuracy or user satisfaction."
)
