
# 🎬 Movie Recommendation System

A content-based movie recommendation system built with **Python**, **TF-IDF**, **cosine similarity**, and the **MovieLens dataset**. The application lets users select a movie and discover other movies with similar genre profiles through a simple Streamlit web interface.

## 🔗 Project Links

- **Live Demo:** https://movie-recommendation-system-qmz8zzvdwppelatdfcaumn.streamlit.app/
- **GitHub Repository:** https://github.com/vaishnaviamane2006-boop/movie-recommendation-system
- **Google Colab Notebook:** https://colab.research.google.com/drive/1r8UIWCrg31kYUD5wF8BRaAB1cI1RBCY_?usp=sharing

## ✨ Features

- Select a movie from the MovieLens dataset.
- Get a ranked list of movies with similar genres.
- Choose how many recommendations to display.
- View movie titles, genres, and cosine similarity scores.
- Download recommendations as a CSV file.
- Explore a simple evaluation table and graph for different recommendation counts.
- Use the application directly in a browser—no local installation is required for the live demo.

## 🧠 How It Works

This project uses **content-based filtering**. It compares movie metadata rather than learning a user's personal preferences.

1. **Load the data:** Read movie information from the MovieLens `movies.csv` file. The dataset also includes ratings in `ratings.csv`.
2. **Preprocess:** Remove duplicate movie IDs and entries without useful genre labels.
3. **TF-IDF vectorization:** Convert each movie's genre text into a numerical feature vector.
4. **Cosine similarity:** Compare the genre vectors to estimate how similar two movies are.
5. **Rank recommendations:** Exclude the selected movie itself and sort the remaining movies by similarity.
6. **Display results:** Show the top recommendations in the Streamlit interface.

### Algorithms

**TF-IDF (Term Frequency–Inverse Document Frequency)** represents text as numerical features, weighting terms according to their occurrence in the collection of genre descriptions.

**Cosine similarity** compares the direction of two feature vectors:

\[
\\text{similarity}(A,B)=\\frac{A\\cdot B}{\\|A\\|\\,\\|B\\|}
\]

A higher score indicates more similar genre representations. It is **not** a rating prediction or a guarantee that a user will enjoy the movie.

## 🛠️ Technology Stack

- **Language:** Python
- **Web application:** Streamlit
- **Data handling:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Visualization:** Matplotlib
- **Dataset:** MovieLens latest-small, provided by GroupLens Research
- **Deployment:** Streamlit Community Cloud
- **Version control:** Git and GitHub

## 📊 Dataset

The application downloads the MovieLens latest-small dataset from GroupLens when it starts.

Dataset page: https://grouplens.org/datasets/movielens/latest/

The main files used are:

- `movies.csv` — movie IDs, titles, and genre labels.
- `ratings.csv` — user ratings and timestamps, available for context or future extensions.

**Current scope:** The recommender uses movie genres to calculate similarity. It does not currently use ratings to personalize recommendations.

## 🚀 Run Locally

### 1. Requirements

Install Python (Python 3.10 or newer is recommended) and Git.

### 2. Clone the repository

```bash
git clone https://github.com/vaishnaviamane2006-boop/movie-recommendation-system.git
cd movie-recommendation-system
```

### 3. Create and activate a virtual environment (recommended)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
```

### 5. Start the application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually `http://localhost:8501`. Open that address in your browser.

An internet connection is needed for the application to download the MovieLens dataset when it is not already cached.

## 📈 Evaluation

The application displays the mean cosine similarity of the top-K recommendations for K = 5, 10, 15, and 20.

This is a **similarity-based diagnostic**, not a full measure of recommendation accuracy or user satisfaction. More comprehensive evaluation could use relevance labels and metrics such as Precision@K or Recall@K.

## ⚠️ Limitations

- Recommendations are based on genres only.
- Movies sharing genres may still differ in story, tone, and quality.
- The current model does not personalize results using an individual user's rating history.
- Similarity scores should not be interpreted as probabilities or quality ratings.

## 🔮 Future Enhancements

- Add collaborative filtering based on user ratings.
- Combine genres with tags, plot summaries, or other metadata.
- Display movie posters and additional details.
- Add user accounts, watchlists, and saved favorites.
- Evaluate recommendations with relevance-based metrics.

## 👩‍💻 Author

**Vaishnavi Mane**

College mini-project: Movie Recommendation System

## 📄 Acknowledgements

- [GroupLens Research](https://grouplens.org/) for the MovieLens dataset.
- The developers and communities behind Python, Streamlit, pandas, NumPy, scikit-learn, and Matplotlib.

---

If you find this project useful for learning, feel free to explore the code and adapt it for your own experiments.
