import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import os
import random
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Game Recommender", layout="wide")

# -----------------------------
# DARK MODE STYLE
# -----------------------------
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("steam.csv")

    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year

    df['owners'] = df['owners'].str.split('-').str[0]
    df['owners'] = pd.to_numeric(df['owners'], errors='coerce')

    df['rating_score'] = df['positive_ratings'] / (
        df['positive_ratings'] + df['negative_ratings']
    )

    df.fillna('', inplace=True)

    def clean_text(text):
        return str(text).lower().replace(';', ' ')

    for col in ['genres', 'categories', 'steamspy_tags', 'platforms']:
        df[col] = df[col].apply(clean_text)

    df['combined'] = (
        df['genres'] + ' ' +
        df['categories'] + ' ' +
        df['steamspy_tags'] + ' ' +
        df['platforms']
    )

    df = df[['name', 'combined', 'rating_score', 'owners', 'average_playtime', 'price']]

    return df

df = load_data()

# -----------------------------
# NORMALIZATION
# -----------------------------
scaler = MinMaxScaler()
df[['rating_score', 'owners', 'average_playtime']] = scaler.fit_transform(
    df[['rating_score', 'owners', 'average_playtime']]
)

# -----------------------------
# TF-IDF
# -----------------------------
@st.cache_resource
def create_tfidf(data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    tfidf_matrix = tfidf.fit_transform(data['combined']).astype('float32')
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = create_tfidf(df)

# -----------------------------
# INDEX MAPPING
# -----------------------------
indices = pd.Series(df.index, index=df['name']).drop_duplicates()

# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------
def recommend(game_name, top_n=5):

    if game_name not in indices:
        return pd.DataFrame()

    idx = indices[game_name]

    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    scores = []

    for i, sim in enumerate(sim_scores):
        if i == idx:
            continue

        score = (
            sim * 0.6 +
            df.iloc[i]['rating_score'] * 0.2 +
            df.iloc[i]['owners'] * 0.1 +
            df.iloc[i]['average_playtime'] * 0.1
        )

        scores.append((i, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in scores[:top_n]]

    return df.iloc[top_indices]

# -----------------------------
# LOGGING FUNCTIONS
# -----------------------------
LOG_FILE = "steam.csv"

def log_recommendations(input_game, recommended_games):
    user_id = random.randint(1, 500)

    rows = []
    for game in recommended_games:
        rows.append({
            "user_id": user_id,
            "input_game": input_game,
            "recommended_game": game,
            "clicked": 0,
            "timestamp": datetime.now()
        })

    log_df = pd.DataFrame(rows)

    if os.path.exists(LOG_FILE):
        log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LOG_FILE, index=False)


def update_click(input_game, recommended_game):
    if not os.path.exists(LOG_FILE):
        return

    df_log = pd.read_csv(LOG_FILE)

    mask = (
        (df_log["input_game"] == input_game) &
        (df_log["recommended_game"] == recommended_game) &
        (df_log["clicked"] == 0)
    )

    if mask.any():
        idx = df_log[mask].index[0]
        df_log.loc[idx, "clicked"] = 1
        df_log.to_csv(LOG_FILE, index=False)

# -----------------------------
# UI
# -----------------------------
st.title("🎮 Game Recommendation System")
st.markdown("### Discover Your Next Favorite Game 🚀")

# -----------------------------
# TRENDING GAMES (GRID)
# -----------------------------
st.markdown("## 🔥 Trending Games")

top_games = df.sort_values(by='rating_score', ascending=False).head(6)

cols = st.columns(3)

for i, (_, row) in enumerate(top_games.iterrows()):
    with cols[i % 3]:
        st.markdown(f"""
        <div style="
            background-color:#1e1e1e;
            padding:15px;
            border-radius:12px;
            margin-bottom:15px;
        ">
            <h4 style="color:white;">🎮 {row['name']}</h4>
            <p style="color:#bbbbbb;">⭐ Rating: {round(row['rating_score'],2)}</p>
            <p style="color:#bbbbbb;">💰 Price: ${row['price']}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# SEARCH SECTION
# -----------------------------
st.markdown("## 🔍 Find Similar Games")

col1, col2 = st.columns([3,1])

with col1:
    game_name = st.selectbox("Choose a Game", df['name'].values)

with col2:
    top_n = st.slider("Results", 1, 10, 5)

# -----------------------------
# BUTTON
# -----------------------------
if st.button("🚀 Recommend Games"):

    results = recommend(game_name, top_n)

    if results.empty:
        st.error("Game not found ❌")

    else:
        # convert dataframe to list of game names
        recommended_games = results['name'].tolist()

        # LOG DATA
        log_recommendations(game_name, recommended_games)

        st.markdown("## 🎯 Recommended Games")

        cols = st.columns(2)

        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 2]:

                st.markdown(f"""
                <div style="
                    background-color:#262730;
                    padding:15px;
                    border-radius:12px;
                    margin-bottom:15px;
                ">
                    <h4 style="color:white;">🎮 {row['name']}</h4>
                    <p style="color:#bbbbbb;">⭐ Rating: {round(row['rating_score'],2)}</p>
                    <p style="color:#bbbbbb;">💰 Price: ${row['price']}</p>
                </div>
                """, unsafe_allow_html=True)

                # CLICK BUTTON
                if st.button(f"Play {row['name']}", key=f"{game_name}_{row['name']}_{i}"):
                    update_click(game_name, row['name'])
                    st.success(f"You clicked on {row['name']}")