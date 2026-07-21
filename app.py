import streamlit as st
import pandas as pd
import numpy as np

from src.utils import build_recommender
from src.data_loader import load_data
from src.preprocessing import preprocess_data

import os
import random
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Steam Game Recommendation System",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# DARK MODE STYLE
# -----------------------------
st.markdown("""
<style>

.stApp{
    background-color:#0E1117;
}

.hero{
    background:linear-gradient(90deg,#2563EB,#1D4ED8,#1E3A8A);
    padding:30px;
    border-radius:18px;
    color:white;
    text-align:center;
    margin-bottom:25px;
}

.hero h1{
    font-size:42px;
}

.hero p{
    font-size:18px;
    color:#E5E7EB;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def get_data():
    df = load_data()
    df = preprocess_data(df)
    return df

df = get_data()

@st.cache_resource
def get_recommender():
    return build_recommender()

recommender = get_recommender()
# -----------------------------
# GAME CARD COMPONENT
# -----------------------------
def game_card(row):

    rating = row["rating_score"] * 100
    playtime = row["average_playtime"]
    appid = row["appid"]
    steam_url = f"https://store.steampowered.com/app/{appid}"

    # Official Steam Header Image
    image_url = f"https://shared.cloudflare.steamstatic.com/store_item_assets/steam/apps/{appid}/header.jpg"

    # Format playtime
    if playtime < 60:
        playtime_text = f"{playtime:.0f} min"
    else:
        playtime_text = f"{playtime/60:.1f} hrs"

    html = f"""
    <div style="
        background:#1F2937;
        padding:20px;
        border-radius:18px;
        border:1px solid #3B82F6;
        margin-bottom:20px;
        box-shadow:0px 6px 18px rgba(0,0,0,.35);
    ">

        <img
    src="{image_url}"
    onerror="this.style.display='none';"
    style="
        width:100%;
        border-radius:12px;
        margin-bottom:18px;
    "
>

        <h2 style="margin-bottom:15px;">
    <a
        href="{steam_url}"
        target="_blank"
        style="
            color:white;
            text-decoration:none;
        ">
        🎮 {row["name"]}
    </a>
</h2>

        <hr style="
            border:1px solid #374151;
            margin-bottom:15px;
        ">

        <p style="color:#FACC15;font-size:17px;">
            ⭐ <b>Rating:</b> {rating:.0f}%
        </p>

        <p style="color:#60A5FA;font-size:16px;">
            🎯 <b>Genre:</b> {row["genres"]}
        </p>

        <p style="color:#A78BFA;font-size:16px;">
            💻 <b>Platform:</b> {row["platforms"]}
        </p>

        <p style="color:#FB7185;font-size:16px;">
            📅 <b>Year:</b> {row["year"]}
        </p>

        <p style="color:#4ADE80;font-size:16px;">
            💰 <b>Price:</b> ${row["price"]}
        </p>

        <p style="color:#22D3EE;font-size:16px;">
            ⏱ <b>Playtime:</b> {playtime_text}
        </p>

    </div>
    """

    st.html(html)

# -----------------------------
# LOGGING FUNCTIONS
# -----------------------------
from src.config import LOG_FILE

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
st.markdown("""
<div class="hero">

<h1>🎮 Steam Game Recommendation System</h1>

<p>
Discover your next favourite game using
Machine Learning & AI
</p>

<p>
Python • Streamlit • TF-IDF • Cosine Similarity
</p>

</div>
""", unsafe_allow_html=True)

# -----------------------------
# DASHBOARD METRICS
# -----------------------------

st.markdown("## 📊 Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("🎮 Games", len(df))

with col2:
    st.metric("🎯 Genres", df["genres"].nunique())

with col3:
    st.metric("💻 Platforms", df["platforms"].nunique())

with col4:
    st.metric(
        "⭐ Avg Rating",
        f"{df['rating_score'].mean()*100:.0f}%"
    )

with col5:
    free_games = (df["price"] == 0).sum()
    st.metric("💰 Free Games", free_games)

# -----------------------------
# TRENDING GAMES (GRID)
# -----------------------------
# -----------------------------
# TRENDING GAMES (GRID)
# -----------------------------
st.markdown("## 🔥 Trending Games")

top_games = df.sort_values(by="rating_score", ascending=False).head(6)

cols = st.columns(3)

for i, (_, row) in enumerate(top_games.iterrows()):

    with cols[i % 3]:

        image_url = f"https://shared.cloudflare.steamstatic.com/store_item_assets/steam/apps/{row['appid']}/header.jpg"

        st.html(f"""
        <div style="
            background:#1F2937;
            padding:18px;
            border-radius:15px;
            margin-bottom:20px;
            border:1px solid #3B82F6;
            box-shadow:0px 5px 12px rgba(0,0,0,.35);
        ">

            <img
                src="{image_url}"
                style="
                    width:100%;
                    border-radius:10px;
                    margin-bottom:12px;
                "
                onerror="this.style.display='none';"
            >

            <h4 style="color:white;">
                🎮 {row['name']}
            </h4>

            <p style="color:#FACC15;">
                ⭐ {row['rating_score']*100:.0f}%
            </p>

            <p style="color:#60A5FA;">
                🎯 {row['genres']}
            </p>

            <p style="color:#4ADE80;">
                💰 ${row['price']}
            </p>

        </div>
        """)

# -----------------------------
# SEARCH SECTION
# -----------------------------
st.markdown("## 🔍 Find Similar Games")

col1, col2 = st.columns([3,1])

with col1:
    game_name = st.selectbox(
    "🎮 Search Game",
    sorted(df["name"].unique()),
    index=None,
    placeholder="Search for a game..."
)

with col2:
    top_n = st.slider("Results", 1, 10, 5)

# -----------------------------
# BUTTON
# -----------------------------
if st.button("🚀 Recommend Games"):

    results = recommender.recommend(
        game_name,
        top_n=top_n
    )

    with st.spinner("Finding similar games..."):
        results = recommender.recommend(
            game_name,
            top_n=top_n
        )

    if results.empty:
        st.error("Game not found ❌")

    else:

        recommended_games = results["name"].tolist()

        log_recommendations(
            game_name,
            recommended_games
        )

        st.markdown("## 🎯 Recommended Games")

        cols = st.columns(2)

        for i, (_, row) in enumerate(results.iterrows()):

            with cols[i % 2]:
                game_card(row)