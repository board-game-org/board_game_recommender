# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from recommender_openai import recommend_games, category_columns

# App config
st.set_page_config(page_title="Board Game Recommender", layout="wide")

# Title
st.title("Board Game Recommender")
st.markdown("Find the perfect board game using AI and real data.")

# Load game data
@st.cache_data
def load_data():
    df = pd.read_csv("../data/games.csv")
    return df

games_df = load_data()

# --- USER INPUTS ---
st.sidebar.header("Filter Options")
min_players = st.sidebar.number_input("Minimum number of players", min_value=1, max_value=20, value=2)
# Dropdown for categories
category_display = [col.replace("Cat:", "") for col in category_columns]
category = st.sidebar.selectbox("Game category", category_display)

top_n = st.sidebar.slider("Number of AI recommendations", 1, 10, 3)

# --- FILTER GAMES LOCALLY ---
if "MinPlayers" in games_df.columns and "MaxPlayers" in games_df.columns:
    filtered_df = games_df[
        (games_df["MinPlayers"] <= min_players)
        & (games_df["MaxPlayers"] >= min_players)
    ]
else:
    st.warning("Your CSV doesn't have 'MinPlayers' or 'MaxPlayers' columns.")
    filtered_df = games_df

if category:
    category_col = f"Cat:{category}"
    if category_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[category_col] == 1]
    else:
        st.warning(f"Category '{category}' not found in data.")

# --- SHOW LOCAL STATS ---
st.subheader("Game Overview")
st.write(f"Found {len(filtered_df)} games matching your preferences.")

if len(filtered_df) > 0:
    # Show bar chart of games per category
    if "Category" in filtered_df.columns:
        category_counts = (
            filtered_df["Category"]
            .value_counts()
            .head(10)
            .sort_values(ascending=True)
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        category_counts.plot(kind="barh", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("Category")
        ax.set_title("Top Game Categories (Filtered)")
        st.pyplot(fig)

    # Optional: show top few games
    st.dataframe(filtered_df[["Name", "YearPublished", "AvgRating"]].head(10))

# --- OPENAI RECOMMENDATION SECTION ---
st.markdown("---")
st.subheader("AI Recommendations")

if st.button("Get AI Recommendations"):
    with st.spinner("Asking the AI for recommendations..."):
        recommendations = recommend_games(min_players, category, top_n)
        st.markdown("Recommended Games:")
        st.write(recommendations)