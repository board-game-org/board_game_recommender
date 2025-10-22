import pandas as pd
import streamlit as st

# === Load Data === #
DATA_PATH = "../data/games.csv"
st.title("Board Game Explorer")

st.header("Load Dataset")
try:
    games_df = pd.read_csv(DATA_PATH, low_memory=False)
    games_df = games_df.convert_dtypes()
    st.success(f"Loaded {len(games_df)} games from {DATA_PATH}")
except FileNotFoundError:
    st.error(f"Could not find {DATA_PATH}. Please check the file path.")
    st.stop()

# === User Preferences === #
st.header("Filter Games by Preference")

# Number of players filter
min_players = st.number_input("Minimum number of players", min_value=1, max_value=20, value=2)
max_players = st.number_input("Maximum number of players", min_value=min_players, max_value=20, value=4)

# Rating filter
min_rating = st.slider("Minimum average rating", 0.0, 10.0, 7.0)

# Category filter
cat_cols = [col for col in games_df.columns if col.startswith("Cat:")]
if cat_cols:
    cat_names = [c.replace("Cat:", "") for c in cat_cols]
    selected_category = st.selectbox("Select game category", ["All"] + cat_names)
else:
    selected_category = "All"

# Apply filters
filtered_df = games_df[
    (games_df["MinPlayers"] <= max_players)
    & (games_df["MaxPlayers"] >= min_players)
    & (games_df["AvgRating"] >= min_rating)
]

if selected_category != "All":
    cat_col = f"Cat:{selected_category}"
    filtered_df = filtered_df[filtered_df[cat_col] == 1]

st.write(f"Found {len(filtered_df)} games matching your preferences.")
st.dataframe(filtered_df[["Name", "AvgRating", "MinPlayers", "MaxPlayers"]].head(10))

# === Visualization === #
st.header("Game Counts by Category")
if cat_cols:
    category_counts = games_df[cat_cols].sum().sort_values(ascending=False)
    category_counts.index = [c.replace("Cat:", "") for c in category_counts.index]
    st.bar_chart(category_counts)
else:
    st.info("No category columns found (columns starting with 'Cat:'). Displaying total game count instead.")
    st.metric("Total Games in Dataset", len(games_df))