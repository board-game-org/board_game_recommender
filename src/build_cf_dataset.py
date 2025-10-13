# create_cf_dataset.py

import pandas as pd

# --- Step 1: Load CSVs ---
games = pd.read_csv("../data/games.csv")
user_ratings = pd.read_csv("../data/user_ratings.csv")

# --- Step 2: Standardize column names ---
# games.csv → BGGId = game_id
games.rename(columns={"BGGId": "game_id"}, inplace=True)

# user_ratings.csv → Username = user_id, Rating = rating
user_ratings.rename(columns={"BGGId": "game_id", "Username": "user_id", "Rating": "rating"}, inplace=True)

# --- Step 3: Merge to keep only games that exist in games.csv ---
cf_dataset = pd.merge(user_ratings, games[["game_id"]], on="game_id", how="inner")

# --- Step 4: Keep only required columns ---
cf_dataset = cf_dataset[["user_id", "game_id", "rating"]]

# --- Step 5: Clean data ---
cf_dataset.dropna(subset=["user_id", "game_id", "rating"], inplace=True)

# --- Step 6: Save output ---
cf_dataset.to_csv("../data/cf_dataset.csv", index=False)

print("cf_dataset.csv created successfully!")
print(f"Rows: {len(cf_dataset)}, Users: {cf_dataset['user_id'].nunique()}, Games: {cf_dataset['game_id'].nunique()}")
print(cf_dataset.head())