import pandas as pd

# Load the two CSV files
larger = pd.read_csv("../data/games_master_data.csv")
smaller = pd.read_csv("../data/bgg_master_data_smaller.csv")

# Identify the common key column to compare
key_col = 'bgg_id'

# Find games present in the larger file but missing in the smaller one
missing_games = larger[~larger[key_col].isin(smaller[key_col])]

# Save missing games to a new CSV
missing_games.to_csv("../data/missing_games.csv", index=False)

print(f"Found {len(missing_games)} missing games. Saved to 'missing_games.csv'.")