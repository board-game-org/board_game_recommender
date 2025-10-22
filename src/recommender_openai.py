# llm_recommender.py
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Initialize client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load game data
games_df = pd.read_csv("../data/games.csv")

# Extract all category columns automatically
category_columns = [col for col in games_df.columns if col.startswith("Cat:")]

def recommend_games(min_players: int, category: str, top_n: int = 5):
    """
    Filter games by min_players and category, then ask OpenAI for top recommendations.
    """
    # Convert user-facing name (e.g. "Strategy") to column name (e.g. "Cat:Strategy")
    category_col = f"Cat:{category}"

    if category_col not in games_df.columns:
        return [f"Invalid category '{category}'. Please choose another one."]

    # Filter dataset
    filtered_df = games_df[
        (games_df["MinPlayers"] <= min_players)
        & (games_df["MaxPlayers"] >= min_players)
        & (games_df[category_col] == 1)
    ]

    if filtered_df.empty:
        return ["No games found with those filters. Try adjusting your criteria."]

    # Take top 20 by rating to reduce tokens
    top_games = filtered_df.sort_values("AvgRating", ascending=False).head(20)

    # Prepare text to send to model
    descriptions = "\n\n".join([
        f"Name: {row['Name']}\nDescription: {row['Description']}"
        for _, row in top_games.iterrows()
    ])

    prompt = f"""
    You are a board game recommender.
    The user wants a game for {min_players} players in the '{category}' category.
    From the following list, recommend the top {top_n} games and explain why briefly.

    {descriptions}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert board game advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()