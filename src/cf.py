"""
cf_model.py
Collaborative Filtering model for board game recommendations.
This module computes similarity based on user ratings.
"""

import pandas as pd
import numpy as np

def get_cf_scores(
    user_id: int,
    ratings_path: str = "../data/user_ratings.csv",
    games_path: str = "../data/games.csv",
    top_n: int = 10
):
    """
    Compute CF-based recommendation scores for a user based on other similar users' ratings.

    Parameters
    ----------
    user_id : int
        ID of the target user.
    ratings_path : str
        Path to the user-game ratings CSV file.
    games_path : str
        Path to the games metadata CSV (used to map names).
    top_n : int
        Number of top recommended games to return.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Name, CF_Score].
    """
    ratings_df = pd.read_csv(ratings_path)
    games_df = pd.read_csv(games_path)[["BGGId", "Name"]].drop_duplicates()

    # Placeholder logic (to be replaced)
    cf_scores_df = (
        ratings_df.groupby("BGGId")["Rating"].mean().reset_index()
    )
    cf_scores_df.rename(columns={"Rating": "CF_Score"}, inplace=True)
    max_score = cf_scores_df["CF_Score"].max()
    if max_score:
        cf_scores_df["CF_Score"] = cf_scores_df["CF_Score"] / max_score

    cf_scores_df = cf_scores_df.merge(games_df, on="BGGId", how="left")
    cf_scores_df = cf_scores_df[["BGGId", "Name", "CF_Score"]]

    return cf_scores_df.sort_values("CF_Score", ascending=False).head(top_n)


if __name__ == "__main__":
    df = get_cf_scores(user_id=123)
    print(df)
