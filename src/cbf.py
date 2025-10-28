"""
cbf_model.py
Content-Based Filtering model using game features like category, mechanics, and description.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_cbf_scores(user_preferences: dict, games_path: str = "../data/games.csv", top_n: int = 10):
    """
    Compute CBF-based recommendation scores using textual similarity of game features.

    Parameters
    ----------
    user_preferences : dict
        Example: {"category": "Strategy", "mechanics": "Deck Building", "min_players": 2}
    games_path : str
        Path to the games metadata CSV file.
    top_n : int
        Number of top recommendations.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Name, CBF_Score].
    """
    games_df = pd.read_csv(games_path)

    # Build textual tokens from available category indicators and descriptions
    category_columns = [col for col in games_df.columns if col.startswith("Cat:")]
    category_labels = {col: col.split(":", 1)[1] for col in category_columns}

    def _categories_as_text(row):
        tokens = [category_labels[col] for col in category_columns if row.get(col, 0) == 1]
        return " ".join(tokens)

    games_df["category_text"] = games_df.apply(_categories_as_text, axis=1)

    games_df["combined_features"] = (
        games_df.get("Description", "").fillna("").astype(str)
        + " "
        + games_df["category_text"]
        + " "
        + games_df.get("Family", "").fillna("").astype(str)
    ).str.strip()

    # Fit TF-IDF and compute similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(games_df["combined_features"])
    user_text = f"{user_preferences.get('category', '')} {user_preferences.get('mechanics', '')}"
    user_vec = vectorizer.transform([user_text])

    sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
    games_df["CBF_Score"] = sims

    return games_df[["BGGId", "Name", "CBF_Score"]].sort_values("CBF_Score", ascending=False).head(top_n)


if __name__ == "__main__":
    df = get_cbf_scores({"category": "Family", "mechanics": "Card Drafting"})
    print(df)
