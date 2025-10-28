"""
model_ensemble.py
Combines CF, CBF, and LLM scores using a hybrid weighting scheme.
"""

import pandas as pd
from cf import get_cf_scores
from cbf import get_cbf_scores
from llm import get_llm_scores

def ensemble_scores(
    user_id: int,
    user_description: str,
    user_preferences: dict,
    alpha: float = 0.4,
    beta: float = 1.2,
    top_n: int = 10
):
    """
    Ensemble CF, CBF, and LLM models using a hybrid weighting formula.

    Parameters
    ----------
    user_id : int
        ID of target user.
    user_description : str
        Free-text description of desired game.
    user_preferences : dict
        Preferences like category and mechanics.
    alpha : float
        Controls the CF vs CBF weighting within the hybrid score.
    beta : float
        Determines the influence of the LLM score relative to CF/CBF.
    top_n : int
        Number of recommendations.

    Returns
    -------
    pd.DataFrame
        Combined recommendations with composite score.
    """
    cf_df = get_cf_scores(user_id=user_id, top_n=top_n)
    cbf_df = get_cbf_scores(user_preferences=user_preferences, top_n=top_n)
    llm_df = get_llm_scores(
        user_description=user_description,
        min_players=user_preferences.get("min_players", 2),
        category=user_preferences.get("category", "Strategy"),
        top_n=top_n
    )

    score_frames = [
        cf_df[["BGGId", "CF_Score"]],
        cbf_df[["BGGId", "CBF_Score"]],
        llm_df[["BGGId", "LLM_Score"]],
    ]
    merged = score_frames[0].set_index("BGGId")
    for frame in score_frames[1:]:
        merged = merged.join(frame.set_index("BGGId"), how="outer")
    merged.fillna(0, inplace=True)
    merged.reset_index(inplace=True)

    games_lookup = pd.read_csv("../data/games.csv")[["BGGId", "Name"]].drop_duplicates()
    merged = merged.merge(games_lookup, on="BGGId", how="left")

    # Compute ensemble score using hybrid formula:
    # score_hybrid = (CF + (1 - alpha) * CBF) * (beta - 1) + LLM
    combined_cf_cbf = merged["CF_Score"] + (1 - alpha) * merged["CBF_Score"]
    merged["Composite_Score"] = combined_cf_cbf * (beta - 1) + merged["LLM_Score"]

    final_df = merged.sort_values("Composite_Score", ascending=False).head(top_n)
    return final_df[["BGGId", "Name", "CF_Score", "CBF_Score", "LLM_Score", "Composite_Score"]]

if __name__ == "__main__":
    user_prefs = {"category": "Strategy", "mechanics": "Deck Building", "min_players": 3}
    result = ensemble_scores(
        user_id=123,
        user_description="I love competitive strategy games with some luck element.",
        user_preferences=user_prefs
    )
    print(result)
