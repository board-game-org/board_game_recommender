import numpy as np
import pandas as pd
from typing import Union, Tuple

def get_hybrid_recommendations(
    games_df: pd.DataFrame,
    cf_scores: Union[np.ndarray, list],
    cbf_scores: Union[np.ndarray, list],
    llm_scores: Union[np.ndarray, list],
    liked_games=None,
    disliked_games=None,
    exclude_games=None,
    attributes=None,
    alpha: float = 0.5,
    beta: float = 0.33,
    n_recommendations: int = 5
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute hybrid recommender scores from CF, CBF, and LLM models,
    apply user filters, and return top recommendations and their scores.

    Returns:
        recommendations (DataFrame),
        hybrid_scores_topN,
        cf_component_topN,
        cbf_component_topN,
        llm_component_topN
    """

    # --- Convert and validate input ---
    cf_scores = np.array(cf_scores)
    cbf_scores = np.array(cbf_scores)
    llm_scores = np.array(llm_scores)
    final_scores = cf_scores.copy()

    liked_games = liked_games or []
    disliked_games = disliked_games or []
    exclude_games = exclude_games or []
    attributes = attributes or {}

    # --- Handle missing or zero-score cases ---
    cf_zero = np.all(cf_scores == 0)
    cbf_zero = np.all(cbf_scores == 0)
    llm_zero = np.all(llm_scores == 0)

    if cf_zero and cbf_zero:
        beta = 1.0  # rely entirely on LLM
    elif llm_zero:
        beta = 0.0  # rely entirely on CF/CBF

    if cf_zero and not cbf_zero:
        alpha = 0.0
    elif cbf_zero and not cf_zero:
        alpha = 1.0

    # --- Compute hybrid components ---
    cf_component = cf_scores * alpha
    cbf_component = cbf_scores * (1 - alpha)
    combined_cf_cbf = (cf_component + cbf_component) * (1 - beta)
    llm_component = llm_scores * beta
    hybrid_scores = combined_cf_cbf + llm_component

    final_scores = hybrid_scores.copy()

    # --- Apply exclusion filters ---
    for gid in liked_games + disliked_games + exclude_games:
        if gid in games_df.index:
            idx = games_df.index.get_loc(gid)
            final_scores[idx] = 0

    # --- Apply attribute filters ---
    if attributes:
        for attr_name in ['game_mechanics', 'game_categories', 'game_types']:
            selected = attributes.get(attr_name, [])
            if selected:
                selected_clean = [s.strip() for s in selected if isinstance(s, str)]
                mask = games_df[attr_name].apply(
                    lambda game_attrs: any(
                        a.strip() in selected_clean for a in game_attrs if isinstance(a, str)
                    )
                )
                final_scores[~mask] = 0

        if 'game_weight' in attributes:
            w_min, w_max = attributes['game_weight']
            mask = (games_df['game_weight'] >= w_min) & (games_df['game_weight'] <= w_max)
            final_scores[~mask] = 0

        if 'players' in attributes:
            p_min, p_max = attributes['players']
            mask = (games_df['players_max'] >= p_min) & (games_df['players_min'] <= p_max)
            final_scores[~mask] = 0

        if 'play_time' in attributes:
            t_min, t_max = attributes['play_time']
            mask = (games_df['time_max'] >= t_min) & (games_df['time_min'] <= t_max)
            final_scores[~mask] = 0

        if 'year_published' in attributes:
            y_min, y_max = attributes['year_published']
            mask = (games_df['year_published'] >= y_min) & (games_df['year_published'] <= y_max)
            final_scores[~mask] = 0

        if 'min_rating' in attributes:
            min_rating = attributes['min_rating']
            if isinstance(min_rating, (list, tuple)):
                min_rating = min_rating[0]
            mask = (games_df['avg_rating'] >= min_rating)
            final_scores[~mask] = 0

    # --- Select top N recommendations ---
    valid_idx = np.where(final_scores >= 0.01)[0]
    top_n_idx = valid_idx[np.argsort(final_scores[valid_idx])[::-1][:n_recommendations]]

    recommendations = games_df.iloc[top_n_idx][[
        'bgg_id', 'name', 'avg_rating', 'game_categories',
        'game_mechanics', 'game_weight', 'game_types',
        'year_published', 'players_min', 'players_max'
    ]].copy()

    recommendations['score'] = final_scores[top_n_idx].round(4)

    # --- Return scores for top N games only ---
    hybrid_scores_topN = hybrid_scores[top_n_idx]
    cf_component_topN = cf_component[top_n_idx]
    cbf_component_topN = cbf_component[top_n_idx]
    llm_component_topN = llm_component[top_n_idx]

    return recommendations, hybrid_scores_topN, cf_component_topN, cbf_component_topN, llm_component_topN
