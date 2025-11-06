"""
cf_model.py
Collaborative Filtering model for board game recommendations.
This module computes similarity based on user ratings.
"""

import pandas as pd
import numpy as np

def fold_in_implicit_user(V, liked_items, alpha=5, lambda_=0.03):
    """
    Compute a new user vector given items they've liked (implicit feedback).
    """
    V_i = V[liked_items]
    # confidence weights
    C_i = 1 + alpha * np.ones(len(liked_items), dtype=np.float32)
    
    A = V_i.T @ (C_i[:, None] * V_i) + lambda_ * np.eye(V.shape[1])
    b = V_i.T @ (C_i * np.ones(len(liked_items), dtype=np.float32))
    
    u_new = np.linalg.solve(A, b)
    return u_new


def get_cf_scores(
    ratings: np.ndarray = np.array([]),
    #games_path: str = "../data/games.csv",
):
    """
    Compute CF-based recommendation scores based on pre-computed item embedding matrix V and a vector of movie IDs of user likes

    Parameters
    ----------
    ratings : array
        array of indices of liked items

    Returns
    -------
    scores
        array of ratings for each board game
    """

    #load board game embeddings
    V = np.load('../data/V_final.npy')
    
    # calculte user embeddings based on inputted likes 
    u = fold_in_implicit_user(V,liked_items=ratings, alpha=5, lambda_=0.3)

    #calculate scores
    scores = V.dot(u)

    #normalize between 0 and 1 
    scores = (scores - min(scores)) / (max(scores) - min(scores))

    # returns array of scores per movie
    return scores 

