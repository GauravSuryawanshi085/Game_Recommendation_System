"""
Utility Module
--------------
This module provides helper functions to
build and initialize the recommendation system.
"""

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.recommender import GameRecommender


def build_recommender():
    """
    Build the complete recommendation system.

    Returns
    -------
    GameRecommender
    """

    # Load raw dataset
    df = load_data()

    # Apply preprocessing
    df = preprocess_data(df)

    # Build recommendation engine
    recommender = GameRecommender(df)

    return recommender