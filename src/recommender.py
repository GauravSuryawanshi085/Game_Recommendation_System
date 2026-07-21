"""
Recommender Module
------------------
This module builds the recommendation engine
using TF-IDF and Cosine Similarity.
"""

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import MAX_FEATURES


class GameRecommender:

    def __init__(self, dataframe: pd.DataFrame):

        self.df = dataframe.copy()

        # Normalize numerical columns
        scaler = MinMaxScaler()

        self.df[
            [
                "rating_score",
                "owners",
                "average_playtime",
            ]
        ] = scaler.fit_transform(
            self.df[
                [
                    "rating_score",
                    "owners",
                    "average_playtime",
                ]
            ]
        )

        # Build TF-IDF model
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=MAX_FEATURES,
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["combined"]
        )

        # Create game index
        self.df["name"] = self.df["name"].astype(str).str.strip()

        self.indices = pd.Series(
            data=self.df.index,
            index=self.df["name"]
        )

        # Keep first occurrence if duplicates exist
        self.indices = self.indices[~self.indices.index.duplicated(keep="first")]

    def recommend(self, game_name: str, top_n: int = 5):

        game_name = game_name.strip()

        if game_name not in self.indices.index:
            return pd.DataFrame()

        idx = self.indices.loc[game_name]

        similarity_scores = cosine_similarity(
            self.tfidf_matrix[idx],
            self.tfidf_matrix,
        ).flatten()

        scores = []

        for i, similarity in enumerate(similarity_scores):

            if i == idx:
                continue

            final_score = (
                similarity * 0.6
                + self.df.iloc[i]["rating_score"] * 0.2
                + self.df.iloc[i]["owners"] * 0.1
                + self.df.iloc[i]["average_playtime"] * 0.1
            )

            scores.append((i, final_score))

        scores = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True,
        )

        top_indices = [
            item[0]
            for item in scores[:top_n]
        ]

        return self.df.iloc[top_indices]