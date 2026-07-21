"""
Data Loader Module
------------------
This module is responsible for loading the raw dataset
and performing basic preprocessing before it is passed
to the recommendation engine.
"""

import pandas as pd

from src.config import RAW_DATA_PATH


def load_data():
    """
    Load the Steam dataset and perform initial preprocessing.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe ready for feature engineering.
    """

    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    # Convert release date
    df["release_date"] = pd.to_datetime(
        df["release_date"],
        errors="coerce"
    )

    # Extract release year
    df["year"] = df["release_date"].dt.year

    # Convert owners into numeric values
    df["owners"] = df["owners"].astype(str).str.split("-").str[0]
    df["owners"] = pd.to_numeric(
        df["owners"],
        errors="coerce"
    )

    # Calculate rating score
    df["rating_score"] = (
        df["positive_ratings"] /
        (
            df["positive_ratings"] +
            df["negative_ratings"]
        )
    )

    # Replace missing values
    df.fillna("", inplace=True)

    return df