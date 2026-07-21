"""
Preprocessing Module
--------------------
This module is responsible for cleaning text,
creating combined features, and preparing the
dataset for recommendation.
"""

import pandas as pd


def clean_text(text: str) -> str:
    """
    Clean text by converting to lowercase
    and replacing semicolons with spaces.
    """

    return str(text).lower().replace(";", " ")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text columns and create
    the combined feature column.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    text_columns = [
        "genres",
        "categories",
        "steamspy_tags",
        "platforms",
    ]

    for column in text_columns:
        df[column] = df[column].apply(clean_text)

    df["combined"] = (
        df["genres"]
        + " "
        + df["categories"]
        + " "
        + df["steamspy_tags"]
        + " "
        + df["platforms"]
    )

    return df