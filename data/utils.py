import pandas as pd

from itertools import accumulate, chain, pairwise
from typing import Iterator, List, Tuple


from sklearn.model_selection import train_test_split



def load_data() -> pd.DataFrame:
    """
    loads the data from the data folder
    """

    df = pd.read_csv("MeltingPointPredictionModels/data/train.csv")

    return df


def load_test_data() -> pd.DataFrame:
    """
    loads the test data from the data folder
    """

    df = pd.read_csv("MeltingPointPredictionModels/data/test.csv")

    return df


def clean_for_xgboost(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    drop ordinal columns, clean df
    """

    data_df = data_df.drop(columns=["SMILES"])

    return data_df


def split_data(data_df: pd.DataFrame, rng: int = 0) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits data into training and validation sets using scikit-learn.
    
    Args:
        data_df (pd.DataFrame): The input dataframe containing features and the target.
        rng (int): The random state for reproducibility.

    Returns:
        A tuple containing: x_train, y_train, x_val, y_val
    """
    # Separate features (X) from the target variable (y) first
    x = data_df.drop(columns=['Tm'])
    y = data_df['Tm']

    # Use train_test_split to get all four sets at once
    # It handles shuffling by default
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, 
        test_size=0.2,
        random_state=rng
    )

    return x_train, y_train, x_val, y_val