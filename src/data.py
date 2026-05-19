import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    HOUSING_URL,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    VALID_SIZE,
)


def load_raw_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    print(f"Raw dataset shape: {df.shape}")
    return df


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    valid_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=target_column)
    y = df[target_column]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=valid_size,
        random_state=random_state,
    )

    train_df = X_train.copy()
    train_df[target_column] = y_train

    valid_df = X_valid.copy()
    valid_df[target_column] = y_valid

    test_df = X_test.copy()
    test_df[target_column] = y_test

    print(f"Train shape: {train_df.shape}")
    print(f"Valid shape: {valid_df.shape}")
    print(f"Test shape : {test_df.shape}")

    return train_df, valid_df, test_df


def prepare_base_data(
    url: str = HOUSING_URL,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    valid_size: float = VALID_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_raw_data(url=url)
    train_df, valid_df, test_df = split_data(
        df,
        target_column=target_column,
        test_size=test_size,
        valid_size=valid_size,
        random_state=random_state,
    )

    return train_df, valid_df, test_df
