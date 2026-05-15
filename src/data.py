import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    HOUSING_URL,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    VALID_SIZE,
)


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(HOUSING_URL)
    print(f"Raw dataset shape: {df.shape}")
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    valid_df = X_valid.copy()
    valid_df[TARGET_COLUMN] = y_valid

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    print(f"Train shape: {train_df.shape}")
    print(f"Valid shape: {valid_df.shape}")
    print(f"Test shape : {test_df.shape}")

    return train_df, valid_df, test_df


def prepare_base_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_raw_data()
    train_df, valid_df, test_df = split_data(df)

    return train_df, valid_df, test_df