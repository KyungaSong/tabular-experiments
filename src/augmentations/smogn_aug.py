import pandas as pd
import smogn

from src.config import TARGET_COLUMN, SMOGN_PARAMS


AUGMENTATION_NAME = "smogn"


def apply_smogn(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    train_df = train_df.copy().reset_index(drop=True)

    categorical_columns = train_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:
        train_df[column] = train_df[column].astype("object")

    augmented_train_df = smogn.smoter(
        data=train_df,
        y=TARGET_COLUMN,
        **SMOGN_PARAMS,
    )

    print(f"Train original shape: {train_df.shape}")
    print(f"Train SMOGN shape    : {augmented_train_df.shape}")

    return augmented_train_df


def get_smogn_info() -> dict:
    return {
        "augmentation": AUGMENTATION_NAME,
        "params": SMOGN_PARAMS,
    }