import pandas as pd
import smogn

from src.config import TARGET_COLUMN, SMOGN_PARAMS


AUGMENTATION_NAME = "smogn"


def apply_smogn(
    train_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    smogn_params: dict | None = None,
) -> pd.DataFrame:
    if target_column not in train_df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    train_df = train_df.copy().reset_index(drop=True)
    smogn_params = dict(SMOGN_PARAMS if smogn_params is None else smogn_params)

    categorical_columns = train_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    for column in categorical_columns:
        train_df[column] = train_df[column].astype("object")

    augmented_train_df = smogn.smoter(
        data=train_df,
        y=target_column,
        **smogn_params,
    )

    print(f"Train original shape: {train_df.shape}")
    print(f"Train SMOGN shape    : {augmented_train_df.shape}")

    return augmented_train_df


def get_smogn_info() -> dict:
    return {
        "augmentation": AUGMENTATION_NAME,
        "params": dict(SMOGN_PARAMS),
    }
