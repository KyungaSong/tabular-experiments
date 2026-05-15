import pandas as pd


AUGMENTATION_NAME = "original"


def apply_no_augmentation(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    return train_df.copy()