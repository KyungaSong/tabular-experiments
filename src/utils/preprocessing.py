import pandas as pd


MISSING_CATEGORY_VALUE = "__missing__"
MISSING_CATEGORY_STRINGS = ["nan", "None", "<NA>", "NaT"]


def get_categorical_columns(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()


def clean_categorical_columns(
    X: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    X = X.copy()

    for column in categorical_columns:
        X[column] = X[column].mask(
            X[column].isna(),
            MISSING_CATEGORY_VALUE,
        )
        X[column] = X[column].astype(str)
        X.loc[
            X[column].isin(MISSING_CATEGORY_STRINGS),
            column,
        ] = MISSING_CATEGORY_VALUE

    return X
