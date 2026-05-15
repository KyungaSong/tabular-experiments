import json
from pathlib import Path

import joblib
import pandas as pd


def save_dataframe(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_metrics(
    metrics: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(path, index=False)


def save_json(
    data,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def save_pickle(
    data,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, path)


def save_model(
    model,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))