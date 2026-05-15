from datetime import datetime
from pathlib import Path

from src.config import RESULT_DIR


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_experiment_dir(
    model_name: str,
    augmentation_name: str,
    timestamp: str | None = None,
) -> Path:
    if timestamp is None:
        timestamp = make_timestamp()

    experiment_dir = RESULT_DIR / model_name / augmentation_name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    return experiment_dir