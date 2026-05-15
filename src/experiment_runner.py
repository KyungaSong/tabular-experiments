from src.augmentations.registry import AUGMENTATION_RUNNERS
from src.models.registry import MODEL_RUNNERS
from src.utils.path import get_experiment_dir


def run_experiment(train_df, valid_df, test_df, experiment_config):
    model_name = experiment_config["model"]
    augmentation_name = experiment_config["augmentation"]

    experiment_dir = get_experiment_dir(
        model_name=model_name,
        augmentation_name=augmentation_name,
    )

    experiment = {
        "model": model_name,
        "augmentation": augmentation_name,
        "experiment_dir": experiment_dir,
    }

    train_input_df = AUGMENTATION_RUNNERS[augmentation_name](train_df)
    MODEL_RUNNERS[model_name](train_input_df, valid_df, test_df, experiment)