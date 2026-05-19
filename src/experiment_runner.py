from src.augmentations.registry import AUGMENTATION_RUNNERS
from src.models.registry import MODEL_RUNNERS
from src.utils.path import get_experiment_dir
from src.utils.saving import save_json


EXPERIMENT_CONFIG_FILENAME = "experiment_config.json"


def get_experiment_config_value(
    experiment_config: dict,
    key: str,
):
    try:
        return experiment_config[key]
    except KeyError as error:
        available_keys = ", ".join(sorted(experiment_config))
        raise ValueError(
            f"Missing experiment config key: {key}. "
            f"Available keys: {available_keys}"
        ) from error


def get_runner(
    runners: dict,
    name: str,
    runner_type: str,
):
    try:
        return runners[name]
    except KeyError as error:
        available_names = ", ".join(sorted(runners))
        raise ValueError(
            f"Unknown {runner_type}: {name}. "
            f"Available {runner_type}s: {available_names}"
        ) from error


def build_experiment_metadata(
    experiment_config: dict,
    experiment: dict,
) -> dict:
    return {
        "model": experiment["model"],
        "augmentation": experiment["augmentation"],
        "run_id": experiment["experiment_dir"].name,
        "experiment_dir": str(experiment["experiment_dir"]),
        "experiment_config": dict(experiment_config),
    }


def run_experiment(train_df, valid_df, test_df, experiment_config):
    model_name = get_experiment_config_value(experiment_config, "model")
    augmentation_name = get_experiment_config_value(
        experiment_config,
        "augmentation",
    )
    model_runner = get_runner(MODEL_RUNNERS, model_name, "model")
    augmentation_runner = get_runner(
        AUGMENTATION_RUNNERS,
        augmentation_name,
        "augmentation",
    )

    experiment_dir = get_experiment_dir(
        model_name=model_name,
        augmentation_name=augmentation_name,
    )

    experiment = {
        "model": model_name,
        "augmentation": augmentation_name,
        "experiment_dir": experiment_dir,
    }

    print(f"\nRunning experiment: {model_name} + {augmentation_name}")
    print(f"Experiment directory: {experiment_dir}")

    experiment_config_path = experiment_dir / EXPERIMENT_CONFIG_FILENAME
    experiment_metadata = build_experiment_metadata(
        experiment_config=experiment_config,
        experiment=experiment,
    )
    save_json(experiment_metadata, experiment_config_path)
    print(f"Saved experiment config to: {experiment_config_path}")

    train_input_df = augmentation_runner(train_df)
    model_runner(train_input_df, valid_df, test_df, experiment)
