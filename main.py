from src.data import prepare_base_data
from src.experiment_runner import run_experiment


EXPERIMENTS = [
    {"model": "catboost", "augmentation": "original"},
    {"model": "catboost", "augmentation": "smogn"},
    {"model": "tabnet", "augmentation": "original"},
    {"model": "tabnet", "augmentation": "smogn"},
]


def main():
    train_df, valid_df, test_df = prepare_base_data()

    for experiment_config in EXPERIMENTS:
        run_experiment(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            experiment_config=experiment_config,
        )


if __name__ == "__main__":
    main()