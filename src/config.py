from pathlib import Path

SAVE_USED_DATA = False

# directory paths
BASE_DIR = Path(__file__).resolve().parents[1]


# data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HOUSING_URL = "https://raw.githubusercontent.com/nickkunz/smogn/master/data/housing.csv"


# output paths
FIG_DIR = BASE_DIR / "fig"
RESULT_DIR = BASE_DIR / "result"


# data augmentation parameters
SMOGN_PARAMS = {
    "k": 9,
    "samp_method": "extreme",
    "rel_thres": 0.80,
    "rel_method": "auto",
    "rel_xtrm_type": "high",
    "rel_coef": 2.25,
}

# model parameters

## general parameters
TARGET_COLUMN = "SalePrice"
RANDOM_STATE = 42

TEST_SIZE = 0.2
VALID_SIZE = 0.2

## CatBoost parameters
CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "verbose": 100,
}

CATBOOST_FIT_PARAMS = {
    "use_best_model": True,
}

## TabNet parameters
TABNET_PARAMS = {
    "n_d": 8,
    "n_a": 8,
    "n_steps": 3,
    "optimizer_params": {"lr": 1e-3},
    "verbose": 1,
}

TABNET_FIT_PARAMS = {
    "max_epochs": 500,
    "patience": 50,
    "batch_size": 64,
    "virtual_batch_size": 32,
}