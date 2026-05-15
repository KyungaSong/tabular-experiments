import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.config import RANDOM_STATE, TARGET_COLUMN, CATBOOST_PARAMS, CATBOOST_FIT_PARAMS
from src.metrics import evaluate_regression
from src.utils.saving import save_dataframe, save_metrics, save_json, save_model

MODEL_NAME = "catboost"


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
        X[column] = X[column].replace(
            {
                np.nan: "__missing__",
                None: "__missing__",
                pd.NA: "__missing__",
            }
        )

        X[column] = X[column].astype(str)

        X.loc[
            X[column].isin(["nan", "None", "<NA>", "NaT"]),
            column,
        ] = "__missing__"

    return X


def validate_categorical_columns(
    X: pd.DataFrame,
    categorical_columns: list[str],
    name: str,
) -> None:
    print(f"\n[{name}] categorical check")

    for column in categorical_columns:
        non_string_count = X[column].map(
            lambda value: not isinstance(value, str)
        ).sum()

        missing_like_count = X[column].isin(
            ["nan", "None", "<NA>", "NaT"]
        ).sum()

        print(
            f"{column}: dtype={X[column].dtype}, "
            f"non_string_count={non_string_count}, "
            f"missing_like_count={missing_like_count}"
        )


def train_catboost(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    experiment: dict,
):
    X_train = train_df.drop(columns=TARGET_COLUMN).copy()
    y_train = train_df[TARGET_COLUMN]
    
    X_valid = valid_df.drop(columns=TARGET_COLUMN).copy()
    y_valid = valid_df[TARGET_COLUMN]

    feature_columns = X_train.columns.tolist()
    X_valid = X_valid[feature_columns]

    categorical_columns = get_categorical_columns(X_train)

    X_train = clean_categorical_columns(X_train, categorical_columns)
    X_valid = clean_categorical_columns(X_valid, categorical_columns)

    validate_categorical_columns(X_train, categorical_columns, "train")
    validate_categorical_columns(X_valid, categorical_columns, "valid")

    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=categorical_columns,
    )

    valid_pool = Pool(
        data=X_valid,
        label=y_valid,
        cat_features=categorical_columns,
    )

    experiment_dir = experiment["experiment_dir"]
    train_dir = experiment_dir / "catboost_info"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    model_params = {
        **CATBOOST_PARAMS,
        "random_seed": RANDOM_STATE,
        "train_dir": str(train_dir),
    }

    fit_params = {
        **CATBOOST_FIT_PARAMS,
        "eval_set": valid_pool,
    }

    model = CatBoostRegressor(
        **model_params
    )

    model.fit(
        train_pool,
         **fit_params,
    )

    training_info = {
        "model": MODEL_NAME,
        "augmentation": experiment["augmentation"],
        "model_params": model_params,
        "fit_params": CATBOOST_FIT_PARAMS,
        "categorical_columns": categorical_columns,
        "feature_columns": feature_columns,
    }

    return model, training_info


def evaluate_and_save_catboost(
    model: CatBoostRegressor,
    test_df: pd.DataFrame,
    training_info: dict,
    experiment: dict,
) -> None:
    X_test = test_df.drop(columns=TARGET_COLUMN).copy()
    y_test = test_df[TARGET_COLUMN]

    categorical_columns = training_info["categorical_columns"]
    feature_columns = training_info["feature_columns"]

    X_test = X_test[feature_columns]    
    X_test = clean_categorical_columns(
        X=X_test,
        categorical_columns=categorical_columns,
    )

    test_pool = Pool(
        data=X_test,
        cat_features=categorical_columns,
    )

    y_pred = model.predict(test_pool)

    metrics = evaluate_regression(y_test, y_pred)

    print(f"[{MODEL_NAME}] MAE  : {metrics['mae']:.4f}")
    print(f"[{MODEL_NAME}] RMSE : {metrics['rmse']:.4f}")
    print(f"[{MODEL_NAME}] R²   : {metrics['r2']:.4f}")

    result_df = X_test.copy()
    result_df["actual"] = y_test.to_numpy()
    result_df["predicted"] = y_pred
    result_df["residual"] = result_df["actual"] - result_df["predicted"]
    result_df["abs_error"] = result_df["residual"].abs()

    augmentation_name = experiment["augmentation"]
    metrics_df = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "augmentation": augmentation_name,
                **metrics,
            }
        ]
    )

    experiment_dir = experiment["experiment_dir"]
    prediction_path = experiment_dir / "predictions.csv"
    metrics_path = experiment_dir / "metrics.csv"
    model_path = experiment_dir / "model.cbm"
    training_info_path = experiment_dir / "training_info.json"

    save_dataframe(result_df, prediction_path)
    save_metrics(metrics_df, metrics_path)
    save_model(model, model_path)
    save_json(training_info, training_info_path)

    print(result_df.head())
    print(f"Saved predictions to: {prediction_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved training info to: {training_info_path}")

def run_catboost(train_df, valid_df, test_df, experiment):
    model, training_info = train_catboost(
        train_df=train_df,
        valid_df=valid_df,
        experiment=experiment,
    )

    evaluate_and_save_catboost(
        model=model,
        test_df=test_df,
        training_info=training_info,
        experiment=experiment,
    )