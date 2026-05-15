import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.config import RANDOM_STATE, TABNET_FIT_PARAMS, TABNET_PARAMS, TARGET_COLUMN
from src.metrics import evaluate_regression
from src.utils.saving import save_dataframe,save_json, save_metrics, save_model, save_pickle


MODEL_NAME = "tabnet"


def get_categorical_columns(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()


def prepare_tabnet_data(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    X_train = train_df.drop(columns=TARGET_COLUMN).copy()
    y_train = train_df[TARGET_COLUMN]

    X_valid = valid_df.drop(columns=TARGET_COLUMN).copy()
    y_valid = valid_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=TARGET_COLUMN).copy()
    y_test = test_df[TARGET_COLUMN]

    feature_columns = X_train.columns.tolist()
    X_valid = X_valid[feature_columns]
    X_test = X_test[feature_columns]

    X_test_original = X_test.copy()

    categorical_columns = get_categorical_columns(X_train)
    numeric_columns = [
        column for column in feature_columns
        if column not in categorical_columns
    ]

    for column in categorical_columns:
        X_train[column] = X_train[column].where(
            X_train[column].notna(),
            "__missing__",
        ).astype(str)

        X_valid[column] = X_valid[column].where(
            X_valid[column].notna(),
            "__missing__",
        ).astype(str)

        X_test[column] = X_test[column].where(
            X_test[column].notna(),
            "__missing__",
        ).astype(str)

    categorical_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    if categorical_columns:
        X_train[categorical_columns] = categorical_encoder.fit_transform(
            X_train[categorical_columns]
        )
        X_valid[categorical_columns] = categorical_encoder.transform(
            X_valid[categorical_columns]
        )
        X_test[categorical_columns] = categorical_encoder.transform(
            X_test[categorical_columns]
        )

        X_train[categorical_columns] = X_train[categorical_columns] + 1
        X_valid[categorical_columns] = X_valid[categorical_columns] + 1
        X_test[categorical_columns] = X_test[categorical_columns] + 1

    numeric_imputer = SimpleImputer(strategy="median")

    if numeric_columns:
        X_train[numeric_columns] = numeric_imputer.fit_transform(
            X_train[numeric_columns]
        )
        X_valid[numeric_columns] = numeric_imputer.transform(
            X_valid[numeric_columns]
        )
        X_test[numeric_columns] = numeric_imputer.transform(
            X_test[numeric_columns]
        )

    cat_idxs = [
        X_train.columns.get_loc(column)
        for column in categorical_columns
    ]

    cat_dims = []
    for column in categorical_columns:
        max_value = int(
            max(
                X_train[column].max(),
                X_valid[column].max(),
                X_test[column].max(),
            )
        )
        cat_dims.append(max_value + 1)

    X_train_np = X_train.astype("float32").to_numpy()
    X_valid_np = X_valid.astype("float32").to_numpy()
    X_test_np = X_test.astype("float32").to_numpy()

    target_scaler = StandardScaler()

    y_train_np = target_scaler.fit_transform(
        y_train.to_numpy().reshape(-1, 1)
    ).astype("float32")

    y_valid_np = target_scaler.transform(
        y_valid.to_numpy().reshape(-1, 1)
    ).astype("float32")

    y_test_np = y_test.to_numpy(dtype="float32")

    return {
        "X_train": X_train_np,
        "y_train": y_train_np,
        "X_valid": X_valid_np,
        "y_valid": y_valid_np,
        "X_test": X_test_np,
        "y_test": y_test_np,
        "X_test_original": X_test_original,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
        "target_scaler": target_scaler,
        "categorical_encoder": categorical_encoder,
        "numeric_imputer": numeric_imputer,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "feature_columns": feature_columns,
    }


def train_tabnet(
    data: dict,
    experiment: dict,
):
    model_params = {
        **TABNET_PARAMS,
        "cat_idxs": data["cat_idxs"],
        "cat_dims": data["cat_dims"],
        "seed": RANDOM_STATE,
    }

    fit_params = {
        **TABNET_FIT_PARAMS,
        "eval_set": [
            (data["X_valid"], data["y_valid"]),
        ],
    }

    model = TabNetRegressor(**model_params)

    model.fit(
        data["X_train"],
        data["y_train"],
        **fit_params,
    )

    training_info = {
        "model": MODEL_NAME,
        "augmentation": experiment["augmentation"],
        "model_params": model_params,
        "fit_params": TABNET_FIT_PARAMS,
        "categorical_columns": data["categorical_columns"],
        "numeric_columns": data["numeric_columns"],
        "feature_columns": data["feature_columns"],
        "cat_idxs": data["cat_idxs"],
        "cat_dims": data["cat_dims"],
    }

    return model, training_info


def evaluate_and_save_tabnet(
    model: TabNetRegressor,
    data: dict,
    training_info: dict,
    experiment: dict,
) -> None:
    experiment_dir = experiment["experiment_dir"]

    y_pred_scaled = model.predict(data["X_test"]).reshape(-1, 1)
    y_pred = data["target_scaler"].inverse_transform(y_pred_scaled).ravel()

    y_test = data["y_test"]

    metrics = evaluate_regression(y_test, y_pred)

    print(f"[{MODEL_NAME}] MAE  : {metrics['mae']:.4f}")
    print(f"[{MODEL_NAME}] RMSE : {metrics['rmse']:.4f}")
    print(f"[{MODEL_NAME}] R²   : {metrics['r2']:.4f}")

    result_df = data["X_test_original"].copy()
    result_df["actual"] = y_test
    result_df["predicted"] = y_pred
    result_df["residual"] = result_df["actual"] - result_df["predicted"]
    result_df["abs_error"] = result_df["residual"].abs()

    metrics_df = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "augmentation": experiment["augmentation"],
                **metrics,
            }
        ]
    )

    preprocessors = {
        "categorical_encoder": data["categorical_encoder"],
        "numeric_imputer": data["numeric_imputer"],
        "target_scaler": data["target_scaler"],
    }

    prediction_path = experiment_dir / "predictions.csv"
    metrics_path = experiment_dir / "metrics.csv"
    model_path = experiment_dir / "model"
    preprocessors_path = experiment_dir / "preprocessors.pkl"
    training_info_path = experiment_dir / "training_info.json"

    save_dataframe(result_df, prediction_path)
    save_metrics(metrics_df, metrics_path)
    save_model(model, model_path)
    save_pickle(preprocessors, preprocessors_path)
    save_json(training_info, training_info_path)

    print(result_df.head())
    print(f"Saved predictions to: {prediction_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved model to: {model_path}.zip")
    print(f"Saved preprocessors to: {preprocessors_path}")
    print(f"Saved training info to: {training_info_path}")

def run_tabnet(train_df, valid_df, test_df, experiment):
    data = prepare_tabnet_data(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    model, training_info = train_tabnet(
        data=data,
        experiment=experiment,
    )

    evaluate_and_save_tabnet(
        model=model,
        data=data,
        training_info=training_info,
        experiment=experiment,
    )