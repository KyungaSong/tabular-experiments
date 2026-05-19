import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def print_regression_metrics(
    model_name: str,
    metrics: dict,
) -> None:
    print(f"[{model_name}] MAE  : {metrics['mae']:.4f}")
    print(f"[{model_name}] RMSE : {metrics['rmse']:.4f}")
    print(f"[{model_name}] R2   : {metrics['r2']:.4f}")


def build_prediction_result_df(
    X: pd.DataFrame,
    y_true,
    y_pred,
) -> pd.DataFrame:
    result_df = X.copy()
    result_df["actual"] = y_true
    result_df["predicted"] = y_pred
    result_df["residual"] = result_df["actual"] - result_df["predicted"]
    result_df["abs_error"] = result_df["residual"].abs()

    return result_df
