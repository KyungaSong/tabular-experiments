# src/models/registry.py

from src.models.catboost_model import run_catboost
from src.models.tabnet_model import run_tabnet


MODEL_RUNNERS = {
    "catboost": run_catboost,
    "tabnet": run_tabnet,
}