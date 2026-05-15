# tabular-experiments

Experiment framework for tabular machine learning.

## Features

- Multiple tabular models
  - CatBoost
  - TabNet

- Multiple augmentation methods
  - Original
  - SMOGN

- Experiment tracking
- Model artifact saving
- Reproducible training configuration

## Run

```bash
python main.py
```

## Experiment Configuration

Configured in:

```python
EXPERIMENTS = [
    {"model": "catboost", "augmentation": "original"},
    {"model": "catboost", "augmentation": "smogn"},
    {"model": "tabnet", "augmentation": "original"},
    {"model": "tabnet", "augmentation": "smogn"},
]
```

## Custom SMOGN Dependency

This project uses a custom fork of SMOGN:

```txt
git+https://github.com/KyungaSong/smogn.git@v0.1.0
```