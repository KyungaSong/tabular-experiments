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

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

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

## Results

Each experiment writes artifacts under:

```txt
result/{model}/{augmentation}/{run_id}/
```

Common outputs:

```txt
experiment_config.json
training_info.json
metrics.csv
predictions.csv
model artifact
```

`experiment_config.json` stores the runner-level experiment snapshot, including
the model, augmentation, run id, output directory, and original experiment
configuration.

## Adding Experiments

Add a model runner in:

```txt
src/models/registry.py
```

Add an augmentation runner in:

```txt
src/augmentations/registry.py
```

Shared preprocessing utilities live in:

```txt
src/utils/preprocessing.py
```

## Custom SMOGN Dependency

This project uses a custom fork of SMOGN:

```txt
git+https://github.com/KyungaSong/smogn.git@v0.1.0
```
