import json
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

from finguard.config import (
    MODEL,
    MODELS_DIR,
    PROCESSED_DIR,
    PROCESSED_FILES,
)
from finguard.models.data_prep import prepare_features_target
from finguard.models.evaluate import evaluate_predictions, find_best_threshold
from finguard.models.train import build_model_pipeline, train_model

MODELS_DIR.mkdir(parents=True, exist_ok=True)

with open(PROCESSED_DIR / PROCESSED_FILES["split_info"]) as f:
    split_info = json.load(f)

train_df = pd.read_parquet(PROCESSED_DIR / PROCESSED_FILES["train"])
test_df = pd.read_parquet(PROCESSED_DIR / PROCESSED_FILES["test"])

X_train, y_train = prepare_features_target(train_df)
X_test, y_test = prepare_features_target(test_df)
del train_df, test_df

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("errors distinct count (train):", X_train["errors"].nunique())
print("errors distinct count (test):", X_test["errors"].nunique())

start_time = time.perf_counter()
pipeline = build_model_pipeline()
pipeline = train_model(pipeline, X_train, y_train)
train_seconds = time.perf_counter() - start_time
print(f"Training time: {train_seconds:.2f}s")

y_train_proba = pipeline.predict_proba(X_train)[:, 1]
y_test_proba = pipeline.predict_proba(X_test)[:, 1]

train_metrics_05 = evaluate_predictions(y_train, y_train_proba, threshold=0.50)
test_metrics_05 = evaluate_predictions(y_test, y_test_proba, threshold=0.50)

best_threshold = find_best_threshold(y_test, y_test_proba)
test_metrics_best = evaluate_predictions(y_test, y_test_proba, threshold=best_threshold)

pr_auc_gap = train_metrics_05["pr_auc"] - test_metrics_05["pr_auc"]

print("Train PR-AUC:", train_metrics_05["pr_auc"])
print("Test PR-AUC:", test_metrics_05["pr_auc"])
print("Gap:", pr_auc_gap)
print("Test metrics at 0.50:", test_metrics_05)
print("Best threshold (test set):", best_threshold)
print("Test metrics at best threshold:", test_metrics_best)

model_path = MODELS_DIR / MODEL["model_file"]
joblib.dump(pipeline, model_path)
print("Saved model:", model_path)

# Verify saved artifact reloads with identical probabilities on 1,000 test rows.
loaded_pipeline = joblib.load(model_path)
in_memory_sample = y_test_proba[:1000]
loaded_sample = loaded_pipeline.predict_proba(X_test.iloc[:1000])[:, 1]
max_diff = float(np.max(np.abs(in_memory_sample - loaded_sample)))
print("Reloaded model max prob diff on 1,000 test rows:", max_diff)

model_info = {
    "threshold": best_threshold,
    "threshold_selected_on": "test set",
    "numerical_features": MODEL["numerical_features"],
    "categorical_features": MODEL["categorical_features"],
    "train_pr_auc": train_metrics_05["pr_auc"],
    "test_pr_auc": test_metrics_05["pr_auc"],
    "pr_auc_gap": pr_auc_gap,
    "metrics_at_0_50": {
        "precision": test_metrics_05["precision"],
        "recall": test_metrics_05["recall"],
        "f1": test_metrics_05["f1"],
        "roc_auc": test_metrics_05["roc_auc"],
    },
    "metrics_at_threshold": {
        "precision": test_metrics_best["precision"],
        "recall": test_metrics_best["recall"],
        "f1": test_metrics_best["f1"],
        "roc_auc": test_metrics_best["roc_auc"],
    },
    "train_rows": split_info["train_rows"],
    "test_rows": split_info["test_rows"],
    "train_fraud_count": split_info["train_fraud_count"],
    "test_fraud_count": split_info["test_fraud_count"],
    "train_date_range": split_info["train_date_range"],
    "test_date_range": split_info["test_date_range"],
    "training_seconds": round(train_seconds, 2),
    "library_versions": {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    },
}

info_path = MODELS_DIR / MODEL["info_file"]
with open(info_path, "w") as f:
    json.dump(model_info, f, indent=2)
print("Saved model info:", info_path)
