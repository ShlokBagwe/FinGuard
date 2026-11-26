import pandas as pd

from finguard.config import PROCESSED_DIR, PROCESSED_FILES, REPORTS_DIR
from finguard.models.data_prep import prepare_features_target
from finguard.monitoring.drift import compute_drift_table

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_parquet(PROCESSED_DIR / PROCESSED_FILES["train"])
test_df = pd.read_parquet(PROCESSED_DIR / PROCESSED_FILES["test"])

X_train, _ = prepare_features_target(train_df)
X_test, _ = prepare_features_target(test_df)
del train_df, test_df

drift_df = compute_drift_table(X_train, X_test)
print(drift_df.to_string(index=False))
print("\nStatus counts:")
print(drift_df["status"].value_counts())

report_path = REPORTS_DIR / "drift_report.csv"
drift_df.to_csv(report_path, index=False)
print("\nSaved drift report:", report_path)
