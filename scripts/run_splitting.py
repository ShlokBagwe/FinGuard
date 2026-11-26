import json

import pandas as pd

from finguard.config import PROCESSED_DIR, PROCESSED_FILES, TIME_COLUMN
from finguard.data.splitting import split_chronologically
from finguard.features.statistics import fit_statistics, apply_statistics
from finguard.features.finalize import finalize_features

df = pd.read_parquet(PROCESSED_DIR / PROCESSED_FILES["velocity"])

train_df, test_df = split_chronologically(df)
print("Train split:", train_df.shape)
print("Test split:", test_df.shape)
print("Train fraud rate:", train_df["target_numeric"].mean() * 100, "%")
print("Test fraud rate:", test_df["target_numeric"].mean() * 100, "%")
print("Train date range:", train_df[TIME_COLUMN].min(), "to", train_df[TIME_COLUMN].max())
print("Test date range:", test_df[TIME_COLUMN].min(), "to", test_df[TIME_COLUMN].max())

# Record split metadata before finalize_features drops the timestamp column.
split_info = {
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "train_fraud_count": int(train_df["target_numeric"].sum()),
    "test_fraud_count": int(test_df["target_numeric"].sum()),
    "train_date_range": [
        str(train_df[TIME_COLUMN].min()),
        str(train_df[TIME_COLUMN].max()),
    ],
    "test_date_range": [
        str(test_df[TIME_COLUMN].min()),
        str(test_df[TIME_COLUMN].max()),
    ],
}
with open(PROCESSED_DIR / PROCESSED_FILES["split_info"], "w") as f:
    json.dump(split_info, f, indent=2)

stats = fit_statistics(train_df)
train_df = apply_statistics(train_df, stats)
test_df = apply_statistics(test_df, stats)

train_df = finalize_features(train_df)
test_df = finalize_features(test_df)

train_df.to_parquet(PROCESSED_DIR / PROCESSED_FILES["train"], index=False)
test_df.to_parquet(PROCESSED_DIR / PROCESSED_FILES["test"], index=False)
print("Final train:", train_df.shape)
print("Final test:", test_df.shape)
