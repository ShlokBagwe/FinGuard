import pandas as pd

from finguard.config import PROCESSED_DIR, PROCESSED_FILES
from finguard.data.splitting import split_chronologically
from finguard.features.statistics import fit_statistics, apply_statistics
from finguard.features.finalize import finalize_features

df = pd.read_parquet(PROCESSED_DIR / PROCESSED_FILES["velocity"])

train_df, test_df = split_chronologically(df)
print("Train split:", train_df.shape)
print("Test split:", test_df.shape)
print("Train fraud rate:", train_df["target_numeric"].mean() * 100, "%")
print("Test fraud rate:", test_df["target_numeric"].mean() * 100, "%")
print("Train date range:", train_df["transaction_datetime"].min(), "to", train_df["transaction_datetime"].max())
print("Test date range:", test_df["transaction_datetime"].min(), "to", test_df["transaction_datetime"].max())

stats = fit_statistics(train_df)
train_df = apply_statistics(train_df, stats)
test_df = apply_statistics(test_df, stats)

train_df = finalize_features(train_df)
test_df = finalize_features(test_df)

train_df.to_parquet(PROCESSED_DIR / PROCESSED_FILES["train"], index=False)
test_df.to_parquet(PROCESSED_DIR / PROCESSED_FILES["test"], index=False)
print("Final train:", train_df.shape)
print("Final test:", test_df.shape)
