import pandas as pd

from finguard.config import RAW_DIR, RAW_FILES, INTERIM_DIR, INTERIM_FILES
from finguard.data.merging import merge_tables

INTERIM_DIR.mkdir(parents=True, exist_ok=True)

transactions = pd.read_parquet(INTERIM_DIR / INTERIM_FILES["transactions"])
users = pd.read_parquet(INTERIM_DIR / INTERIM_FILES["users"])
cards = pd.read_parquet(INTERIM_DIR / INTERIM_FILES["cards"])
fraud_labels = pd.read_json(RAW_DIR / RAW_FILES["fraud_labels"])

master = merge_tables(transactions, fraud_labels, users, cards)
master.to_parquet(INTERIM_DIR / INTERIM_FILES["master"], index=False)
print("master:", master.shape)
