import pandas as pd

from finguard.config import RAW_DIR, RAW_FILES, INTERIM_DIR, INTERIM_FILES
from finguard.data.cleaning import (
    clean_user_data,
    clean_card_data,
    clean_transaction_data,
)

INTERIM_DIR.mkdir(parents=True, exist_ok=True)

users = pd.read_csv(RAW_DIR / RAW_FILES["users"])
users = clean_user_data(users)
users.to_parquet(INTERIM_DIR / INTERIM_FILES["users"], index=False)
print("users:", users.shape)

cards = pd.read_csv(RAW_DIR / RAW_FILES["cards"])
cards = clean_card_data(cards)
cards.to_parquet(INTERIM_DIR / INTERIM_FILES["cards"], index=False)
print("cards:", cards.shape)

transactions = pd.read_csv(RAW_DIR / RAW_FILES["transactions"])
transactions = clean_transaction_data(transactions)
transactions.to_parquet(INTERIM_DIR / INTERIM_FILES["transactions"], index=False)
print("transactions:", transactions.shape)
