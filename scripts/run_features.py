import pandas as pd

from finguard.config import (
    INTERIM_DIR,
    INTERIM_FILES,
    PROCESSED_DIR,
    PROCESSED_FILES,
)
from finguard.features.preprocessing import (
    filter_and_drop_columns,
    add_engineered_features,
    add_client_aggregates,
    build_transaction_datetime,
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INTERIM_DIR / INTERIM_FILES["master"])
df = filter_and_drop_columns(df)
df = add_engineered_features(df)
df = add_client_aggregates(df)
df = build_transaction_datetime(df)

df.to_parquet(PROCESSED_DIR / PROCESSED_FILES["classification"], index=False)
print("classification:", df.shape)
