import pandas as pd

from finguard.config import PROCESSED_DIR, PROCESSED_FILES
from finguard.features.velocity import add_velocity_features

df = pd.read_parquet(PROCESSED_DIR / PROCESSED_FILES["classification"])
df = add_velocity_features(df)

df.to_parquet(PROCESSED_DIR / PROCESSED_FILES["velocity"], index=False)
print("velocity:", df.shape)
