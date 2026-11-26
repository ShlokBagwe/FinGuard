import pandas as pd
import polars as pl


def add_velocity_features(df):
    """Add 24-hour rolling transaction count and spend sum per client.

    Takes the preprocessed features DataFrame and returns a copy with
    txn_count_last_24h and amount_sum_last_24h merged in.
    """
    df = df.copy()

    pl_df = pl.from_pandas(df[["client_id_x", "transaction_datetime", "abs_amount"]])
    pl_df = pl_df.sort(["client_id_x", "transaction_datetime"])

    # Exclude the current transaction from its own 24-hour lookback totals.
    pl_df = pl_df.rolling(
        index_column="transaction_datetime",
        period="24h",
        group_by="client_id_x",
    ).agg([
        (pl.len() - 1).alias("txn_count_last_24h"),
        (pl.col("abs_amount").sum() - pl.col("abs_amount").last()).alias("amount_sum_last_24h"),
    ])

    # About 1.3% of rows share (client_id_x, transaction_datetime); deduplicating
    # prevents a many-to-many merge from inflating the row count.
    pl_df = pl_df.unique(subset=["client_id_x", "transaction_datetime"], keep="first")

    df = df.merge(
        pl_df.to_pandas(),
        on=["client_id_x", "transaction_datetime"],
        how="left",
    )
    return df
