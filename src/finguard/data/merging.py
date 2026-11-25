import pandas as pd


def downcast_dtypes(df, money_cols=None, skip_cols=None):
    """Downcast numeric columns to smaller dtypes to reduce memory usage."""
    money_cols = money_cols or []
    skip_cols = skip_cols or []

    for col in df.columns:
        if col in skip_cols or df[col].dtype.name in ("category",) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if col in money_cols:
            df[col] = df[col].astype("float32")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def merge_tables(transactions, fraud_labels, users, cards):
    """Merge transactions, fraud labels, users and cards into the master table.

    Takes cleaned transactions, users, cards and raw fraud labels DataFrames.
    Returns the combined master DataFrame.
    """
    trans_df = transactions.copy()
    trans_df["abs_amount"] = abs(trans_df["amount"])

    # Fraud labels JSON loads with transaction IDs as the DataFrame index.
    labels_df = fraud_labels.reset_index().rename({"index": "id"}, axis=1)

    master_df = pd.merge(trans_df, labels_df, on="id", how="left")
    master_df = downcast_dtypes(master_df, money_cols=["amount", "abs_amount"])

    master_df["date"] = pd.to_datetime(master_df["date"])
    master_df["target_numeric"] = master_df["target"].map({"No": 0, "Yes": 1})

    users_df = downcast_dtypes(
        users.copy(),
        money_cols=["per_capita_income", "yearly_income", "total_debt"],
    )
    master_df = master_df.merge(users_df, how="left", left_on="client_id", right_on="id")
    master_df = master_df.drop(columns=["id_y"])
    master_df = master_df.rename(columns={"id_x": "id"})

    master_df = master_df.merge(cards, how="left", left_on="card_id", right_on="id")
    # Drop duplicate join keys while preserving client_id_x expected downstream.
    master_df = master_df.drop(columns=["id_y", "id_x", "client_id_y"])
    master_df = downcast_dtypes(master_df, money_cols=["card_number", "credit_limmit"])

    return master_df
