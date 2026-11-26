import numpy as np


def fit_statistics(train_df):
    """Learn customer, merchant-group, global fill, and rare-error statistics on train only.

    Takes the training DataFrame and returns a dictionary of learned statistics.
    """
    df = train_df.copy()

    # Fill missing merchant_state before grouping so online transactions form one group.
    df["merchant_state"] = df["merchant_state"].fillna("Online")

    client_stats = df.groupby("client_id_x")["abs_amount"].agg(
        customer_avg_amount="mean",
        customer_std_amount="std",
        customer_txn_count="count",
    )

    global_avg = df["abs_amount"].mean()
    global_std = df["abs_amount"].std()

    mcc_stats = df.groupby("mcc").agg(
        mcc_fraud_rate=("target_numeric", "mean"),
        mcc_avg_amount=("abs_amount", "mean"),
        mcc_txn_count=("mcc", "size"),
    )

    city_stats = df.groupby("merchant_city").agg(
        city_fraud_rate=("target_numeric", "mean"),
        city_avg_amount=("abs_amount", "mean"),
        city_txn_count=("merchant_city", "size"),
        city_unique_customers=("client_id_x", "nunique"),
    )

    state_stats = df.groupby("merchant_state").agg(
        state_fraud_rate=("target_numeric", "mean"),
        state_avg_amount=("abs_amount", "mean"),
        state_txn_count=("merchant_state", "size"),
        state_unique_customers=("client_id_x", "nunique"),
    )

    client_stats_fraud = df.groupby("client_id_x").agg(
        client_fraud_rate=("target_numeric", "mean"),
    )

    global_fraud_rate = df["target_numeric"].mean()
    global_amount = df["abs_amount"].mean()

    error_counts = df["errors"].fillna("No Error").value_counts()
    rare_errors = error_counts[error_counts < 100].index.tolist()

    return {
        "client_stats": client_stats,
        "global_avg": global_avg,
        "global_std": global_std,
        "mcc_stats": mcc_stats,
        "city_stats": city_stats,
        "state_stats": state_stats,
        "client_stats_fraud": client_stats_fraud,
        "global_fraud_rate": global_fraud_rate,
        "global_amount": global_amount,
        "rare_errors": rare_errors,
    }


def apply_statistics(df, stats):
    """Apply train-learned statistics and fallback fills to a train or test DataFrame.

    Takes a DataFrame and the statistics dictionary from fit_statistics, and returns a copy.
    """
    df = df.copy()

    df = df.merge(stats["client_stats"], on="client_id_x", how="left")

    df["customer_avg_amount"] = df["customer_avg_amount"].fillna(stats["global_avg"])
    df["customer_std_amount"] = (
        df["customer_std_amount"].fillna(stats["global_std"]).replace(0, stats["global_std"])
    )
    df["customer_txn_count"] = df["customer_txn_count"].fillna(0)
    df["amount_vs_avg"] = df["abs_amount"] / df["customer_avg_amount"]
    df["amount_zscore"] = (
        ((df["abs_amount"] - df["customer_avg_amount"]) / df["customer_std_amount"])
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    df["merchant_state"] = df["merchant_state"].fillna("Online")

    df = df.merge(stats["mcc_stats"], on="mcc", how="left")
    df = df.merge(stats["city_stats"], on="merchant_city", how="left")
    df = df.merge(stats["state_stats"], on="merchant_state", how="left")
    df = df.merge(stats["client_stats_fraud"], on="client_id_x", how="left")

    fraud_rate_cols = ["mcc_fraud_rate", "city_fraud_rate", "state_fraud_rate", "client_fraud_rate"]
    amount_cols = ["mcc_avg_amount", "city_avg_amount", "state_avg_amount"]
    count_cols = [
        "mcc_txn_count",
        "city_txn_count",
        "state_txn_count",
        "city_unique_customers",
        "state_unique_customers",
    ]

    for col in fraud_rate_cols:
        df[col] = df[col].fillna(stats["global_fraud_rate"])
    for col in amount_cols:
        df[col] = df[col].fillna(stats["global_amount"])
    for col in count_cols:
        df[col] = df[col].fillna(0)

    df["errors"] = df["errors"].fillna("No Error")
    df["errors"] = df["errors"].replace(stats["rare_errors"], "Other Error")

    return df
