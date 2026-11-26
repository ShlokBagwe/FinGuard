from finguard.config import FEATURES


def finalize_features(df):
    """Drop pruned and identifier columns and encode final binary features.

    Takes a DataFrame with statistics applied and returns the finalized copy.
    """
    df = df.copy()

    drop_cols = (
        FEATURES["drop_correlated"]
        + FEATURES["drop_vif"]
        + FEATURES["drop_low_importance"]
        + ["client_id_x", "date", "time", "transaction_datetime"]
    )
    df = df.drop(columns=drop_cols)

    df["has_chip"] = df["has_chip"].astype(int)
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    df["is_online"] = (df["merchant_city"] == "Online").astype("int8")

    # Drop raw location text after extracting is_online and location statistics.
    df = df.drop(columns=["merchant_city", "merchant_state"])
    return df
