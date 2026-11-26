from finguard.config import MODEL


def prepare_features_target(df):
    """Regroup rare errors, drop low-impact columns, and split features from target.

    Takes a train or test DataFrame and returns (X, y).
    """
    X = df.drop("target_numeric", axis=1)
    y = df["target_numeric"]

    keep_errors = MODEL["keep_errors"]
    X["errors"] = X["errors"].where(X["errors"].isin(keep_errors), "Other Error")

    drop_low_impact = MODEL["drop_low_impact"]
    X = X.drop(columns=[c for c in drop_low_impact if c in X.columns])

    return X, y
