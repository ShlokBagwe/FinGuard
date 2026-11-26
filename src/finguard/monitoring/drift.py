import numpy as np
import pandas as pd

from finguard.config import MODEL


def calculate_psi(train, test, bins=10):
    """Calculate Population Stability Index for a numerical feature.

    Takes train and test Series and bin count; returns the PSI scalar.
    """
    bin_edges = np.percentile(train, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)

    train_pct = np.histogram(train, bins=bin_edges)[0] / len(train)
    test_pct = np.histogram(test, bins=bin_edges)[0] / len(test)

    # Replace zero bin proportions so log ratios remain finite.
    train_pct = np.where(train_pct == 0, 0.0001, train_pct)
    test_pct = np.where(test_pct == 0, 0.0001, test_pct)

    psi = np.sum((train_pct - test_pct) * np.log(train_pct / test_pct))
    return psi


def psi_categorical(train_col, test_col):
    """Calculate Population Stability Index for a categorical feature.

    Takes train and test Series and returns the summed PSI scalar.
    """
    train_pct = train_col.value_counts(normalize=True)
    test_pct = test_col.value_counts(normalize=True)

    psi_table = pd.DataFrame({"train_pct": train_pct, "test_pct": test_pct}).fillna(0.0001)
    psi_table["psi"] = (
        (psi_table["test_pct"] - psi_table["train_pct"])
        * np.log(psi_table["test_pct"] / psi_table["train_pct"])
    )
    return psi_table["psi"].sum()


def compute_drift_table(X_train, X_test):
    """Compute PSI and drift status across all 23 model input features.

    Takes prepared X_train and X_test DataFrames and returns a sorted summary DataFrame.
    """
    rows = []

    for col in MODEL["numerical_features"]:
        psi_val = float(calculate_psi(X_train[col], X_test[col]))
        rows.append({"feature": col, "type": "numerical", "psi": psi_val})

    for col in MODEL["categorical_features"]:
        psi_val = float(psi_categorical(X_train[col], X_test[col]))
        rows.append({"feature": col, "type": "categorical", "psi": psi_val})

    drift_df = pd.DataFrame(rows)

    statuses = []
    for psi_val in drift_df["psi"]:
        if psi_val < 0.1:
            statuses.append("stable")
        elif psi_val < 0.25:
            statuses.append("moderate")
        else:
            statuses.append("significant")

    drift_df["status"] = statuses
    drift_df = drift_df.sort_values("psi", ascending=False).reset_index(drop=True)
    return drift_df
