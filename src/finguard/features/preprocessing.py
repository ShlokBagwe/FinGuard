import pandas as pd


def filter_and_drop_columns(df):
    """Drop unlabeled rows and unused columns from the merged master table.

    Takes the master DataFrame and returns a filtered copy.
    """
    df = df.dropna(subset=["target"]).copy()

    # Preserve column drop list from the notebook; EDA-only bins/groups are absent in master.
    drop_columns = [
        "card_id",
        "merchant_id",
        "card_number",
        "address",
        "target",
        "amount_bins",
        "birth_year",
        "birth_month",
        "age_group",
        "income_group",
        "debt_group",
        "expires",
        "cvv",
        "zip",
        "retirement_age",
    ]
    missing_columns = [col for col in drop_columns if col not in df.columns]
    present_columns = [col for col in drop_columns if col in df.columns]
    if missing_columns:
        print("Absent drop columns skipped:", missing_columns)

    df = df.drop(columns=present_columns)
    return df


def add_engineered_features(df):
    """Create financial ratios, calendar features, refund flag, and card-age features.

    Takes the filtered DataFrame and returns a transformed copy.
    """
    df = df.copy()

    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")

    df["debt_to_income"] = df["total_debt"] / df["yearly_income"]
    df["available_credit_per_card"] = df["credit_limit"] / df["num_credit_cards"]

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["weekend_flag"] = (df["day_of_week"] >= 5).astype(int)

    # Signed amount is replaced by the refund indicator alongside abs_amount.
    df["is_refund"] = (df["amount"] < 0).astype(int)
    df = df.drop("amount", axis=1)

    df["acct_open_date"] = pd.to_datetime(df["acct_open_date"])
    df["account_age_years"] = df["date"].dt.year - df["acct_open_date"].dt.year
    df["years_since_pin_change"] = df["date"].dt.year - df["year_pin_last_changed"]
    df["credit_limit_to_income"] = df["credit_limit"] / df["yearly_income"]

    df = df.drop(columns=["latitude", "longitude", "acct_open_date", "year_pin_last_changed"])
    return df


def build_transaction_datetime(df):
    """Combine date and time into transaction_datetime and sort by client and timestamp.

    Takes the feature DataFrame and returns the sorted copy.
    """
    df = df.copy()

    time_strings = df["time"].dt.strftime("%H:%M:%S")
    df["transaction_datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + time_strings,
        format="%Y-%m-%d %H:%M:%S",
    )
    # Restore time to Python time objects after extracting transaction_datetime.
    df["time"] = df["time"].dt.time

    df = df.sort_values(["client_id_x", "transaction_datetime"])
    return df
