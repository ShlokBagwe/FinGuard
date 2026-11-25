import pandas as pd


def clean_user_data(df):
    """Clean the raw users table. Returns a cleaned copy."""
    df = df.copy()

    # Income and debt are stored as text with a "$" prefix.
    money_columns = ["per_capita_income", "yearly_income", "total_debt"]
    for col in money_columns:
        df[col] = df[col].str.replace("$", "", regex=False).str.strip().astype(float)

    return df


def clean_card_data(df):
    """Clean the raw cards table. Returns a cleaned copy."""
    df = df.copy()

    # Downstream feature engineering expects real datetimes and an integer year;
    # Period columns break pd.to_datetime and arithmetic.
    df["expires"] = pd.to_datetime(df["expires"], format="%m/%Y")

    df["has_chip"] = df["has_chip"].map({"YES": True, "NO": False})

    df["credit_limit"] = df["credit_limit"].str.replace("$", "", regex=False).str.strip().astype(float)

    df["acct_open_date"] = pd.to_datetime(df["acct_open_date"], format="%m/%Y")

    df["year_pin_last_changed"] = pd.to_datetime(df["year_pin_last_changed"], format="%Y").dt.year

    # Single constant value across all cards, so it carries no signal.
    df = df.drop(columns="card_on_dark_web")

    return df


def clean_transaction_data(df):
    """Clean the raw transactions table. Returns a cleaned copy."""
    df = df.copy()

    # Split the timestamp into separate date and time columns, with time placed after date.
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df.insert(2, "time", df["date"].dt.time)
    df["date"] = df["date"].dt.date

    df["amount"] = df["amount"].str.replace("$", "", regex=False).str.strip().astype(float)

    # Title case turns "ONLINE" into "Online", which later steps filter on.
    df["merchant_city"] = df["merchant_city"].str.strip().str.title()

    return df
