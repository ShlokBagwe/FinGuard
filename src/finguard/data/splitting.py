from finguard.config import TRAIN_RATIO, TIME_COLUMN


def split_chronologically(df, train_ratio=TRAIN_RATIO, time_column=TIME_COLUMN):
    """Sort by timestamp and split chronologically into train and test tables.

    Takes a DataFrame and returns (train_df, test_df).
    """
    df = df.sort_values(time_column).reset_index(drop=True)

    split_point = int(len(df) * train_ratio)
    split_date = df.iloc[split_point][time_column]

    # Cut on split_date so transactions sharing the boundary timestamp stay together.
    train_df = df[df[time_column] < split_date].copy()
    test_df = df[df[time_column] >= split_date].copy()

    return train_df, test_df
