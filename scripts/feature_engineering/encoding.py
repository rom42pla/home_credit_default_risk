import pandas as pd
from Timer import Timer


def frequency_encoding(df, just_one_hot=False, log=False):
    """
    encodes the dataframe with the frequencies on the categorical features.
    :param df: dataframe to be encoded
    :param log: true if we want to set the timer on
    :return: encoded dataframe
    """
    if log: section_timer = Timer(log=f"frequency encoding a dataframe with shape {df.shape}")

    # does a one-hot encoding
    old_cols = set(df.columns)
    df = pd.get_dummies(df)

    # eventually do the frequency encoding
    if not just_one_hot:
        for col in list(set(df.columns) - old_cols):
            df[col] = (df[col] * (df[col].sum() / df.shape[0]))

    if log: section_timer.end_timer(log=f"done")
    return df

def get_columns_types(df):
    columns = list(df.columns)
    types = df.dtypes.tolist()
    return list(zip(columns, types))