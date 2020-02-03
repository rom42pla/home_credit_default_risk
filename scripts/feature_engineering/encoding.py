import pandas as pd
from Timer import Timer


def frequency_encoding(df):
    """
    encodes the dataframe with the frequencies on the categorical features.
    :param df: dataframe to be encoded
    :param log: true if we want to set the timer on
    :return: encoded dataframe
    """
    old_cols = set(df.columns)
    df = pd.get_dummies(df)
    new_cols = list(set(df.columns) - old_cols)
    for col in new_cols:
        df[col] = (df[col] * (df[col].sum() / df.shape[0])).astype("float32")
    return df

def get_columns_types(df):
    columns = list(df.columns)
    types = df.dtypes.tolist()
    return list(zip(columns, types))