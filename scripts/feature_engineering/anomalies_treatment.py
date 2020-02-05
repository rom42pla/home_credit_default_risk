import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Timer import Timer

def remove_ids(df):
    for col in df.columns:
        if "sk_id" in col.lower().strip() and col in df.columns:
            df = df.drop(columns=[col])
    return df

def remove_infs(df):
    df = df.replace([np.inf, -np.inf, "inf", "inf"], np.nan)
    return df


def correct_nan_values(df, log=False):
    if log: section_timer = Timer(log=f"searching for anomalies")

    # loop through each column
    for col in df.columns:
        # if it's a numerical column (we operate only on them)
        if df[col].dtype != "object" and "sk_id" not in col.lower().strip():
            unique_values = set(df[col].unique())

            # we don't want to delete columns with over 90% of same values
            #df[col] = __check_unuseful_col(df[col], log=False)

            # we want to delete single values with opposite sign
            df[col] = __check_single_sign_value(df[col], log=False)

            # if it's a continuous feature
            if len(unique_values) >= 50:
                df[col] = __check_frequency_anomaly(df[col], log=False)

    if log: section_timer.end_timer(log=f"done")

    return df

def __check_single_sign_value(series, log=False):
    """
    Transforms single values to nones
    :param series: the column
    :param log:
    :return: the transformed column
    """
    # gets useful values
    negative_values_unique, positive_values_unique = set(series[series < 0]), \
                                                     set(series[series > 0])
    if len(negative_values_unique) == 1 and len(positive_values_unique) > 1:
        series = series.replace(to_replace=list(negative_values_unique), value=np.nan)
    elif len(positive_values_unique) == 1 and len(negative_values_unique) > 1:
        series = series.replace(to_replace=list(positive_values_unique), value=np.nan)

    return series

def __check_frequency_anomaly(series, log=False):
    """
    Transforms values with a lot of frequency and far from the mean to nones
    :param series: the column
    :param log:
    :return: transformed column
    """
    # pass if there is just one unique value
    unique_values = set(series.unique())
    if len(unique_values) == 1: return series

    # gets useful values
    min, max, mean, std = series.min(), series.max(), series.mean(), series.std()
    min_perc, max_perc = series.value_counts(normalize=True).loc[min], series.value_counts(normalize=True).loc[max]

    # checks percentages and distances from standard deviation
    if min_perc > 0.05 and mean - min > 2 * std:
        series = series.replace(to_replace=min, value=np.nan)
    if max_perc > 0.05 and mean + max > 2 * std:
        series = series.replace(to_replace=max, value=np.nan)

    return series

# maybe useless
def __check_unuseful_col(series, log=False):
    # gets useful values
    unique_values = set(series.unique())
    percs = series.value_counts(normalize=True)
    # if the column has nearly a single value
    if percs.iloc[0] > 0.99:
        series = series.replace(to_replace=list(unique_values), value=np.nan)
    return series