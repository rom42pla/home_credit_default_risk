import pandas as pd
import numpy as np
import parsing

from Timer import Timer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

def remove_columns(df, threshold, log=False):
    """
    Removes columns with more than a percentage of NaN values

    :param df:
        Input dataframe
    :param threshold:
        Maximum percentage of NaN values for columns
    :param log:
        Flag for log on the console
    :return:
    """
    if log: sectionTimer = Timer(log=f"removing columns with more than {threshold * 100}% of nans")
            
    # removes columns with many nans
    non_nan_values = int(df.shape[0] * (1 - threshold))
    df_clean = df.dropna(thresh=non_nan_values, axis=1)
    dropped_cols = list(set(df.columns) - set(df_clean.columns))

    if log: sectionTimer.end_timer(log=f"removed {len(set(df.columns)) - df_clean.shape[1]} columns")
    return df_clean, dropped_cols

def remove_rows(df, threshold, log=False):
    """
    Removes rows with more than a percentage of NaN values

    :param df:
        Input dataframe
    :param threshold:
        Maximum percentage of NaN values for rows
    :param log:
        Flag for log on the console
    :return:
    """
    if log: section_timer = Timer(log=f"removing rows with more than {threshold * 100}% of NaNs")

    non_nan_values = int(df.shape[1] * (1 - threshold))
    df_clean = df.dropna(thresh=non_nan_values, axis=0)

    if log: section_timer.end_timer(log=f"removed {df.shape[0] - df_clean.shape[0]} rows")
    return df_clean

def align_left(df1, df2, log=False):
    if log: section_timer = Timer(log=f"removing columns of the second dataframe that are not in the first")

    df1, df2 = df1.align(df2, join="inner", axis=1)

    if log: section_timer.end_timer(log=f"done, with final shapes of {df1.shape} and {df2.shape}")
    return df1, df2

def impute_missing_values(df, mode="simple", columns=None, reduce_size=False, log=False):
    if columns == None:
        columns_to_impute = list(df.columns)
    elif columns == []:
        return df
    else:
        columns_to_impute = columns

    if log: section_timer = Timer(log=f"imputing missing values")

    X = df[columns_to_impute].to_numpy()

    if mode.lower().strip() == "simple 0":
        imputer = SimpleImputer(strategy="constant", fill_value=0)

    elif mode.lower().strip() == "simple median":
        imputer = SimpleImputer(strategy="median", copy=False)

    elif mode.lower().strip() == "simple mean":
        imputer = SimpleImputer(strategy="mean", copy=False)

    elif mode.lower().strip() == "simple most common":
        imputer = SimpleImputer(strategy="most_frequent", copy=False)

    elif mode.lower().strip() == "iterative":
        imputer = IterativeImputer(max_iter=3, n_nearest_features=5)

    else:
        raise Exception(f'Unrecognized mode f{mode.strip()}.\nOnly supported modes are "simple 0", "simple mean", "simple median", "simple most common", "iterative"')

    X_pred = imputer.fit_transform(X)

    df[columns_to_impute] = X_pred

    if reduce_size:
        df = parsing.reduce_dataframe_size(df, log=False)

    if log:
        section_timer.end_timer(log=f"done")

    return df
