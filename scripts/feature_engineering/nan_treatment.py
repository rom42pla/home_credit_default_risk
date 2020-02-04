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
    if log:
        sectionTimer = Timer(log=f"removing columns with more than {threshold * 100}% of NaNs")

    non_none_values = int(df.shape[0] * (1 - threshold))
    df_clean = df.dropna(thresh=non_none_values, axis=1).reset_index(drop=True)
    dropped_cols = list(set(df.columns) - set(df_clean.columns))

    if log:
        sectionTimer.end_timer(log=f"removed {df.shape[1] - df_clean.shape[1]} columns")
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
    if log:
        section_timer = Timer(log=f"removing rows with more than {threshold * 100}% of NaNs")

    non_none_values = int(df.shape[1] * (1 - threshold))
    df_clean = df.dropna(thresh=non_none_values, axis=0).reset_index(drop=True)

    if log:
        section_timer.end_timer(log=f"removed {df.shape[0] - df_clean.shape[0]} rows")
    return df_clean

def align(df1, df2, log=False):
    if log:
        section_timer = Timer(log=f"removing columns of the second dataframe that are not in the first")

    new_df1, new_df2 = df1.align(df2, join="left", axis=1)
    new_df1, new_df2 = new_df1.dropna(axis=1, how="all"), new_df2.dropna(axis=1, how="all")

    if log:
        section_timer.end_timer(log=f"done, with final shapes of {new_df1.shape} and {new_df2.shape}")
    return new_df1, new_df2

def impute_missing_values(df, mode="simple", columns=None, log=False):
    if log: section_timer = Timer(log=f"imputing missing values")

    if columns == []:
        columns = None

    if columns == None:
        df_to_impute = df
    else:
        df_to_impute = df[columns]

    if "TARGET" in df_to_impute.columns:
        old_cols_wo_TARGET = df_to_impute.drop(columns=["TARGET"]).columns
        X, y = df_to_impute.drop(columns=["TARGET"]).to_numpy(), df_to_impute.loc[:, "TARGET"].to_numpy()
    else:
        old_cols_wo_TARGET = df_to_impute.columns
        X = df_to_impute.to_numpy()

    if mode.lower().strip() == "simple 0":
        X_pred = SimpleImputer(strategy="constant", fill_value=0).fit_transform(X)
    elif mode.lower().strip() == "simple median":
        X_pred = SimpleImputer(strategy="median", copy=False).fit_transform(X)
    elif mode.lower().strip() == "simple mean":
        X_pred = SimpleImputer(strategy="mean", copy=False).fit_transform(X)
    elif mode.lower().strip() == "iterative":
        X_pred = IterativeImputer().fit_transform(X)
    elif mode.lower().strip() == "knn":
        X_pred_splits = np.array_split(X, 4)
        for i, X_pred_split in enumerate(X_pred_splits):
            X_pred_splits[i] = KNNImputer().fit_transform(X_pred_split)
        X_pred = np.concatenate(X_pred_splits, axis=0)
    else:
        raise Exception(f'Unrecognized mode f{mode.strip()}.\nOnly supported modes are "simple", "iterative", "knn"')

    if columns == None:
        df_new = pd.DataFrame(columns=old_cols_wo_TARGET, data=X_pred)
    else:
        df_imputed, df_new = pd.DataFrame(columns=old_cols_wo_TARGET, data=X_pred), df
        df_new = df_new.drop(columns=list(df_imputed.columns))
        for col in df_imputed.columns:
            df_new[col] = df_imputed[col].to_numpy()

    if "TARGET" in df_to_impute.columns:
        df_new["TARGET"] = y

    if log:
        section_timer.end_timer(log=f"done")

    return parsing.reduce_dataframe_size(df_new, log=False)
