# !python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from Timer import Timer
from feature_engineering import encoding, nan_treatment, anomalies_treatment
import main


def parse_CSV_to_df(file_path, lines_cap=None, reduce_size=False, log=False):
    """
    Parses a .csv file into a dataframe.

    :param file_path:
        Path of the file being parsed to a pandas' dataframe
    :param lines_cap:
        Number of lines being parsed
        If None, parses all the lines
    :param log:
        Flag for log on the console

    :return df:
        Returns fetched dataframe
    """
    if log:
        section_timer = Timer(
            log=f"parsing file {file_path} ({'{0:.1f}'.format(os.path.getsize(file_path) * 2 ** (-20))}Mb)")

    df = pd.read_csv(file_path, nrows=lines_cap, index_col=False, encoding="utf_8", header=0)

    if reduce_size:
        df = reduce_dataframe_size(df.infer_objects())

    if log:
        section_timer.end_timer(log=f"parsed a dataframe of {df.shape[0]} rows and {df.shape[1]} features")

    return df


def write_df_to_file(df, file_path, index=False, header=True, log=False):
    """
    :param df:
        Pandas' dataframe being written to a file
    :param file_path:
        Path of the file being creating from a pandas' dataframe
    """
    if log:
        section_timer = Timer(
            log=f"writing file {file_path}")

    df.to_csv(path_or_buf=file_path, index=index, header=header, sep=",")

    if log:
        section_timer.end_timer(log=f"written a dataframe of {df.shape[0]} rows and {df.shape[1]} features")

def merge_dfs(dfs, data_path, groupby_mode="mean", just_one_hot=False, log=False):  # to join all the other dataframes.
    """
    Merges the dataframe to the original one, to have a dataframe complete with all the information.
    :param original_dataframe:
        dataframe where we add all the others
    :param data_path:
        location where there are all the data
    :param log:
        flag for the time
    :return:
        dataframe completed
    """
    if log: section_timer = Timer(log=f"merging the dataframes")

    bureau, prev_application = __joining_minor_csvs(data_path, just_one_hot=just_one_hot, groupby_mode="mean")
    for i in range(len(dfs)):
        dfs[i] = pd.merge(left=dfs[i], right=bureau,
                          how='left', on="SK_ID_CURR", left_index=True)
        dfs[i] = pd.merge(left=dfs[i], right=prev_application,
                          how='left', on="SK_ID_CURR", left_index=True)

    if log: section_timer.end_timer(log=f"join completed")

    return [reduce_dataframe_size(df) for df in dfs]

def __joining_minor_csvs(data_path, groupby_mode="mean", just_one_hot=False, log=False):  # to join all the other dataframes.
    """
    Merges the bureau and previous_application to all their lower levels dataframes.
    :param data_path:
        location where there are all the data
    :return:
        The bureau and previous_application ready to be merged, encoded and grouped by SK_ID_CURR
    """

    #----------------------#
    #------# bureau #------#
    #----------------------#
    bureau = pd.read_csv(data_path + "bureau.csv")  # the dataframe to add to the old one
    bureau = anomalies_treatment.correct_nan_values(bureau, log=False)
    bureau, _ = nan_treatment.remove_columns(df=bureau, threshold=0.5, log=False)
    bureau = encoding.frequency_encoding(bureau, just_one_hot=just_one_hot)

    bur_balance = pd.read_csv(data_path + "bureau_balance.csv")  # the dataframe to add to the old one
    bur_balance = anomalies_treatment.correct_nan_values(bur_balance, log=False)
    bur_balance, _ = nan_treatment.remove_columns(df=bur_balance, threshold=0.5, log=log)
    bur_balance = encoding.frequency_encoding(bur_balance, just_one_hot=just_one_hot)

    if groupby_mode == "mean":
        bur_balance = bur_balance.groupby("SK_ID_BUREAU", as_index=False).mean()
    elif groupby_mode == "sum":
        bur_balance = bur_balance.groupby("SK_ID_BUREAU", as_index=False).sum()
    elif groupby_mode == "mode":
        bur_balance = bur_balance.groupby("SK_ID_BUREAU", as_index=False).mode()
    else:
        raise ValueError("Only recognized mode are 'sum' and 'mean'")

    bureau = pd.merge(left=bureau, right=bur_balance,
                      how='left', on="SK_ID_BUREAU", left_index=True)
    if groupby_mode == "mean":
        bureau = bureau.groupby("SK_ID_CURR", as_index=False).mean()
    elif groupby_mode == "sum":
        bureau = bureau.groupby("SK_ID_CURR", as_index=False).sum()
    elif groupby_mode == "mode":
        bureau = bureau.groupby("SK_ID_CURR", as_index=False).mode()
    else:
        raise ValueError("Only recognized mode are 'sum', 'mean' and 'mode'")

    #--------------------------------#
    #------# prev_application #------#
    #--------------------------------#
    prev_application = pd.read_csv(data_path + "previous_application.csv")
    prev_application = anomalies_treatment.correct_nan_values(prev_application)
    prev_application, _ = nan_treatment.remove_columns(df=prev_application, threshold=0.5, log=False)
    prev_application = encoding.frequency_encoding(prev_application, just_one_hot=just_one_hot)

    # merging each dataframe to prev_application
    csvs_to_add = [("POS_CASH_balance.csv", "SK_ID_PREV"),
                   ("credit_card_balance.csv", "SK_ID_PREV"),
                   ("installments_payments.csv", "SK_ID_PREV")]
    for csv_name, key in csvs_to_add:
        df_to_join = pd.read_csv(data_path + csv_name).drop(columns="SK_ID_CURR")  # the dataframe to add to the old one
        df_to_join = anomalies_treatment.correct_nan_values(df_to_join)
        df_to_join, _ = nan_treatment.remove_columns(df=df_to_join, threshold=0.5, log=False)
        df_to_join = encoding.frequency_encoding(df_to_join, just_one_hot=just_one_hot)
        if groupby_mode == "mean":
            df_to_join = df_to_join.groupby(key, as_index=False).mean()
        elif groupby_mode == "sum":
            df_to_join = df_to_join.groupby(key, as_index=False).sum()
        elif groupby_mode == "mode":
            df_to_join = df_to_join.groupby(key, as_index=False).mode()
        else:
            raise ValueError("Only recognized mode are 'sum' and 'mean'")


        prev_application = pd.merge(left=prev_application, right=df_to_join,
                                    how='left', on=key, left_index=True)

    if groupby_mode == "mean":
        prev_application = prev_application.groupby("SK_ID_CURR", as_index=False).mean()
    elif groupby_mode == "sum":
        prev_application = prev_application.groupby("SK_ID_CURR", as_index=False).sum()
    elif groupby_mode == "sum":
        prev_application = prev_application.groupby("SK_ID_CURR", as_index=False).mean()
    else:
        raise ValueError("Only recognized mode are 'sum' and 'mean'")

    return bureau, prev_application

def reduce_dataframe_size(df, log=False):
    size_MB_before = get_size(df)
    for col in df.columns:
        if df[col].dtype == "object":
            continue
        max, min = df[col].max(), df[col].min()
        is_int = all(True if v != None and float(v).is_integer() else False for v in df[col].tolist())

        if is_int:
            if min >= 0:
                if max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if min > np.iinfo(np.int8).min and max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif min > np.iinfo(np.int16).min and max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif min > np.iinfo(np.int32).min and max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif min > np.iinfo(np.int64).min and max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

        else:
            df[col] = df[col].astype(np.float32)
    size_MB_after = get_size(df)
    if log: print(f"\t...reduced size by {int(size_MB_after / size_MB_before * 100)}%")
    return df

def get_size(df):
    size_MB = df.memory_usage().sum() / 1024**2
    return size_MB