# !python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
from Timer import Timer
from feature_engineering import encoding


def parse_CSV_to_df(file_path, lines_cap=None, log=False):
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
        sectionTimer = Timer(
            log=f"parsing file {file_path} ({'{0:.1f}'.format(os.path.getsize(file_path) * 2 ** (-20))}Mb)")

    df = pd.read_csv(file_path, nrows=lines_cap, index_col=False, encoding="utf_8", header=0)

    if log:
        sectionTimer.end_timer(log=f"parsed a dataframe of {df.shape[0]} rows and {df.shape[1]} features")

    return df.infer_objects()


def write_df_to_file(df, file_path, index=False):
    """
    :param df:
        Pandas' dataframe being written to a file
    :param file_path:
        Path of the file being creating from a pandas' dataframe
    """
    df.to_csv(path_or_buf=file_path, index=index, sep=",")


def merge_dfs(dfs, data_path, log=False):  # to join all the other dataframes.
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

    # list of the dataframes and the key for the merge
    dataframes = [("bureau.csv", "SK_ID_CURR"),
                  ("bureau_balance.csv", "SK_ID_BUREAU"),
                  ("previous_application.csv", "SK_ID_CURR"),
                  ("POS_CASH_balance.csv", "SK_ID_PREV"),
                  ("credit_card_balance.csv", "SK_ID_PREV"),
                  ("installments_payments.csv", "SK_ID_PREV")]
    merged_dfs = dfs.copy()


    for csv_name, key in dataframes:  # merging each dataframe
        df_to_join = pd.read_csv(data_path + csv_name)  # the dataframe to add to the old one

        df_to_join = encoding.frequency_encoding(df_to_join)

        # group the information for averaging all the rows
        df_to_join = df_to_join.groupby(key, as_index=False).mean()
        for i, df in enumerate(dfs):
            merged_dfs[i] = pd.merge(left=merged_dfs[i], right=df_to_join, how='left', on=key, left_index=True)  # merging the dataframes

    if log: section_timer.end_timer(log=f"join completed")

    return merged_dfs


def new_merge_dfs(dfs, data_path, log=False):  # to join all the other dataframes.

    # reducing the dataframes:
    bureau = pd.merge(left=merged_dfs[i], right=df_to_join, how='left', on=key,
                      left_index=True)  # merging the dataframes

    dataframes = [("POS_CASH_balance.csv", "SK_ID_PREV"),
                  ("credit_card_balance.csv", "SK_ID_PREV"),
                  ("installments_payments.csv", "SK_ID_PREV")]
    for csv_name, key in dataframes:  # merging each dataframe
        df_to_join = pd.read_csv(data_path + csv_name)  # the dataframe to add to the old one

        df_to_join.drop(columns="SK_ID_CURR", inplace=True)

        prev_appication = pd.merge(left=prev_appication, right=df_to_join, how='left', on=key, left_index=True)  # merging the dataframes

