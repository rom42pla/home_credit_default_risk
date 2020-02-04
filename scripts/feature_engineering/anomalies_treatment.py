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


def correct_nan_values(df, file_path=None, log=False):
    if log: section_timer = Timer(log=f"searching for anomalies")

    for col in df.columns:
        # if it's a numerical column
        if df[col].dtype != "object" and "sk_id" not in col.lower().strip():
            unique_values = set(df[col].unique())
            #if log: print(f"\nColumn: {col}\t{len(unique_values)} unique values")
            df[col] = __check_unuseful_col(df[col], log=False)
            # if it's a continuous feature
            if len(unique_values) >= 50:
                df[col] = __check_single_sign_value(df[col], log=False)
                df[col] = __check_frequency_anomaly(df[col], log=False)
            # if it's a discrete continuous feature
            elif len(unique_values) > 2:
                df[col] = __check_single_sign_value(df[col], log=False)



                ''''
                min, max, mean = df[col].min(), df[col].max(), df[col].mean()
                negative_values_count, positive_values_count = sum(n < 0 for n in df[col]), \
                                                               sum(n > 0 for n in df[col])
                if log:
                    print(f"\tmin: {min}\tmax: {max}\tmean: {mean}\tstandard deviation: {'%.3f' % std}")
                    print(
                        f"\tnegative values: {negative_values_count} ({'%.4f'%((negative_values_count/n_values)*100)}%)"
                        f"\tpositive values: {positive_values_count} ({'%.4f'%((positive_values_count/n_values)*100)}%)")
                '''

    # writes to file
    if file_path != None:
        with open(file_path, "w") as fp:
            pprint("cool", stream=fp)

    if log: section_timer.end_timer(log=f"done")

    return df


def __check_unuseful_col(series, log=False):
    #series = series.dropna()
    unique_values = set(series.unique())
    percs = series.value_counts(normalize=True)
    if percs.iloc[0] > 0.95:
        if log: print(f"########--------########--------\tfound a majority value")
        series = series.replace(to_replace=list(unique_values), value=np.nan)
    return series

def __check_single_sign_value(series, log=False):
    # gets useful values
    unique_values = set(series.unique())
    negative_values_unique, positive_values_unique = set([n for n in unique_values if n < 0]), \
                                                     set([n for n in unique_values if n > 0])
    if len(negative_values_unique) == 1:
        if log: print(f"########--------########--------\tfound a single unique negative value")
        series = series.replace(to_replace=list(negative_values_unique), value=np.nan)
    elif len(positive_values_unique) == 1:
        if log: print(f"########--------########--------\tfound a single unique positive value")
        series = series.replace(to_replace=list(positive_values_unique), value=np.nan)
    return series

def __check_frequency_anomaly(series, log=False):
    unique_values = set(series.unique())
    if len(unique_values) == 1:
        return series

    # gets useful values
    min, max, mean, std = series.min(), series.max(), series.mean(), series.std()
    min_perc, max_perc = series.value_counts(normalize=True).loc[min], series.value_counts(normalize=True).loc[max]
    negative_values_count, zero_values_count, positive_values_count = sum(n < 0 for n in series), \
                                                                      sum(n == 0 for n in series), \
                                                                      sum(n >= 0 for n in series)
    # checks something
    if min_perc > 0.05 and mean - min > 2 * std:
        if log: print(f"########--------########--------\tfound min")
    if max_perc > 0.05 and mean + max > 2 * std:
        if log: print(f"########--------########--------\tfound max")

    return series
