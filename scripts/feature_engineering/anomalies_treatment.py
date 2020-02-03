import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

def remove_ids(df):
    for col in df.columns:
        if "sk_id" in col.lower().strip() and col in df.columns:
            df = df.drop(columns=[col])
    return df

def remove_infs(df):
    df = df.replace([np.inf, -np.inf, "inf", "inf"], np.nan)
    return df

def show_unique_values(df, file_path=None, log=False):
    if log:
        section_timer = Timer(
            log=f"searching for unique values")

    for col in df.columns:
        # if it's a numerical column
        if df[col].dtype != "object":
            values = df[col]
            # if there are no nan's
            if not values.isnull().values.any():
                unique_values = set(values.tolist())
                # if it's a continuous feature
                if len(unique_values) > 10:
                    print(f"Column {col}")
                    min, max, mean = values.min(), values.max(), values.mean()
                    negative_values_count, positive_values_count = sum(n < 0 for n in values), sum(n > 0 for n in values)
                    negative_values_unique, positive_values_unique = [n for n in unique_values if n<0], [n for n in unique_values if n>0]
                    print(negative_values_unique)
                    print(f"min: {min}\tmax: {max}\tmean: {mean}")
                    print(f"negative values: {negative_values_count}\tpositive values: {positive_values_count}")

                    #print(unique_values)
                    #print(values)

                #print(values.isnull().value_counts(normalize=True))

    #pprint(unique_values)

    if file_path != None:
        with open(file_path, "w") as fp:
            pprint(unique_values, stream=fp)

    if log:
        section_timer.end_timer(log=f"done")