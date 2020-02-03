from pprint import pprint
from Timer import Timer
import matplotlib.pyplot as plt

def show_unique_values(df, file_path=None, log=False):
    if log:
        section_timer = Timer(
            log=f"searching for unique values")

    unique_values = {}
    for col in df.columns:
        # if it's a discrete column
        if df[col].dtype == "object":
            print(col)
            unique_values[col] = set(df[col].tolist())
        # pene
    pprint(unique_values)

    if file_path != None:
        with open(file_path, "w") as fp:
            pprint(unique_values, stream=fp)

    if log:
        section_timer.end_timer(log=f"done")

def plot_hists(df, img_path):
    df.hist(figsize=(60,50))
    plt.savefig(img_path)

def plot_column_hists(df, column, img_path):
    n_unique_values = len(set(df[column].tolist()))
    df[column].hist(bins=50)
    print(df[column].max(), df[column].min(), df[column].mean())
    print(df[column].value_counts(normalize=True).sort_values())
    plt.savefig(img_path)