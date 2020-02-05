import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from Timer import Timer

def pca_transform(df_train, dfs_to_transform, corr_threshold=0.8, log=False):
    if log: section_timer = Timer(log=f"computing PCA")

    if "TARGET" in df_train.columns:
        df_train, y_train = df_train.drop(columns=["TARGET"]), df_train.loc[:, "TARGET"].to_numpy().tolist()
    else:
        y_train = None

    ys_to_transform = [None, None, None]
    for i, df_to_transform in enumerate(dfs_to_transform):
        if "TARGET" in df_to_transform.columns:
            dfs_to_transform[i], ys_to_transform[i] = df_to_transform.drop(columns=["TARGET"]), df_to_transform.loc[:, "TARGET"].to_numpy().tolist()
        else:
            ys_to_transform[i] = None

    # normalizing training set
    normalizer = Normalizer().fit_transform(df_train)

    # feature selection
    selected_features = feature_selection_new(df=df_train, y_train=y_train, corr_threshold=corr_threshold, log=False)
    df_train, dfs_to_transform = df_train[selected_features], [df[selected_features] for df in dfs_to_transform]

    # PCA
    pca = PCA(whiten=True).fit(df_train)
    selected_components = list(df_train.columns)

    result = []
    for i, df_to_transform in enumerate(dfs_to_transform):
        result.append(pd.DataFrame(data=pca.transform(df_to_transform)))

    for i, df_to_transform in enumerate(dfs_to_transform):
        if ys_to_transform[i] != None:
            result[i]["TARGET"] = ys_to_transform[i]

    if log:
        section_timer.end_timer(log=f"found {len(df_train.columns)} components")
    return result


def feature_selection_new(df, y_train=None, corr_threshold=0.8, log=False):
    if log: section_timer = Timer(log=f"finding the features")

    if y_train != None:
        df["TARGET"] = y_train
    elif "TARGET" not in df.columns:
        raise ValueError("Target missing in the dataframe")

    corr_series = abs(pd.Series(df.corr()["TARGET"]).drop("TARGET")).sort_values(ascending=False)
    columns = corr_series.index.tolist()[:int(len(corr_series) * corr_threshold)]

    if log:
        section_timer.end_timer(log=f"selected {len(columns)} (out of {len(df.columns)}) features")

    return columns