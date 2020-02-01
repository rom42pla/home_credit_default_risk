import pandas as pd

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def pca_transform(df_train, dfs_to_transform):
    if "TARGET" in df_train.columns:
        df_train, y_train = df_train.drop(columns=["TARGET"]), df_train.loc[:, "TARGET"].to_numpy().tolist()
    ys_to_transform = [None, None, None]
    for i, df_to_transform in enumerate(dfs_to_transform):
        if "TARGET" in df_to_transform.columns:
            dfs_to_transform[i], ys_to_transform[i] = df_to_transform.drop(columns=["TARGET"]), df_to_transform.loc[:, "TARGET"].to_numpy().tolist()

    std_pca = make_pipeline(StandardScaler(), PCA(n_components=0.8)).fit(df_train)
    #components = std_pca.transform(df_train).shape[1]

    result = []
    for i, df_to_transform in enumerate(dfs_to_transform):
        df_to_transform = std_pca.transform(df_to_transform)
        df_to_transform = pd.DataFrame(data=df_to_transform)
        result.append(df_to_transform)

    for i, df_to_transform in enumerate(dfs_to_transform):
        if ys_to_transform[i] != None:
            result[i]["TARGET"] = ys_to_transform[i]

    return result