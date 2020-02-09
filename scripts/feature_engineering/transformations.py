import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures
from feature_engineering import sampling, nan_treatment
import parsing
from sklearn.ensemble import ExtraTreesClassifier
from Timer import Timer

def pca_transform(dfs_to_transform, corr_threshold=0.7, PCA_n_components=None, old_cols=None, log=False):
    if log: section_timer = Timer(log=f"computing PCA")

    ys_to_transform = [None, None, None]
    for i, df_to_transform in enumerate(dfs_to_transform):
        if "TARGET" in df_to_transform.columns:
            dfs_to_transform[i], ys_to_transform[i] = df_to_transform.drop(columns=["TARGET"]), df_to_transform.loc[:, "TARGET"].to_numpy().tolist()
        else:
            ys_to_transform[i] = None

    concatenated_dfs, indexes = pd.DataFrame(columns=dfs_to_transform[1].columns), []
    for i, df in enumerate(dfs_to_transform):
        indexes += [concatenated_dfs.shape[0]]
        concatenated_dfs = pd.concat([concatenated_dfs, df], sort=False).reset_index(drop=True)

    if log: print(f"\t...scaling dataframes...")

    # scaling
    concatenated_dfs = scale(concatenated_dfs)

    if log: print(f"\t...adding polynomial features...")
    # polynomial features
    cols_to_poly = set(feature_selection(dfs_to_transform[0], y_train=ys_to_transform[0], corr_threshold=corr_threshold)) & set(old_cols)
    concatenated_dfs = add_poly_features(concatenated_dfs, features_list=cols_to_poly)
    
    if log: print(f"\t...finding principal components...")

    # PCA
    pca = PCA(n_components=PCA_n_components, whiten=True, svd_solver="auto")
    concatenated_dfs = pd.DataFrame(data=pca.fit_transform(concatenated_dfs))
    selected_components = list(concatenated_dfs.columns)

    # reproducing dfs
    dfs_to_transform = []
    for i, index in enumerate(indexes):
        if i < len(indexes) - 1:
            dfs_to_transform.append(concatenated_dfs.iloc[indexes[i]:indexes[i+1], :])
        else:
            dfs_to_transform.append(concatenated_dfs.iloc[indexes[i]:, :])

    selected_components = list(feature_selection(dfs_to_transform[0], ys_to_transform[0], corr_threshold=0.7, log=False))
    dfs_to_transform = [df[selected_components] for df in dfs_to_transform]

    for i, df_to_transform in enumerate(dfs_to_transform):
        if ys_to_transform[i] != None:
            df_to_transform = dfs_to_transform[i]
            df_to_transform.insert(1, "TARGET", ys_to_transform[i])
            dfs_to_transform[i] = df_to_transform

    if log:
        section_timer.end_timer(log=f"found {len(selected_components)} components")
    return dfs_to_transform

def feature_selection(df, y_train, corr_threshold=0.8, log=False):
    if log: section_timer = Timer(log=f"finding the features")

    features_to_keep = int(len(df.columns) * corr_threshold)
    columns = ExtraTreesClassifier(n_estimators=100).fit(df.to_numpy(), y_train).feature_importances_
    columns, correlations = pd.Series(columns).sort_values(ascending=False).index.tolist()[:features_to_keep], pd.Series(columns).sort_values(ascending=False).tolist()[:features_to_keep]
    columns = [df.columns[i] for i in columns]

    if log:
        section_timer.end_timer(log=f"selected {len(columns)} (out of {len(df.columns)}) features")

    return columns

def scale(df):
    cols = list(df.columns)
    for col in cols:
        if len(df[col].unique()) <= 2:
            cols.remove(col)
    if "TARGET" in df.columns:
        cols.remove("TARGET")

    df[cols] = pd.DataFrame(data=StandardScaler().fit_transform(df[cols]), columns=cols)

    return df

def add_poly_features(df, features_list=None):
    if features_list == None: #if the list of features to transform isn't passed, transform everything
        features_list = list(df.columns)

    features_list = parsing.get_quantitative_features(df[features_list], min_unique_values=20)
    
    # create the polynomial object with specified degree
    df = nan_treatment.impute_missing_values(df, mode="simple median")
    poly_transformer = PolynomialFeatures(degree=2, order="F").fit(df[features_list])

    # transform the features
    transformed_matrix = poly_transformer.transform(df[features_list])
    
    poly_df = pd.DataFrame(data=transformed_matrix, columns=poly_transformer.get_feature_names(features_list))

    if set(features_list) != set(df.columns):  # if the list of features to transform is passed:
        poly_df = pd.concat([df.drop(columns=features_list), poly_df], axis=1)

    return poly_df

def discretize_columns(df):
    if "DAYS_EMPLOYED" in df.columns:
        prev_col = df['DAYS_EMPLOYED']
        df['DAYS_EMPLOYED'] = pd.cut(prev_col, bins =[prev_col.min()-1, -8000, -5500, -3500, -2000, 0, prev_col.max()], labels = False)
    if "OWN_CAR_AGE" in df.columns:
        prev_col = df["OWN_CAR_AGE"]
        df['OWN_CAR_AGE'] = pd.cut(prev_col, bins = [-1,5,15,30,40, prev_col.max()], labels = False)
    if "DAYS_ID_PUBLISH" in df.columns:
        prev_col = df["DAYS_ID_PUBLISH"]
        df['DAYS_ID_PUBLISH'] = pd.cut(prev_col, bins = [prev_col.min()-1,-4000, prev_col.max()], labels = False)
    if "DAYS_LAST_PHONE_CHANGE" in df.columns:
        prev_col = df["DAYS_LAST_PHONE_CHANGE"]
        df['DAYS_LAST_PHONE_CHANGE'] = pd.cut(prev_col, bins = [prev_col.min()-1,-2500, -1000, prev_col.max()], labels = False)
    if "DAYS_REGISTRATION" in df.columns:
        prev_col = df["DAYS_REGISTRATION"]
        df['DAYS_REGISTRATION'] = pd.cut(prev_col, bins = [prev_col.min()-1,-10000, -5000, -2500, prev_col.max()], labels = False)
    if "HOUR_APPR_PROCESS_START" in df.columns:
        prev_col = df["HOUR_APPR_PROCESS_START"]
        df["HOUR_APPR_PROCESS_START"] = pd.cut(prev_col, bins = [prev_col.min()-1, prev_col.quantile(0.2), prev_col.quantile(0.4), prev_col.quantile(0.6), prev_col.quantile(0.8), prev_col.max()], labels = False)
    return df

def log_transformation(df):
    quantitative_cols = parsing.get_quantitative_features(df, min_unique_values=20)
    df[quantitative_cols] = df[quantitative_cols].apply(lambda col: np.log(col + abs(min(col)) + 1), axis=1)
    return df

