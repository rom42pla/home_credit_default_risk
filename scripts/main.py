import pandas as pd
import logging

import parsing
import classification
import evaluation
from feature_engineering import nan_treatment, encoding, sampling, anomalies_treatment, PCA
from tests import feature_screening
from Timer import Timer


if __name__ == "__main__":
    '''
    P A R A M E T E R S
    '''
    # behavioural parameters
    log = True # choose to show logs of many important operations into the console
    columns_threshold, rows_threshold = 0.5, 0.1 # max null values for not being dropped
    groupby_mode = "mean"
    classifier = "regression"

    read_merged_files = False  # skip the feature engineering part
    do_merge = True
    do_PCA = True

    # paths
    data_path, logs_path, imgs_path = "../../data/", "../../logs/", "../imgs"
    file_path_train, file_path_test = data_path + "application_train.csv", data_path + "application_test.csv"
    df_train_preprocessed_path, df_validate_preprocessed_path, df_test_preprocessed_path = logs_path + "df_train_preprocessed.csv", logs_path + "df_validate_preprocessed.csv", logs_path + "df_test_preprocessed.csv"
    cols_before_merge_path, df_train_merged_path, df_validate_merged_path, df_test_merged_path = logs_path + "cols_before_merge.csv", logs_path + "df_train_merged.csv", logs_path + "df_validate_merged.csv", logs_path + "df_test_merged.csv"
    df_train_description_path, df_validate_description_path, df_test_description_path = logs_path + "df_train_description.csv", logs_path + "df_validate_description.csv", logs_path + "df_test_description.csv"
    submission_path = logs_path + "submission.csv"

    '''
    P R O G R A M
    '''
    logging.getLogger().setLevel(logging.CRITICAL)
    if log: total_timer = Timer(log=f"starting FDS 2019/20 project")
    if not read_merged_files:
        # retrieving data from the .csv(s)
        df_train_original = parsing.parse_CSV_to_df(file_path=file_path_train, log=log)
        df_test_original = parsing.parse_CSV_to_df(file_path=file_path_test, log=log)
        df_train, df_test = df_train_original.copy(), df_test_original.copy()

        # removing anomalies
        df_train = anomalies_treatment.correct_nan_values(df_train, log=log)
        df_test = anomalies_treatment.correct_nan_values(df_test, log=log)

        # removing useless columns and rows
        df_train, dropped_cols = nan_treatment.remove_columns(df=df_train, threshold=columns_threshold, log=log)
        df_train = nan_treatment.remove_rows(df=df_train, threshold=rows_threshold, log=log)
        df_train, df_test = nan_treatment.align(df_train, df_test, log=log)

        # one-hot encoding
        first_test_element = df_train.shape[0]
        concatenated_dfs = pd.concat([df_train, df_test], sort=False)
        concatenated_dfs = encoding.frequency_encoding(concatenated_dfs, log=log)
        df_train, df_test = concatenated_dfs.iloc[:first_test_element, :], concatenated_dfs.iloc[first_test_element:, :].drop(columns=["TARGET"])

        # undersampling and creating a validation set
        df_train, df_validate = sampling.undersample(df_train, log=log)
        df_validate = pd.concat([df_validate, df_train], sort=False)
        df_validate, _ = sampling.undersample(df_validate, log=False)

        # merging with other dataframes
        old_cols = set(df_train.columns)
        if do_merge:
            df_train, df_validate, df_test = parsing.merge_dfs(dfs=[df_train, df_validate, df_test],
                                                               data_path=data_path, groupby_mode=groupby_mode, log=log)

        # saves the dataframes to files
        parsing.write_df_to_file(pd.DataFrame(data=old_cols), cols_before_merge_path, header=False, log=log)
        parsing.write_df_to_file(df_train, df_train_preprocessed_path, log=log)
        parsing.write_df_to_file(df_validate, df_validate_preprocessed_path, log=log)
        parsing.write_df_to_file(df_test, df_test_preprocessed_path, log=log)

    else:
        # reads dataframes backups
        old_cols = parsing.parse_CSV_to_df(file_path=cols_before_merge_path, log=log).iloc[:, 0].tolist()
        df_train = parsing.parse_CSV_to_df(file_path=df_train_preprocessed_path, log=log)
        df_validate = parsing.parse_CSV_to_df(file_path=df_validate_preprocessed_path, log=log)
        df_test = parsing.parse_CSV_to_df(file_path=df_test_preprocessed_path, log=log)

    # removing IDs
    df_train = anomalies_treatment.remove_ids(df_train)
    df_validate = anomalies_treatment.remove_ids(df_validate)
    df_test = anomalies_treatment.remove_ids(df_test)

    # removing infinites
    df_train = anomalies_treatment.remove_infs(df_validate)
    df_validate = anomalies_treatment.remove_ids(df_validate)
    df_test = anomalies_treatment.remove_ids(df_test)

    new_cols = []
    for col in df_train.columns:
        if not col in old_cols:
            new_cols.append(col)
    df_train = nan_treatment.impute_missing_values(df_train, mode="simple 0", columns=new_cols, log=log)
    df_validate = nan_treatment.impute_missing_values(df_validate, mode="simple 0", columns=new_cols, log=log)
    df_test = nan_treatment.impute_missing_values(df_test, mode="simple 0", columns=new_cols, log=log)

    # removing useless columns and rows
    df_train, dropped_cols = nan_treatment.remove_columns(df=df_train, threshold=columns_threshold, log=log)
    df_train = nan_treatment.remove_rows(df=df_train, threshold=rows_threshold, log=log)
    target_train, target_validate = df_train["TARGET"], df_validate["TARGET"]
    df_train, df_test = nan_treatment.align(df_train, df_test, log=False)
    df_test, df_train = nan_treatment.align(df_test, df_train, log=False)
    df_train, df_validate = nan_treatment.align(df_train, df_validate, log=False)
    df_validate, df_train = nan_treatment.align(df_validate, df_train, log=False)
    df_train["TARGET"], df_validate["TARGET"] = target_train, target_validate

    # imputing missing values
    df_train = nan_treatment.impute_missing_values(df_train, mode="simple mean", log=log)
    df_validate = nan_treatment.impute_missing_values(df_validate, mode="simple mean", log=log)
    df_test = nan_treatment.impute_missing_values(df_test, mode="simple mean", log=log)

    # PCA
    if do_PCA:
        df_train, df_validate, df_test = PCA.pca_transform(df_train, [df_train, df_validate, df_test], log=log)
    test_ids = parsing.parse_CSV_to_df(file_path=file_path_test, log=False)["SK_ID_CURR"]
    X_train, y_train = df_train.drop(columns=["TARGET"]).to_numpy(), df_train.loc[:, "TARGET"].to_numpy()
    X_validate, y_validate = df_validate.drop(columns=["TARGET"]).to_numpy(), df_validate.loc[:, "TARGET"].to_numpy()
    X_test = df_test.to_numpy()

    features = list(df_test.columns)
    #y_validate_pred, proba = classification.predict(X_train=X_train, X_test=X_validate, X_validate=X_validate, y_train=y_train,
                                                        #y_validate=y_validate, mode="logistic", tuning=False, log=log)
    y_test_pred, proba = classification.predict(X_train=X_train, X_test=X_test, X_validate=X_validate, y_train=y_train, \
                                            y_validate=y_validate, mode=classifier, tuning=False, log=log)
    #evaluation.get_confusion_matrix(y_validate, y_validate_pred)
    #evaluation.get_classification_report(y_validate, y_validate_pred, imgs_path)
    #evaluation.get_roc_auc(y_validate, y_validate_pred, proba, imgs_path)
    #evaluation.features_importance(X_validate, y_validate, features, imgs_path)

    df_submission = pd.DataFrame(columns=["SK_ID_CURR"])
    df_submission["SK_ID_CURR"], df_submission["TARGET"] = test_ids, y_test_pred
    parsing.write_df_to_file(df=df_submission, file_path=submission_path, log=log)

    if log: total_timer.end_timer(log=f"everything done")