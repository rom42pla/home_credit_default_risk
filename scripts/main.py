import pandas as pd
import logging

import parsing
import classification
import evaluation
from feature_engineering import nan_treatment, encoding, sampling, anomalies_treatment, transformations
from Timer import Timer


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.CRITICAL)
    '''
    ////////////////////////
    P A R A M E T E R S ////
    ////////////////////////
    '''
    # behavioural parameters
    log = True

    # nan treatment parameters
    cols_to_discretize = ["DAYS_EMPLOYED", "OWN_CAR_AGE", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE", "DAYS_REGISTRATION"]
    columns_threshold, rows_threshold = 1, 0.3  # max null values for not being dropped

    # sampling parameters
    do_sampling = True
    sampling_mode = "undersampling"

    # merge parameters
    read_merged_files = True   # skip the feature engineering part
    do_merge = True
    do_imputing = True
    groupby_mode = "mean"

    # PCA/FAMD parameters
    read_PCA_files = True
    do_PCA = True
    just_one_hot = False  # switch between one-hot and frequency encoding
    features_corr_threshold = 0.7
    PCA_n_components = None

    # machine learning parameters
    classifier = "lgbt"
    k_fold_splits = 3
    hyperparameters_tuning = False
    predict_probabilities = True

    # paths
    data_path, logs_path, imgs_path = "../../data/", "../../logs/", "../imgs"
    file_path_train, file_path_test = data_path + "application_train.csv", data_path + "application_test.csv"
    df_train_preprocessed_path, df_test_preprocessed_path = logs_path + "df_train_preprocessed.csv", logs_path + "df_test_preprocessed.csv"
    df_train_after_PCA_path, df_test_after_PCA_path = logs_path + "df_train_after_PCA.csv", logs_path + "df_test_after_PCA.csv"
    cols_before_merge_path, df_train_merged_path, df_validate_merged_path, df_test_merged_path = logs_path + "cols_before_merge.csv", logs_path + "df_train_merged.csv", logs_path + "df_validate_merged.csv", logs_path + "df_test_merged.csv"
    df_train_description_path, df_validate_description_path, df_test_description_path = logs_path + "df_train_description.csv", logs_path + "df_validate_description.csv", logs_path + "df_test_description.csv"
    submission_path = logs_path + "submission.csv"

    '''
    //////////////////////////
    F E A T U R E ////////////
    E N G I N E E R I N G ////
    //////////////////////////
    '''
    if log: total_timer = Timer(log=f"starting FDS 2019/20 project")

    if not read_merged_files:
        # retrieving data from the .csv(s)
        df_train = parsing.parse_CSV_to_df(file_path=file_path_train, log=log)
        df_test = parsing.parse_CSV_to_df(file_path=file_path_test, log=log)

        # removing anomalies
        df_train = anomalies_treatment.remove_useless_columns(df=df_train, log=log)
        df_test = anomalies_treatment.remove_useless_columns(df=df_test, log=False)
        df_train = anomalies_treatment.correct_nan_values(df_train, log=log)
        df_test = anomalies_treatment.correct_nan_values(df_test, log=log)

        # removing useless columns and rows
        df_train, dropped_cols = nan_treatment.remove_columns(df=df_train, threshold=columns_threshold, log=log)
        df_train = nan_treatment.remove_rows(df=df_train, threshold=rows_threshold, log=log)
        target = df_train["TARGET"]
        df_train = df_train.drop(columns=["TARGET"])
        df_train, df_test = nan_treatment.align_left(df_train, df_test, log=log)
        
        # discretizing some columns
        df_train = transformations.discretize_columns(df=df_train)
        df_test = transformations.discretize_columns(df=df_test)

        # one-hot encoding
        first_test_element = df_train.shape[0]
        concatenated_dfs = pd.concat([df_train, df_test], sort=False)
        concatenated_dfs = encoding.frequency_encoding(concatenated_dfs, just_one_hot=just_one_hot, log=log)
        df_train, df_test = concatenated_dfs.iloc[:first_test_element, :], \
                            concatenated_dfs.iloc[first_test_element:, :]
        df_train = pd.concat([df_train, target], axis=1, join="inner")

        # merging with other dataframes
        old_cols = set(df_train.columns)
        if do_merge:
            df_train, df_test = parsing.merge_dfs(dfs=[df_train, df_test],
                                                  just_one_hot=just_one_hot, data_path=data_path,
                                                  groupby_mode=groupby_mode, do_imputing=do_imputing, log=log)

        # removing IDs
        df_train = anomalies_treatment.remove_ids(df_train)
        df_test = anomalies_treatment.remove_ids(df_test)

        # removing infinites
        df_train = anomalies_treatment.remove_infs(df_train)
        df_test = anomalies_treatment.remove_infs(df_test)
        
        new_cols = []
        for col in df_train.columns:
            if not col in old_cols:
                new_cols.append(col)
        if "TARGET" in new_cols:
            new_cols.remove("TARGET")

        # imputing missing values
        df_train = nan_treatment.impute_missing_values(df_train, mode="simple 0", columns=new_cols, log=False)
        df_test = nan_treatment.impute_missing_values(df_test, mode="simple 0", columns=new_cols, log=False)

        # removing useless rows and columns
        df_train = nan_treatment.remove_rows(df=df_train, threshold=rows_threshold, log=False)
        df_train, _ = nan_treatment.remove_columns(df=df_train, threshold=columns_threshold, log=False)
        df_test, _ = nan_treatment.remove_columns(df=df_test, threshold=columns_threshold, log=False)

        # imputing missing values
        df_train = nan_treatment.impute_missing_values(df_train, mode="simple median", log=log)
        df_test = nan_treatment.impute_missing_values(df_test, mode="simple median", log=log)

        if do_sampling:
            # under/oversampling and creating a validation set
            if sampling_mode in ["oversample", "oversampling", "over sampling"]:
                df_train = sampling.oversample(df_train, log=log)
            elif sampling_mode in ["undersample", "undersampling", "under sampling"]:
                df_train = sampling.undersample(df_train, log=log)
            else:
                df_train = sampling.smote(df_train, log=log)
            target = df_train["TARGET"]

        # log transformations
        df_train = transformations.log_transformation(df=df_train)
        df_test = transformations.log_transformation(df=df_test)

        # removing infinites
        df_train = anomalies_treatment.remove_infs(df_train)
        df_test = anomalies_treatment.remove_infs(df_test)

        # saves the dataframes to files
        parsing.write_df_to_file(pd.DataFrame(data=old_cols), cols_before_merge_path, header=False, log=False)
        parsing.write_df_to_file(df_train, df_train_preprocessed_path, log=log)
        parsing.write_df_to_file(df_test, df_test_preprocessed_path, log=log)

    else:
        # reads dataframes backups
        old_cols = parsing.parse_CSV_to_df(file_path=cols_before_merge_path, log=False).iloc[:, 0].tolist()
        df_train = parsing.parse_CSV_to_df(file_path=df_train_preprocessed_path, log=log)
        df_test = parsing.parse_CSV_to_df(file_path=df_test_preprocessed_path, log=log)

    '''
    //////////
    P C A ////
    //////////
    '''

    if not read_PCA_files:
        y_train = df_train.loc[:, "TARGET"].to_numpy()
        df_test, df_train = nan_treatment.align_left(df_test, df_train, log=False)
        df_train["TARGET"] = y_train

        # PCA
        if do_PCA:
            df_train, df_test = transformations.pca_transform([df_train, df_test], corr_threshold=features_corr_threshold, PCA_n_components=PCA_n_components, old_cols=old_cols, log=log)

        parsing.write_df_to_file(df_train, df_train_after_PCA_path, log=log)
        parsing.write_df_to_file(df_test, df_test_after_PCA_path, log=log)

    else:
        df_train = parsing.parse_CSV_to_df(file_path=df_train_after_PCA_path, log=log)
        df_test = parsing.parse_CSV_to_df(file_path=df_test_after_PCA_path, log=log)

    '''
    ////////////////////////
    P R E D I C T I O N ////
    ////////////////////////
    '''
    test_ids = parsing.parse_CSV_to_df(file_path=file_path_test, log=False)["SK_ID_CURR"]
    X_train, y_train = df_train.drop(columns=["TARGET"]).to_numpy(), df_train["TARGET"]
    X_test = df_test.to_numpy()
    features = list(df_test.columns)

    y_test_pred, proba = classification.predict(X_train=X_train, X_test=X_test, X_validate=X_train, y_train=y_train,
                                                y_validate=y_train, mode=classifier, tuning=hyperparameters_tuning,
                                                probabilities=predict_probabilities, k_fold_splits=k_fold_splits, log=log)

    df_submission = pd.DataFrame(columns=["SK_ID_CURR", "TARGET"])
    df_submission["SK_ID_CURR"], df_submission["TARGET"] = test_ids, y_test_pred
    parsing.write_df_to_file(df=df_submission, file_path=submission_path, log=log)
    parsing.write_df_to_file(df=df_submission, file_path="../submission.csv", log=log)

    if log: total_timer.end_timer(log=f"everything done")