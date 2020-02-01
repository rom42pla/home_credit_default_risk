import pandas as pd

import parsing
import classification
import evaluation
from feature_engineering import nan_treatment, encoding, sampling, anomalies_treatment, PCA
from Timer import Timer


if __name__ == "__main__":
    '''
    P A R A M E T E R S
    '''
    log = True
    data_path, logs_path, imgs_path = "../../data/", "../../logs/", "../imgs"
    file_path_train, file_path_test = data_path + "application_train.csv", data_path + "application_test.csv"
    df_train_preprocessed_path, df_validate_preprocessed_path, df_test_preprocessed_path = logs_path + "df_train_preprocessed.csv", logs_path + "df_validate_preprocessed.csv", logs_path + "df_test_preprocessed.csv"
    df_train_description_path, df_validate_description_path, df_test_description_path = logs_path + "df_train_description.csv", logs_path + "df_validate_description.csv", logs_path + "df_test_description.csv"
    lines_cap = None
    read_preprocessed_files = False
    columns_threshold, rows_threshold = 0.5, 0.1

    '''
    P R O G R A M
    '''
    if log: total_timer = Timer(log=f"starting FDS 2019/20 project")
    if not read_preprocessed_files:
        # fetching data from the .csv(s)
        df_train_original = parsing.parse_CSV_to_df(file_path=file_path_train, lines_cap=lines_cap, log=log)
        df_test_original = parsing.parse_CSV_to_df(file_path=file_path_test, lines_cap=lines_cap, log=log)
        df_train, df_test = df_train_original.copy(), df_test_original.copy()

        # removing useless columns and rows
        df_train, dropped_cols = nan_treatment.remove_columns(df=df_train, threshold=columns_threshold, log=log)
        df_train = nan_treatment.remove_rows(df=df_train, threshold=rows_threshold, log=log)
        df_train, df_test = nan_treatment.align(df_train, df_test, log=log)

        # one-hot encoding
        first_test_element = df_train.shape[0]
        concatenated_dfs = pd.concat([df_train, df_test], sort=False)
        concatenated_dfs = encoding.frequency_encoding(concatenated_dfs)
        df_train, df_test = concatenated_dfs.iloc[:first_test_element, :], concatenated_dfs.iloc[first_test_element:, :].drop(columns=["TARGET"])

        # undersampling and creating a validation set
        df_train, df_validate = sampling.undersample(df_train, log=log)
        df_validate = pd.concat([df_validate, df_train], sort=False)
        df_validate, _ = sampling.undersample(df_validate, log=False)

        # merging with other dataframes
        old_cols = set(df_train.columns)
        df_train, df_validate, df_test = parsing.merge_dfs(dfs=[df_train, df_validate, df_test], data_path=data_path, log=log)

        #removing IDs
        df_train = anomalies_treatment.remove_ids(df_train)
        df_validate = anomalies_treatment.remove_ids(df_validate)
        df_test = anomalies_treatment.remove_ids(df_test)

        new_cols = [col if not col in old_cols else None for col in list(df_train.columns)]
        res = []
        for val in new_cols:
            if val != None:
                res.append(val)
        new_cols = res
        df_train = nan_treatment.impute_missing_values(df_train, mode="simple", columns=new_cols, log=log)

        # removing useless columns and rows
        df_train, dropped_cols = nan_treatment.remove_columns(df=df_train, threshold=columns_threshold, log=log)
        df_train = nan_treatment.remove_rows(df=df_train, threshold=rows_threshold, log=log)
        target_train, target_validate = df_train["TARGET"], df_validate["TARGET"]
        df_train, df_test = nan_treatment.align(df_train, df_test, log=False)
        df_test, df_train = nan_treatment.align(df_test, df_train, log=False)
        df_train, df_validate = nan_treatment.align(df_train, df_validate, log=False)
        df_validate, df_train = nan_treatment.align(df_validate, df_train, log=False)
        df_train["TARGET"], df_validate["TARGET"] = target_train, target_validate

        # describes stats on features
        df_train_description, df_test_description = df_train.describe(), df_test.describe()
        df_train_description.columns, df_test_description.columns = df_train.columns, df_test.columns
        parsing.write_df_to_file(df_train_description, df_train_description_path, index=True)
        parsing.write_df_to_file(df_test_description, df_test_description_path, index=True)

        # imputing missing values
        df_train = nan_treatment.impute_missing_values(df_train, mode="knn", log=log)
        df_validate = nan_treatment.impute_missing_values(df_validate, mode="knn", log=log)
        df_test = nan_treatment.impute_missing_values(df_test, mode="knn", log=log)

        # PCA:
        df_train, df_validate, df_test = PCA.pca_transform(df_train, [df_train, df_validate, df_test])

        # saves the dataframes to files
        print(f"Saving preprocessed dataframes as .csv...")
        parsing.write_df_to_file(df_train, df_train_preprocessed_path)
        parsing.write_df_to_file(df_validate, df_validate_preprocessed_path)
        parsing.write_df_to_file(df_test, df_test_preprocessed_path)
        print(f"\t...saved preprocessed dataframes inside {logs_path}")

    else:
        df_train = parsing.parse_CSV_to_df(file_path=df_train_preprocessed_path, lines_cap=lines_cap, log=log)
        df_validate = parsing.parse_CSV_to_df(file_path=df_validate_preprocessed_path, lines_cap=lines_cap, log=log)
        df_test = parsing.parse_CSV_to_df(file_path=df_test_preprocessed_path, lines_cap=lines_cap, log=log)

    X_train, y_train = df_train.drop(columns=["TARGET"]).to_numpy(), df_train.loc[:, "TARGET"].to_numpy()
    X_validate, y_validate = df_validate.drop(columns=["TARGET"]).to_numpy(), df_validate.loc[:, "TARGET"].to_numpy()
    X_test = df_test.to_numpy()

    print(f"Shapes:")
    print(f"\tX_train: {X_train.shape}\ty_train: {y_train.shape}")
    print(f"\tX_validate: {X_validate.shape}\ty_validate: {y_validate.shape}")
    print(f"\tX_test: {X_test.shape}")

    y_validate_pred, proba = classification.predict(X_train=X_train, X_test=X_validate, X_validate=X_validate, y_train=y_train,
                                         y_validate=y_validate, mode="logistic", tuning=False)
    #y_test_pred = classification.predict(X_train=X_train, X_test=X_test, X_validate=X_validate, y_train=y_train, y_validate=y_validate, mode="bayes", tuning=False)

    evaluation.get_classification_report(y_validate, y_validate_pred, imgs_path)

    evaluation.get_roc_auc(y_validate, y_validate_pred, proba, imgs_path)

    if log: total_timer.end_timer(log=f"everything done")