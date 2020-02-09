import numpy as np
import pandas as pd 

import itertools
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from feature_engineering import transformations
import lightgbm as lgb

from Timer import Timer

def random_forest(X_train=None, X_validate=None, y_train=None, y_validate=None, tuning=False, log=False):
    # positive class symbol (usually 1)
    positive_label = list(filter(lambda value: "1" in str(value), set(y_train.tolist())))[0]
    # tuning the classifier
    if tuning:
        if log: section_timer = Timer(log=f"tuning Random Forest classifier")
        # testing several parameters
        bestScore, n_estimators_best, max_samples_best, max_features_best = 0, None, None, None
        for n_estimators in [1, 100, 500, 1000]:
            for max_samples in [0.1, 0.25]:
                for max_features in ["sqrt", "log2"]:
                    classifier = RandomForestClassifier(n_estimators=n_estimators, criterion="gini",
                                                        max_features=max_features, class_weight=None,
                                                        bootstrap=True, warm_start=True,
                                                        max_samples=max_samples, n_jobs=4).fit(X_train, y_train)
                    score = roc_auc_score(y_validate, classifier.predict(X_validate))
                    if (score > bestScore):
                        bestScore, n_estimators_best, max_samples, max_features_best = \
                            score, n_estimators, max_samples, max_features
        # choosing best parameters
        classifier = RandomForestClassifier(n_estimators=n_estimators_best, criterion="gini",
                                            max_features=max_features_best, class_weight=None,
                                            bootstrap=True, warm_start=True,
                                            max_samples=max_samples, n_jobs=4)
        if log: section_timer.end_timer(log=f"done with a max score of {bestScore}")
    # default classifier
    else:
        classifier = RandomForestClassifier(n_estimators=300, criterion="gini", bootstrap=True, max_samples=0.2, n_jobs=4)

    return classifier

def naive_bayes(X_train=None, X_validate=None, y_train=None, y_validate=None, tuning=False, log=False):
    return GaussianNB()

def logistic_regression(X_train=None, X_validate=None, y_train=None, y_validate=None, tuning=False, log=False):
    # positive class symbol (usually 1)
    positive_label = list(filter(lambda value: "1" in str(value), set(y_train.tolist())))[0]
    # tuning the classifier
    if tuning:
        if log: section_timer = Timer(log=f"tuning Logistic Regression classifier")
        # testing several parameters
        bestScore, solver_best = 0, None
        for solver in ["liblinear", "lbfgs", "newton-cg", "saga"]:
            classifier = LogisticRegression(solver=solver, dual=False,
                                            warm_start=True, max_iter=500, n_jobs=4, C=1).fit(X_train, y_train)
            score = roc_auc_score(y_validate, classifier.predict(X_validate))
            if (score > bestScore):
                bestScore, solver_best = \
                    score, solver
        # choosing best parameters
        classifier = LogisticRegression(solver=solver_best, dual=False,
                                        warm_start=True, max_iter=1000, n_jobs=4, C=1)
        if log: section_timer.end_timer(log=f"done with a max score of {bestScore}")
    # default classifier
    else:
        classifier = LogisticRegression(dual=False, max_iter=1000, n_jobs=4, C=1)

    return classifier

def multilayer_perceptron(X_train=None, X_validate=None, y_train=None, y_validate=None, tuning=False, log=False):
    # positive class symbol (usually 1)
    positive_label = list(filter(lambda value: "1" in str(value), set(y_train.tolist())))[0]
    # tuning the classifier
    if tuning:
        if log: section_timer = Timer(log=f"tuning Multilayer Perceptron classifier")
        # testing several parameters
        bestScore, activation_best, learning_rate_best = 0, None, None
        for activation in ["logistic", "relu"]:
            for learningRate in ["constant", "adaptive"]:
                classifier = MLPClassifier(activation=activation, learning_rate=learningRate,
                                           solver="adam", max_iter=200).fit(X_train, y_train)
                score = f1_score(y_validate, classifier.predict(X_validate))
                if (score > bestScore):
                    bestScore, activation_best, learning_rate_best = score, activation, learningRate
        # choosing best parameters
        classifier = MLPClassifier(activation=activation_best, learning_rate=learning_rate_best,
                                   solver="adam", max_iter=200)
        if log: section_timer.end_timer(log=f"done with a max score of {bestScore}")
    # default classifier
    else:
        classifier = MLPClassifier(solver="adam", activation="relu", learning_rate="constant", max_iter=200)

    return classifier

def knn(X_train=None, X_validate=None, y_train=None, y_validate=None, tuning=False, log=False):
    # positive class symbol (usually 1)
    positive_label = list(filter(lambda value: "1" in str(value), set(y_train.tolist())))[0]
    # tuning the classifier
    if tuning:
        if log: section_timer = Timer(log=f"tuning KNN classifier")
        # testing several parameters
        bestScore, n_neighbors_best = 0, None
        for neighbors in [2, 8]:
            classifier = KNeighborsClassifier(n_neighbors=neighbors, weights="distance", p=2,
                                              n_jobs=4).fit(X_train, y_train)
            score = f1_score(y_validate, classifier.predict(X_validate))
            if (score > bestScore):
                bestScore, n_neighbors_best = score, neighbors
        # choosing best parameters
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors_best, weights="distance",
                                          p=2, n_jobs=4)
        if log: section_timer.end_timer(log=f"done with a max score of {bestScore}")
    # default classifier
    else:
        classifier = KNeighborsClassifier(n_neighbors=2, weights="distance", p=2, n_jobs=4)

    return classifier

def lda(X_train=None, X_validate=None, y_train=None, y_validate=None, tuning=False, log=False):
    # positive class symbol (usually 1)
    positive_label = list(filter(lambda value: "1" in str(value), set(y_train.tolist())))[0]
    # tuning the classifier
    if tuning:
        if log: section_timer = Timer(log=f"tuning Linear Discriminant Analysis classifier")
        # testing several parameters
        bestScore, solver_best, shrinkage_best = 0, "svd", None
        for solver in ["svd", "eigen", "lsqr"]:
            if solver != "svd":
                for shrinkage in [None, "auto"]:
                    classifier = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage).fit(X_train, y_train)
                    score = f1_score(y_validate, classifier.predict(X_validate))
                    if (score > bestScore):
                        bestScore, solver_best, shrinkage_best = score, solver, shrinkage
        # choosing best parameters
        classifier = LinearDiscriminantAnalysis(solver=solver_best, shrinkage=shrinkage_best)
        if log: section_timer.end_timer(log=f"done with a max score of {bestScore}")
    # default classifier
    else:
        classifier = LinearDiscriminantAnalysis()

    return classifier

def svm(X_train=None, X_validate=None, y_train=None, y_validate=None, tuning=False, log=False):
    classifier = SVC(kernel='poly')
    return classifier

def adaboost(X_train=None, X_validate=None, y_train=None, y_validate=None, tuning=False, log=False):
    classifier = AdaBoostClassifier(base_estimator=LogisticRegression(dual=False, max_iter=300, n_jobs=1),n_estimators=250)
    return classifier

def predict(X_train, X_test, y_train, X_validate=None, y_validate=None, mode="ensemble", tuning=False, probabilities=True, k_fold_splits=3, log=False):
    # ensemble
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
    if mode.lower().strip() in ["ensemble", "voting"]:
        classifier_name = "Ensemble"
        if log: section_timer = Timer(log=f"predicting using {classifier_name} classifier")
        classifiers = [
            ("Random forest", random_forest(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning, log=log)),
            #("Naive Bayes", naive_bayes(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning, log=log)),
            ("Logistic Regression", logistic_regression(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning, log=log)),
            ("MLP", multilayer_perceptron(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning, log=log)),
            ("KNN", knn(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning, log=log)),
            #("SVM", svm(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning, log=log)),
            ("AdaBoost", adaboost(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning,
                         log=log))
            ]

        classifier = VotingClassifier(estimators=classifiers, voting='soft')

    # random forest classifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    elif mode.lower().strip() in ["random forest", "rf", "forest"]:
        classifier_name = "Random Forest"
        if log: section_timer = Timer(log=f"predicting using {classifier_name} classifier")
        classifier = random_forest(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning, log=log)

    # naive bayes
    # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    elif mode.lower().strip() in ["bayes", "naive bayes", "nb"]:
        classifier_name = "Naive Bayes"
        if log: section_timer = Timer(log=f"predicting using {classifier_name} classifier")
        classifier = naive_bayes(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning, log=log)

    # logistic regression
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    elif mode.lower().strip() in ["logistic", "logistic regression", "regression"]:
        classifier_name = "Logistic Regression"
        if log: section_timer = Timer(log=f"predicting using {classifier_name} classifier")
        classifier = logistic_regression(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning, log=log)

    # multilayer perceptron
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    elif mode.lower().strip() in ["mlp", "multilayer perceptron", "perceptron"]:
        classifier_name = "Multilayer Perceptron"
        if log: section_timer = Timer(log=f"predicting using {classifier_name} classifier")
        classifier = multilayer_perceptron(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning, log=log)

    # K nearest neighbors classifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    elif mode.lower().strip() in ["knn", "nearest neighbors"]:
        classifier_name = "K-Nearest Neighbors"
        if log: section_timer = Timer(log=f"predicting using {classifier_name} classifier")
        classifier = knn(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning, log=log)

    # Support Vector Machine
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
    elif mode.lower().strip() in ["svm"]:
        classifier_name = "SVM"
        if log: section_timer = Timer(log=f"predicting using {classifier_name} classifier")
        classifier = svm(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning,
                         log=log)

    # AdaBoost
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    elif mode.lower().strip() in ["adaboost", "ada boost", "ada"]:
        classifier_name = "AdaBoost"
        if log: section_timer = Timer(log=f"predicting using {classifier_name} classifier")
        classifier = adaboost(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate, tuning=tuning,
                         log=log)

    elif mode.lower().strip() in ["lgb", "lgbt"]:
        classifier_name = "lgb"

        if log: section_timer = Timer(log=f"predicting using {classifier_name} classifier")
        features = X_train
        test_features = X_test
        best_preds, best_score = np.zeros(X_test.shape[0]), 0
        if tuning:
            # parameters are:
            # learning_rate, max_bin, num_leaves, min_data_in_leaf, max_depth, lambdal1, lambdal2
            parameters_combinations = list(itertools.product([0.05, 0.1], [100, 250, 500], [8, 512, 2048], [100, 500, 1000, 2500], [-1, 5, 10], [0, 0.25, 0.5], [0, 0.25, 0.5]))
            best_combination = parameters_combinations[0]
            
            for i, combination in enumerate(parameters_combinations):
                if log: print(f"\n\t...trying combination {i + 1} of {len(parameters_combinations)}, with a current best score of {best_score} and combination {best_combination}...\n")
                try:
                    learning_rate, max_bin, num_leaves, min_data_in_leaf, max_depth, lambdal1, lambdal2 = combination
                    model = lgb.LGBMClassifier(n_estimators=500, objective='binary', n_jobs=-1, verbose=-1,
                                            class_weight='balanced', device="cpu",
                                            learning_rate=learning_rate,
                                            reg_alpha=0.1, reg_lambda=0.1, min_data_in_leaf=min_data_in_leaf,
                                            bagging_fraction=0.25, bagging_freq=5, 
                                            max_bin=max_bin, num_leaves=num_leaves, max_depth=max_depth,
                                            lambdal1=lambdal1, lambdal2=lambdal2)

                    test_predictions = np.zeros(X_test.shape[0])
                    
                    for train_indices, valid_indices in KFold(n_splits=k_fold_splits, shuffle=True, random_state=42).split(features):
                        train_features, train_labels = features[train_indices], y_train[train_indices]
                        valid_features, valid_labels = features[valid_indices], y_train[valid_indices]

                        # training
                        model = model.fit(train_features, train_labels, eval_metric='auc',
                                eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                                eval_names=['test', 'train'], categorical_feature='auto',
                                early_stopping_rounds=500, verbose=-1)
                        best_iteration = model.best_iteration_
                        print(model.best_score_)
                        train_score, test_score = model.best_score_["train"]["auc"] - model.best_score_["train"]["binary_logloss"], model.best_score_["test"]["auc"] - - model.best_score_["test"]["binary_logloss"]
                        
                        # if we are over/underfitting, current parameters are bad
                        if train_score - test_score > 0.05 or train_score - test_score < -0.05: break

                        # prediction
                        test_predictions += model.predict_proba(test_features, num_iteration=best_iteration)[:, 1] / k_fold_splits

                        # updates parameters
                        if test_score > best_score:
                            best_combination, best_preds, best_score = combination, test_predictions, test_score
                except:
                    continue
            if log: section_timer.end_timer(log=f"found best combination {best_combination} and best score {best_score}")
        else:
            learning_rate, max_bin, num_leaves, min_data_in_leaf, max_depth, lambdal1, lambdal2 = (0.05, 100, 8, 100, -1, 0, 0)
            '''
            model = lgb.LGBMClassifier(n_estimators=500, objective='binary', n_jobs=-1, verbose=-1,
                                            class_weight='balanced', device="cpu",
                                            learning_rate=learning_rate,
                                            reg_alpha=0.1, reg_lambda=0.1, min_data_in_leaf=min_data_in_leaf,
                                            bagging_fraction=0.25, bagging_freq=5, 
                                            max_bin=max_bin, num_leaves=num_leaves, max_depth=max_depth,
                                            lambdal1=lambdal1, lambdal2=lambdal2)
            '''
            model = lgb.LGBMClassifier(n_estimators=1000, objective='binary', n_jobs=-1,
                                       class_weight='balanced', learning_rate=0.05,
                                       reg_alpha=0.3, reg_lambda=0.2,
                                       max_bin=50)

            test_predictions = np.zeros(X_test.shape[0])

            i = 0

            for train_indices, valid_indices in KFold(n_splits=k_fold_splits, shuffle=True).split(features):
                i += 1
                print("\n----------------> ", i)
                
                train_features, train_labels = features[train_indices], y_train[train_indices]
                valid_features, valid_labels = features[valid_indices], y_train[valid_indices]

                # training
                model = model.fit(train_features, train_labels, eval_metric='auc',
                        eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                        eval_names=['test', 'train'], categorical_feature='auto',
                        early_stopping_rounds=500, verbose=-1)
                best_iteration = model.best_iteration_
                best_score = max(best_score, model.best_score_["test"]["auc"])

                # prediction
                test_predictions += model.predict_proba(test_features, num_iteration=best_iteration)[:, 1]

            if log: section_timer.end_timer(log=f"best score: {best_score}")

        return test_predictions / k_fold_splits, None

    elif mode.lower().strip() in ["lda", "linear discriminant"]:
        classifier_name = "Linear Discriminant Analysis"
        if log: section_timer = Timer(log=f"predicting using {classifier_name} classifier")
        classifier = lda(X_train=X_train, X_validate=X_validate, y_train=y_train, y_validate=y_validate,
                              tuning=tuning, log=log)

    elif mode.lower().strip() in ["gb", "gradient boosting"]:
        classifier_name = "Gradient Boosting"
        if log: section_timer = Timer(log=f"predicting using {classifier_name} classifier")
        classifier = GradientBoostingClassifier()

    else:
        raise Exception(f'Unrecognized mode f{mode.strip()}.\nOnly supported modes are "ensemble", "bayes", "logistic", "rf", "mlp", "knn", "lda"')

    # prediction
    if probabilities and classifier_name != "SVM":
        y_pred = classifier.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    else:
        y_pred = classifier.fit(X_train, y_train).predict(X_test)
    if classifier_name not in ["SVM"]:
        proba = classifier.predict_proba(X_test)
    else:
        proba = None

    if log: section_timer.end_timer(log=f"done")
    return y_pred, proba