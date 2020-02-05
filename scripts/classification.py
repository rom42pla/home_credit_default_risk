import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

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
        if log: section_timer = Timer(log=f"tuning Logistic Regression classifier")
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

def svm(X_train=None, X_validate=None, y_train=None, y_validate=None, tuning=False, log=False):
    classifier = LinearSVC(max_iter=3000)
    return classifier

def adaboost(X_train=None, X_validate=None, y_train=None, y_validate=None, tuning=False, log=False):
    classifier = AdaBoostClassifier(base_estimator=LogisticRegression(dual=False, max_iter=300, n_jobs=1),n_estimators=250)
    return classifier

def predict(X_train, X_test, y_train, X_validate=None, y_validate=None, mode="ensemble", tuning=False, probabilities=True, log=False):
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
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, reference=lgb_train)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'n_jobs': 6,
            'verbosity': 1
        }

        gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

        if log: section_timer.end_timer(log=f"done")
        return y_pred, None

    else:
        raise Exception(f'Unrecognized mode f{mode.strip()}.\nOnly supported modes are "ensemble", "bayes", "logistic", "rf", "mlp", "knn"')

    # prediction
    if probabilities:
        y_pred = classifier.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    else:
        y_pred = classifier.fit(X_train, y_train).predict(X_test)
    if classifier_name not in ["SVM"]:
        proba = classifier.predict_proba(X_test)
    else:
        proba = None

    if log: section_timer.end_timer(log=f"done")
    return y_pred, proba