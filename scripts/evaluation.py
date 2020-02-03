import numpy as np 
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve, roc_auc_score

def get_classification_report(y_true, y_pred, imgs_path):
    report = classification_report(y_true, y_pred, target_names=["Not at risk", "At risk"], digits=4)
    print(f"Classification report:\n{report}")

    scores_names, scores = ["accuracy", "precision", "recall", "f1", "auc"], \
                           [accuracy_score(y_true, y_pred),
                            precision_score(y_true, y_pred),
                            recall_score(y_true, y_pred),
                            f1_score(y_true, y_pred),
                            roc_auc_score(y_true, y_pred) ]
    fig = plt.figure()
    plt.title('Classification report')
    x = np.arange(len(scores))
    plt.bar(x, height=scores)
    plt.xticks(x, scores_names)
    plt.legend(loc='lower right')
    plt.ylim([0, 1])
    plt.ylabel('score')
    plt.xlabel('measures')
    fig.savefig(imgs_path + '/classification_report.png', dpi=fig.dpi)
    return scores_names, scores

def get_confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"Confusion matrix:\n\tTN:\t{tn}\tFP:\t{fp}\n\tFN:\t{fn}\tTP:\t{tp}")
    return tn, fp, fn, tp

def features_importance(X, y, feature_names, imgs_path, max_features=10):
    forest = ExtraTreesClassifier(n_estimators=100)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    features_order = list(np.argsort(importances)[::-1])
    importances = importances[features_order]

    for i, feature_index in enumerate(features_order):
        features_order[i] = feature_names[feature_index]
    print(f"Most important features:\n\t{features_order[:max_features]}")

    fig = plt.figure()
    plt.title('Features importance')
    x = np.arange(len(features_order[:max_features]))
    plt.bar(x, height=importances[:max_features])
    plt.xticks(x, features_order[:max_features])
    plt.legend(loc='lower right')
    plt.ylabel('score')
    plt.xlabel('features')
    fig.savefig(imgs_path + '/features_importance.png', dpi=fig.dpi)
    # plots feature importances
    #images.plotBarChart(x=featuresOrder, y=importances, title="Feature importances", textSize=8)
    #images.plotBarChart(x=featuresOrder[:20], y=importances[:20], title="Feature importances (top 20)", textSize=8)

    return features_order, importances

def count_values(df, column):
    return df.loc[:, column].value_counts()

def get_roc_auc(y_true, y_pred, proba, imgs_path):
    preds = proba[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.title('ROC')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    fig.savefig(imgs_path + '/roc.png', dpi=fig.dpi)