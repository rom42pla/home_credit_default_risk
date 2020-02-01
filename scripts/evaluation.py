import numpy as np 
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve

def get_classification_report(y_true, y_pred, imgs_path):
    report = classification_report(y_true, y_pred, target_names=["Not at risk", "At risk"], digits=4)
    print(report)

    scores_names, scores = ["accuracy", "precision", "recall", "f1", "auc"], \
                           [accuracy_score(y_true, y_pred),
                            precision_score(y_true, y_pred),
                            recall_score(y_true, y_pred),
                            f1_score(y_true, y_pred),
                            auc(y_true, y_pred) ]
    fig = plt.figure()
    plt.title('ROC')
    x = np.arange(len(scores))
    plt.bar(x, height=scores)
    plt.xticks(x, scores_names)
    plt.legend(loc='lower right')
    plt.ylim([0, 1])
    plt.ylabel('score')
    plt.xlabel('measures')
    fig.savefig(imgs_path + '/classification_report.png', dpi=fig.dpi)
    return scores_names, scores

def printConfusionMatrix(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"\tTN:\t{tn}\tFP:\t{fp}\n\tFN:\t{fn}\tTP:\t{tp}\t")
    return tn, fp, fn, tp

def featureImportance(X, y, featureNames):
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    featuresOrder = list(np.argsort(importances)[::-1])
    importances = importances[featuresOrder]

    for i, featureIndex in enumerate(featuresOrder):
        featuresOrder[i] = featureNames[featureIndex]

    # plots feature importances
    #images.plotBarChart(x=featuresOrder, y=importances, title="Feature importances", textSize=8)
    #images.plotBarChart(x=featuresOrder[:20], y=importances[:20], title="Feature importances (top 20)", textSize=8)

    return featuresOrder, importances

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