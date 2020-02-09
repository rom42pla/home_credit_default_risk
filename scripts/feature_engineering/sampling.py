import pandas as pd
import evaluation
from Timer import Timer

from feature_engineering import nan_treatment

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

def undersample(df, complex=False, log=False):
    if log:
        section_timer = Timer(log=f"undersampling")
    if "_merge" in df.columns:
        df = df.drop("_merge", axis=1)

    count = evaluation.count_values(df, "TARGET")
    lessLabel = 0 if count[0] < count[1] else 1
    df_new = pd.concat(
        [df[df["TARGET"] == 0].sample(count[lessLabel]), df[df["TARGET"] == 1].sample(count[lessLabel])])

    if log:
        section_timer.end_timer(log=f"done with a final shape of {df_new.shape}")
    return df_new

def smote(df, log=False):
    if log:
        section_timer = Timer(log=f"oversampling using SMOTE")
    if "_merge" in df.columns:
        df = df.drop("_merge", axis=1)

    target_values = pd.unique(df["TARGET"]).tolist()
    target_values.sort()
    false_number, true_number = target_values
    df = df.replace(to_replace={false_number: 0, true_number: 1})

    df, df["TARGET"] = SMOTE(n_jobs=4).fit_resample(df.drop(columns="TARGET"), df["TARGET"])

    if log:
        section_timer .end_timer(log=f"for a total shape of {df.shape}")

    return df