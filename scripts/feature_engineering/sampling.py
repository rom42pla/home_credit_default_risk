import pandas as pd
import evaluation
from Timer import Timer

from feature_engineering import nan_treatment

def undersample(df, log=False):
    if log:
        section_timer = Timer(
            log=f"undersampling")
    if "_merge" in df.columns:
        df = df.drop("_merge", axis=1)

    count = evaluation.count_values(df, "TARGET")
    lessLabel = 0 if count[0] < count[1] else 1
    df_new = pd.concat([df[df["TARGET"] == 0].sample(count[lessLabel]), df[df["TARGET"] == 1].sample(count[lessLabel])])
    df_discarded = df.drop(df_new.index)

    if log:
        section_timer .end_timer(log=f"discarded {df_discarded.shape[0]} rows for a total shape of {df_new.shape}")
    return df_new, df_discarded