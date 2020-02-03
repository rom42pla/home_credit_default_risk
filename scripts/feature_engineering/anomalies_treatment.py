import numpy as np

def remove_ids(df):
    for col in df.columns:
        if "sk_id" in col.lower().strip() and col in df.columns:
            df = df.drop(columns=[col])
    return df

def remove_infs(df):
    df = df.replace([np.inf, -np.inf, "inf", "inf"], np.nan)
    return df