

def remove_ids(df):
    for col in df.columns:
        if "sk_id" in col.lower().strip() and col in df.columns:
            df = df.drop(columns=[col])
    return df