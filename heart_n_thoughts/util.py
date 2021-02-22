import pandas as pd


def sep_group(df, name):
    mask = df['participant_id'].str.contains(r'CON', na=True)
    if name == "control":
        return df[mask]
    elif name == "asd":
        return df[~mask]
    else:
        return df

def insert_groups(df):
    """
    check subject ID (index of dataframe)
    determing if its a control or patient

    input
    -----
    df: pandas.DataFame
        data frame with ADIE project subject id as index
    """
    groups = []
    for sub in df.index:
        if "CON" in sub:
            groups.append("control")
        else:
            groups.append("patient")
    df.insert(loc=0, column='groups', value=groups)
    return df