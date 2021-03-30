import pandas as pd


def _check_subject_index(df):
    """get the correct subject list"""
    sublist = df.index.tolist()
    sub = str(sublist[0])
    if "sub-" in sub:
        return sublist
    else:
        return df["participant_id"].tolist()


def _look_up_group(sub, groups_dict):
    d = groups_dict.copy()
    if len(d) == 0:
        raise ValueError(
            f"No matching key from groups_dict and {sub}; \
                          check the searching key words"
        )
    key, item = d.popitem()
    if key in sub:
        return item
    else:
        return _look_up_group(sub, d)


def insert_groups(df, groups_dict):
    """
    check subject ID (index of dataframe)
    determing if its a control or patient

    input
    -----
    df: pandas.DataFame
        data frame with project subject id as index
    groups_dict: dict
        dictionary to check sting content for looking up group info
        {"key_in_participant_id": "group_name"}
    """
    groups = []
    sublist = _check_subject_index(df)

    for sub in sublist:
        group_name = _look_up_group(sub, groups_dict)
        groups.append(group_name)
    df.insert(loc=0, column="groups", value=groups)
    return df
