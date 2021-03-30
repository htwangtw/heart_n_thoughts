from pathlib import Path

import pandas as pd

from scipy.stats import zscore
from sklearn.decomposition import PCA


from heart_n_thoughts.utils import insert_groups


# execute from the top of dir
data_dir = Path("data/")
behdata_dir = data_dir / "derivatives/nback_derivatives/"
results_dir = Path("results/")

# I want probe names in this order
probe_names = [
    "Focus",
    "Future",
    "Past",
    "Self",
    "Other",
    "Emotion",
    "Images",
    "Words",
    "Evolving",
    "Deliberate",
    "Detailed",
    "Habit",
    "Vivid",
]

group_dict = {"sub-CONADIE": "control", "sub-ADIE": "patient"}


def parse_taskperform(in_file):
    """calculate accuracy and reaction time"""
    df = pd.read_csv(behdata_dir / in_file, sep="\t", index_col=0)
    df = insert_groups(df, group_dict)
    df = df.reset_index()
    files = []
    for type_name in ["acc", "respRT"]:
        d = df.melt(
            id_vars=["participant_id", "groups", "ses", "nBack"],
            value_vars=[type_name],
        )
        d = d.drop(["variable"], axis=1)
        d = d.rename(columns={"value": type_name})
        files.append(d)

    performance = files[0]
    performance["rt"] = files[1]["respRT"]
    return performance


def get_probes(in_file):
    probes = pd.read_csv(behdata_dir / in_file, sep="\t")
    baseline = probes["ses"] == "baseline"
    return probes[baseline]


def save_pca(df, feature_scores):
    perf = df.loc[:, ["probe_index", "participant_id", "ses", "nBack"]]
    perf.loc[:, "rt"] = df.loc[:, "stimEnd"] - df.loc[:, "stimStart"]

    master = pd.concat([perf] + feature_scores, axis=1)
    master = master.fillna("n/a")
    return master


def cal_scores(df, name, modifies=None):
    """
    run scikit learn PCA
    calculate principle componet scores and corret patterns.
    Save output
    """
    z_probes = df[probe_names].apply(zscore)
    pca = PCA()
    res = pca.fit(z_probes)

    pattern = res.components_.T[:, :4]
    if type(modifies) is list:
        pattern *= modifies

    # project scores
    scores = z_probes.dot(pattern)
    scores.index = df.index
    scores.columns = [f"{name}_factor{x:02d}" for x in range(1, 5)]

    pattern = pd.DataFrame(pattern, columns=range(1, 5), index=probe_names)
    return pattern, scores, res.explained_variance_ratio_


def sep_adie_group(df, name=None):
    """get control or patiet group"""
    if "groups" not in df.columns:
        raise KeyError(f"'groups' not in columns")
    mask = df["groups"] == "control"
    if name == "control":
        return df[mask]
    elif name == "patient":
        return df[~mask]
    elif name is None or name == "full":
        return df
    else:
        raise ValueError(f"group {name} not presented in data")
