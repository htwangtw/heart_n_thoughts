import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

import seaborn as sns
import matplotlib.pyplot as plt

from heart_n_thoughts.utils import insert_groups
from heart_n_thoughts.dataset import get_probes, sep_adie_group
from heart_n_thoughts import dataset

PROBES_FILE = "task-nbackmindwandering_probes.tsv"
PATH_CONTROLS = "data/phenotype/adie-questionnaires_controls.tsv"
PATH_PATIENTS = "data/phenotype/adie-questionnaires_patients.tsv"
GROUP_DICT = {"sub-CONADIE": "control", "sub-ADIE": "patient"}


def load_probes():
    # average per subject
    probes = get_probes(PROBES_FILE)
    probes = probes[probes["ses"] == "baseline"]
    pca_labels = probes.columns.tolist()[-13:]
    probes = probes.pivot_table(index="participant_id", values=pca_labels)
    probes = insert_groups(probes, GROUP_DICT)
    return probes


def load_assessments():
    # get common measures (baseline session)
    patients = pd.read_csv(PATH_PATIENTS, sep="\t")
    controls = pd.read_csv(PATH_CONTROLS, sep="\t")
    data = pd.concat([controls, patients], axis=0, join="inner")
    data = insert_groups(data, GROUP_DICT)
    # remove pointless headers
    data.columns = [
        c.replace("BL_", "") if "BL_" in c else c for c in data.columns
    ]
    return data


def impute_group(rawdata, col_names, strategy):
    # impute per group
    dfs = []
    for name in ["patient", "control"]:
        df = sep_adie_group(rawdata, name)
        imp_mean = SimpleImputer(missing_values=np.nan, strategy=strategy)
        tmp = imp_mean.fit_transform(df.loc[:, col_names])
        df.loc[:, col_names] = tmp
        dfs.append(df)
    return pd.concat(dfs, axis=0, join="inner")


def cv_silhouette_scorer(estimator, X):
    estimator.fit(X)
    preprocessed_data = estimator["pca"].transform(X)
    cluster_labels = estimator["kmeans"].labels_
    num_labels = len(set(cluster_labels))
    num_samples = X.shape[0]
    if num_labels in [1, num_samples]:
        return -1
    else:
        return silhouette_score(preprocessed_data, cluster_labels)


def gen_sklables(data):
    # build k mean pipeline
    true_label_names = data["groups"].tolist()
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(true_label_names)


def build_clf(X):
    pipe = Pipeline(
        steps=[
            ("standardise", StandardScaler()),
            ("pca", PCA(whiten=True, random_state=42)),
            (
                "kmeans",
                KMeans(
                    init="k-means++", n_init=50, max_iter=500, random_state=42
                ),
            ),
        ]
    )

    param_grid = {
        "pca__n_components": range(2, X.shape[1]),
        "kmeans__n_clusters": range(2, X.shape[1]),
    }
    cv = [(slice(None), slice(None))]
    return GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        refit=True,
        scoring=cv_silhouette_scorer,
        cv=cv,
        n_jobs=5,
    )


def cv_grid(cv_results):
    results = pd.DataFrame.from_dict(cv_results)
    results["params_str"] = results.params.apply(str)
    return results.pivot(
        index="param_pca__n_components",
        columns="param_kmeans__n_clusters",
        values="mean_test_score",
    )


def clf_subset(data, col_names):
    X = data.loc[:, col_names].values

    gs = build_clf(X)
    gs.fit(X)
    scores_matrix = cv_grid(gs.cv_results_)
    kmean_labels = gs.best_estimator_.steps[2][1].labels_
    pca_comp = gs.best_estimator_.steps[1][1].components_.T
    pca_comp = pd.DataFrame(
        pca_comp, columns=range(1, pca_comp.shape[1] + 1), index=col_names
    )
    pca_scores = gs.best_estimator_.steps[1][1].fit_transform(X)

    print(gs.best_params_)

    plt.imshow(scores_matrix)
    plt.xlabel("K-mean: number of clusters")
    plt.xticks(range(scores_matrix.shape[1]), scores_matrix.columns.tolist())
    plt.ylabel("PCA: number of components")
    plt.yticks(range(scores_matrix.shape[0]), scores_matrix.index.tolist())
    plt.show()

    sns.heatmap(
        pca_comp,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.show()
    return kmean_labels, pca_scores


trait_col = [
    "BPQ",
    "TAS_total",
    "AQ",
    "EQ",
    "STAI_S",
    "STAI_T",
    "GAD7",
    "PANAS_positive",
    "PANAS_negative",
    "PHQ9",
    "UCLA_upset",
    "GSQ_TOTAL",
]

maia_col = [
    "MAIA_Noticing",
    "MAIA_NotDistracting",
    "MAIA_NotWorrying",
    "MAIA_AttentionRegulation",
    "MAIA_EmotionalAwareness",
    "MAIA_SelfRegulation",
    "MAIA_BodyListening",
    "MAIA_Trusting",
]

exp = [
    "Deliberate",
    "Detailed",
    "Emotion",
    "Evolving",
    "Focus",
    "Future",
    "Habit",
    "Images",
    "Other",
    "Past",
    "Self",
    "Vivid",
    "Words",
]


# impute all data
rawdata = load_assessments()
col_names = trait_col + maia_col
data = impute_group(rawdata, col_names, "most_frequent")
probes = load_probes()
probes = impute_group(probes, exp, "most_frequent")

# fit model
true_labels = gen_sklables(data)
# trait_kmean_labels, trait_pca_scores = clf_subset(data, trait_col)
# plt.close()
# maia_kmean_labels, maia_pca_scores = clf_subset(data, maia_col)
# plt.close()
exp_kmean_labels, exp_pca_scores = clf_subset(probes, exp)
plt.close()
