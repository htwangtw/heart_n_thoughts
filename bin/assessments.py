from itertools import product
import numpy as np
import pandas as pd
from pandas.core import base

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
from sklearn.utils import shuffle

from heart_n_thoughts.utils import insert_groups
from heart_n_thoughts.dataset import sep_adie_group


patients = pd.read_csv("data/phenotype/adie-questionnaires_patients.tsv",
                       sep="\t")
controls = pd.read_csv("data/phenotype/adie-questionnaires_controls.tsv",
                      sep="\t")

col_names = ['BPQ', 'TAS_total', 'AQ',
       'EQ', 'STAI_S', 'STAI_T', 'GAD7', 'PANAS_positive', 'PANAS_negative',
       'PHQ9', 'UCLA_LS', 'UCLA_upset', 'GSQ_TOTAL', 'MAIA_Noticing',
       'MAIA_NotDistracting', 'MAIA_NotWorrying', 'MAIA_AttentionRegulation',
       'MAIA_EmotionalAwareness', 'MAIA_SelfRegulation', 'MAIA_BodyListening',
       'MAIA_Trusting']

maia_headers = ['MAIA_NotDistracting', 'MAIA_NotWorrying', 'MAIA_AttentionRegulation',
                'MAIA_EmotionalAwareness', 'MAIA_SelfRegulation', 'MAIA_BodyListening',
                'MAIA_Trusting']

col_names_maia = ['BPQ', 'TAS_total', 'AQ',
       'EQ', 'STAI_S', 'STAI_T', 'GAD7', 'PANAS_positive', 'PANAS_negative',
       'PHQ9', 'UCLA_LS', 'UCLA_upset', 'GSQ_TOTAL',
       'MAIA_factor01', 'MAIA_factor02', 'MAIA_factor03']

# get common measures (baseline session)
common = pd.concat([controls, patients], axis=0, join="inner")
common = insert_groups(common, {"sub-CONADIE": "control", "sub-ADIE": "patient"})
# remove pointless headers
common.columns = [c.replace("BL_", "") if "BL_" in c else c for c in common.columns]

# impute per group
dfs = []
for name in ["patient", "control"]:
    df = sep_adie_group(common, name)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    tmp = imp_mean.fit_transform(df.loc[:, col_names])
    df.loc[:, col_names] = tmp
    dfs.append(df)
data = pd.concat(dfs, axis=0, join="inner")

# standardise
scaler = StandardScaler()
data_z = scaler.fit_transform(data.loc[:, col_names])
data.loc[:, col_names] = data_z

# # colapse MAIA score with PCA
# # PCA
# pca = PCA()
# res = pca.fit(data.loc[:, maia_headers])
# plt.plot(res.explained_variance_ratio_, "-o")
# plt.show()
# pattern = pd.DataFrame(res.components_.T, columns=range(1, len(maia_headers) + 1), index=maia_headers)
# sns.heatmap(pattern, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.show()

# get top 3
pca = PCA(n_components=3)
maia_pca = pca.fit_transform(data.loc[:, maia_headers])
for i, pc in enumerate(maia_pca.T):
    data[f"MAIA_factor{i + 1:02d}"] = pc
# pattern = pd.DataFrame(res.components_.T[:, :3], columns=range(1, 4), index=maia_headers)
# sns.heatmap(pattern, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.show()


# off-diagnal of correlation matrix
# corr = data.loc[:, col_names].corr()
corr = data.loc[:, col_names_maia].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

# explore variables with heatmap
# f, ax = plt.subplots(figsize=(11, 9))
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.show()

# PCA
pca = PCA()
res = pca.fit(data.loc[:, col_names_maia])
# res = pca.fit(data.loc[:, col_names])

# plt.plot(res.explained_variance_ratio_, "-o")
# plt.show()
# # pattern = pd.DataFrame(res.components_.T, columns=range(1, len(col_names) + 1), index=col_names)
# pattern = pd.DataFrame(res.components_.T, columns=range(1, len(col_names_maia) + 1), index=col_names_maia)
# sns.heatmap(pattern, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.show()

# build k mean pipeline
true_label_names = data["groups"].tolist()
label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(true_label_names)
n_clusters = len(label_encoder.classes_)


pipe = Pipeline(steps=[
    ("pca", PCA(random_state=42)),
    ("kmeans", KMeans(init="k-means++", n_init=50, max_iter=500, random_state=42))])
param_grid = {
    'pca__n_components': range(2, len(col_names_maia)),
    'kmeans__n_clusters': range(2, len(col_names_maia)),
}
X = data.loc[:, col_names_maia].values

def cv_silhouette_scorer(estimator, X):
    estimator.fit(X)
    preprocessed_data = estimator["pca"].transform(X)
    cluster_labels = estimator["kmeans"].labels_
    num_labels = len(set(cluster_labels))
    num_samples = X.shape[0]
    if num_labels in [1,num_samples]:
        return -1
    else:
        return silhouette_score(preprocessed_data, cluster_labels)

cv = [(slice(None), slice(None))]
gs = GridSearchCV(estimator=pipe, param_grid=param_grid,
                  scoring=cv_silhouette_scorer, cv=cv, n_jobs=5)
gs.fit(X)