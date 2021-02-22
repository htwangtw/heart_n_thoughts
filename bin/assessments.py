import numpy as np
import pandas as pd
from pandas.core import base

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import seaborn as sns
import matplotlib.pyplot as plt


patients = pd.read_csv("data/phenotype/adie-questionnaires_patients.tsv",
                       sep="\t")
controls = pd.read_csv("data/phenotype/adie-questionnaires_controls.tsv",
                      sep="\t")
common = pd.concat([controls, patients], axis=0, join="inner").columns

dfs = []
for df in [patients, controls]:
    # remove pointless header
    df = df[common]
    df.columns = [c.replace("BL_", "") if "BL_" in c else c
                        for c in df.columns]

    # impute per group
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    tmp = imp_mean.fit_transform(df.loc[:, "BPQ":])
    df.loc[:, "BPQ":] = tmp
    dfs.append(df)

data = pd.concat(dfs, axis=0, join="inner")
# # preprocessing
# imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
# X = baseline.loc[:, "BPQ":]
# tmp = imp_mean.fit_transform(baseline)
# scaler = StandardScaler()
# scaler.fit(tmp)
# baseline_z = scaler.transform(tmp)

# off-diagnal of correlation matrix
corr = data.iloc[:, 4:].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

# explore variables with heatmap
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
# # let's explore AQ and EQ
# data = baseline[["AQ", "EQ"]]
# imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
# X = imp_mean.fit_transform(data)
# scaler = StandardScaler()
# scaler.fit(X)
# Xz = scaler.transform(X)

# estimators = [('k_means_8', KMeans(n_clusters=8)),
#               ('k_means_3', KMeans(n_clusters=3))]

# for i, (name, est) in enumerate(estimators):
#     est.fit(Xz)
#     labels = est.labels_
#     fig = plt.figure(i + 1, figsize=(4, 3))
#     plt.scatter(Xz[:, 0], Xz[:, 1], c=labels)


