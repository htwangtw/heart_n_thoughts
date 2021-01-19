from pathlib import Path

import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# execute from the top of dir
data_dir = Path("data/derivatives/nback_derivatives")
results_dir = Path("results/")
print(str(data_dir / "task-nbackmindwandering_probes.tsv"))

probes = pd.read_csv(data_dir / "task-nbackmindwandering_probes.tsv", sep="\t")

# I want probe names in this order
probe_names = ['Focus', 'Future','Past', 'Self', 'Other', 'Emotion',
    'Images', 'Words', 'Evolving', 'Deliberate', 'Detailed','Habit', 'Vivid']

# PCA on full dataset
p = zscore(probes[probe_names].values)
pca = PCA(n_components=4)
res = pca.fit(p)

plt.matshow(res.components_.T)
plt.yticks(range(len(probe_names)), probe_names)
plt.title("full sample")
plt.savefig(results_dir / "pca_full-sample.png", dpi=300)

