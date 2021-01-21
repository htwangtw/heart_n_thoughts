from pathlib import Path

import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# execute from the top of dir
data_dir = Path("data/derivatives/nback_derivatives")
results_dir = Path("results/")

probes = pd.read_csv(data_dir / "task-nbackmindwandering_probes.tsv", sep="\t")
mask = probes['participant_id'].str.contains(r'CON', na=True)
controls = probes[mask]
asd = probes[~mask]

# I want probe names in this order
probe_names = ['Focus', 'Future','Past', 'Self', 'Other', 'Emotion',
    'Images', 'Words', 'Evolving', 'Deliberate', 'Detailed','Habit', 'Vivid']

fig, axes = plt.subplots(nrows=1, ncols=2,
                         figsize=(5, 6))
for i, (name, df) in enumerate(zip(["controls", "patients"],
                                   [controls, asd])):
    p = zscore(df[probe_names].values)
    pca = PCA(n_components=4)
    res = pca.fit(p)
    f = axes[i].matshow(res.components_.T)

    axes[i].set_title(name)
    axes[i].set_xlabel("Components")
    axes[i].set_xticklabels(range(5))
    axes[i].set_yticks(range(len(probe_names)))
    axes[i].set_yticklabels(probe_names)

    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(f, cax=cax, orientation='vertical')

fig.tight_layout()
fig.savefig(results_dir / "basic_pca/pca_control-vs-patients.png", dpi=300)

# PCA on full dataset
p = zscore(probes[probe_names].values)
pca = PCA(n_components=4)
res = pca.fit(p)

plt.matshow(res.components_.T)
plt.yticks(range(len(probe_names)), probe_names)
plt.title("full sample")
plt.savefig(results_dir / "basic_pca/pca_full-sample.png", dpi=300)