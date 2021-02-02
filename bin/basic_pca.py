from pathlib import Path
import pickle

import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors


# execute from the top of dir
data_dir = Path("data/derivatives/nback_derivatives")
results_dir = Path("results/")

# load data
probes = pd.read_csv(data_dir / "task-nbackmindwandering_probes.tsv", sep="\t")
baseline = probes["ses"] == "baseline"
probes_forPCA = probes[baseline]
mask = probes['participant_id'].str.contains(r'CON', na=True)
controls = probes[mask]
asd = probes[~mask]

# I want probe names in this order
probe_names = ['Focus', 'Future','Past', 'Self', 'Other', 'Emotion',
    'Images', 'Words', 'Evolving', 'Deliberate', 'Detailed','Habit', 'Vivid']


def subplot_pca(fig, axes, selected, scree, i, name):
    f = axes[i, 0].matshow(selected, cmap="RdBu_r")
    axes[i, 0].set_title(name)
    axes[i, 0].set_xlabel("Components")
    axes[i, 0].set_xticklabels(range(5))
    axes[i, 0].set_yticks(range(len(probe_names)))
    axes[i, 0].set_yticklabels(probe_names)

    divider = make_axes_locatable(axes[i, 0])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(f, cax=cax, orientation='vertical')

    axes[i, 1].plot(scree, "-o")
    axes[i, 1].set_xlabel("Components")
    axes[i, 1].set_xticks(range(len(probe_names)))
    axes[i, 1].set_xticklabels(range(1, 14))

# centre heat map to 0
colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1)

fig, axes = plt.subplots(nrows=3, ncols=2,
                         figsize=(5, 9))
pc_patterns = {}
pc_scores = []
# PCA on full dataset - see if there's a PC to separate two groups
z_probes = zscore(probes_forPCA[probe_names].values)
pca = PCA()
res = pca.fit(z_probes)
scree = res.explained_variance_ratio_
selected = res.components_.T[:, :4]

save = probes[['probe_index', 'participant_id',
    'ses', 'nBack']]
save["rt"] = probes['stimEnd'] - probes['stimStart']

# project
full_probes = zscore(probes[probe_names].values)
scores = pd.DataFrame(full_probes.dot(selected), index=probes.index,
    columns=[f"full_factor_{x}" for x  in range(1, 5)])

pc_patterns["Full sample"] = selected
pc_scores.append(pd.concat([save, scores], axis=1))

# dump figure
subplot_pca(fig, axes, selected, scree, -1, "Full sample")

# PCA on separate sample
for i, (name, df) in enumerate(zip(["controls", "patients"],
                                   [controls, asd])):
    # PCA computed on group; z score within group
    p = zscore(df[probe_names].values)
    pca = PCA()
    res = pca.fit(p)
    scree = res.explained_variance_ratio_
    selected = res.components_.T[:, :4]

    if i == 0:  #control
        # reverse component 1, 3, 4
        selected *= [-1, 1, -1, -1]

    # project on all probes regardless of group
    scores = pd.DataFrame(full_probes.dot(selected), index=probes.index,
        columns=[f"{name}_factor_{x}" for x  in range(1, 5)])

    pc_patterns[name] = selected
    pc_scores.append(scores)

    # plotting
    subplot_pca(fig, axes, selected, scree, i, name)

fig.tight_layout()
fig.savefig(results_dir / "basic_pca/pca_control-vs-patients.png", dpi=300)

master = pd.concat(pc_scores, axis=1)
master = master.fillna("n/a")
master.to_csv(
    results_dir / "basic_pca/pca_control-vs-patients.tsv",
    sep="\t", index=False, )

# save pattern
with open(results_dir / "basic_pca/pca_control-vs-patients.pkl", "wb") as f:
    pickle.dump(pc_patterns, f)

# def load_obj(name ):
#     with open('obj/' + name + '.pkl', 'rb') as f:
#         return pickle.load(f)