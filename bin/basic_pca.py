from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from heart_n_thoughts import dataset
from heart_n_thoughts.figures import subplot_pca
from heart_n_thoughts.util import sep_group


results_dir = Path("results/")
# load data
probes = dataset.get_probes("task-nbackmindwandering_probes.tsv")

feature_scores = []
# plotting
# centre heat map to 0
colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1)

fig, axes = plt.subplots(nrows=3, ncols=2,
                         figsize=(5, 9))

for i, name in enumerate(["full", "control", "asd"]):
    df = sep_group(probes, name)
    if name == "control":
        pattern, scores, exp_var = dataset.cal_scores(df, name, modifies=[-1, 1, -1, -1])
    else:
        pattern, scores, exp_var = dataset.cal_scores(df, name)
    pattern.to_csv(results_dir / "basic_pca" / f"pc_loading_{name}.tsv", sep="\t")
    feature_scores.append(scores)
    subplot_pca(fig, axes,
                pattern,
                exp_var, i, name)

# save figures
fig.tight_layout()
fig.savefig(results_dir / "basic_pca/pca_control-vs-patients.png", dpi=300)

# save pca scores
master = dataset.save_pca(df, feature_scores)
master.to_csv(results_dir / "basic_pca/pca_control-vs-patients.tsv", sep="\t", index=False)