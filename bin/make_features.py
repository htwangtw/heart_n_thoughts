from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from heart_n_thoughts import dataset
from heart_n_thoughts.figures import subplot_pca
from heart_n_thoughts.utils import insert_groups


OUT_DIR = "results/"
PROBES_FILE = "task-nbackmindwandering_probes.tsv"
BEH_FILE = "task-nbackmindwandering_performance.tsv"
GROUP_DICT = {"sub-CONADIE": "control", "sub-ADIE": "patient"}


def flatten_conditions(performance):
    """
    one subject per row
    """
    merge = []
    for label in performance.nBack.unique():
        label_switch = {"acc": f"nback_{label}_acc", "rt": f"nback_{label}_rt"}
        mask = performance["nBack"].str.contains(label, na=True)
        df = performance[mask].rename(columns=label_switch)
        merge.append(df[mask].set_index(["participant_id", "ses"]))
    perf_flat = pd.concat(merge, axis=1)
    perf_flat = perf_flat.loc[:, ~perf_flat.columns.duplicated()]
    return perf_flat


def main():
    """
    do three kind of PCA on experience sampling data

    - full sample (baseline session)
    - patients
    - controls

    and then create  principle component scores
    """
    results_dir = Path(OUT_DIR)
    # load data
    probes = dataset.get_probes(PROBES_FILE)
    probes = insert_groups(probes, GROUP_DICT)
    feature_scores = []
    # plotting
    # centre heat map to 0
    colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(5, 9))
    pca_labels = []
    for i, name in enumerate(["full", "control", "patient"]):
        df = dataset.sep_adie_group(probes, name)
        if name == "control":  # filp some PCs for easy comparisons
            pattern, scores, exp_var = dataset.cal_scores(
                df, name, modifies=[-1, 1, -1, -1]
            )
        else:
            pattern, scores, exp_var = dataset.cal_scores(df, name)
        pattern.to_csv(
            results_dir / "basic_pca" / f"pc_loading_{name}.tsv", sep="\t"
        )
        feature_scores.append(scores)
        pca_labels += scores.columns.tolist()
        subplot_pca(fig, axes, pattern, exp_var, i, name)

    # save figures
    fig.tight_layout()
    fig.savefig(results_dir / "basic_pca/pca_control-vs-patients.png", dpi=300)

    # save pca scores
    pca = dataset.save_pca(probes, feature_scores)
    pca = pca.rename(
        columns={"rt": "probe_rt"}
    )  # give more details in the header

    # summaride PCA
    pca_val = ["probe_rt"] + pca_labels
    pca_idx = ["participant_id", "ses", "nBack"]

    pca_cond = pca.pivot_table(pca_val, pca_idx).reset_index()
    pca_all = pca.pivot_table(pca_val, pca_idx[:2])
    merge = [pca_all]
    for label in pca_cond.nBack.unique():
        df = pca_cond[pca_cond.nBack == label]
        df.columns = pca_idx + [f"nback_{label}_{v}" for v in df.columns[3:]]
        merge.append(df.set_index(["participant_id", "ses"]))
    pca_flat = pd.concat(merge, axis=1)

    performance = dataset.parse_taskperform(BEH_FILE)
    performance = performance[
        performance["ses"] == "baseline"
    ]  # select base line only
    perf_flat = flatten_conditions(performance)

    # combine performance and pca
    summary = pd.concat([perf_flat, pca_flat], axis=1).reset_index()
    summary = summary.loc[:, ~summary.columns.duplicated()]
    summary = summary.drop(["nBack"], axis=1)
    summary.to_csv(f"results/task-nback_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
