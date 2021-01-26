import pandas as pd
from pathlib import Path

path_perf = Path("results/task_performance.tsv")
path_pca = Path("results/basic_pca/pca_control-vs-patients.tsv")

perf = pd.read_csv(path_perf, sep="\t")
pca = pd.read_csv(path_pca, sep="\t")
pca = pca.rename(columns={"rt": "probe_rt"})

merge = []
for label in perf.nBack.unique():
    mask = perf['nBack'].str.contains(label, na=True)
    df = perf[mask].rename(columns={
        "acc": f"nback_{label}_acc",
        "rt": f"nback_{label}_rt"
        })
    merge.append(df[mask].set_index(["participant_id", "ses"]))
perf_flat = pd.concat(merge,axis=1)
perf_flat = perf_flat.loc[:,~perf_flat.columns.duplicated()]

pca_val = ['probe_rt', 'full_factor_1',
    'full_factor_2', 'full_factor_3', 'full_factor_4', 'controls_factor_1',
    'controls_factor_2', 'controls_factor_3', 'controls_factor_4',
    'patients_factor_1', 'patients_factor_2', 'patients_factor_3',
    'patients_factor_4']
pca_idx = ['participant_id', 'ses', 'nBack']

pca_cond = pca.pivot_table(pca_val, pca_idx).reset_index()
pca_all = pca.pivot_table(pca_val, pca_idx[:2])
merge = [pca_all]
for label in pca_cond.nBack.unique():
    df = pca_cond[pca_cond.nBack == label]
    df.columns = pca_idx + [f"nback_{label}_{v}" for v in df.columns[3:]]
    merge.append(df.set_index(["participant_id", "ses"]))
pca_flat = pd.concat(merge, axis=1)

summary = pd.concat([perf_flat, pca_flat], axis=1).reset_index()
summary = summary.loc[:,~summary.columns.duplicated()]
summary = summary.drop(["nBack"], axis=1)
summary.to_csv(f"results/task-nback_summary.tsv",
        sep="\t", index=False)


# for ses in summary.ses.unique():
#     df = summary[summary.ses == ses]
#     df.to_csv(f"results/ses-{ses}_task-nback_summary.tsv",
#         sep="\t", index=False)