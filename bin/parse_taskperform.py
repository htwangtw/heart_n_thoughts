from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.util import insert_groups


# execute from the top of dir
data_dir = Path("data/derivatives/nback_derivatives")
results_dir = Path("results/")
df = pd.read_csv(data_dir / "task-nbackmindwandering_performance.tsv",
    sep="\t", index_col=0)
df = insert_groups(df)
mask = df['ses'].str.contains(r'base', na=True)


df = df[mask].reset_index()
df["eff"] = df["acc"] / df["respRT"]
eff = df.melt(id_vars=["participant_id", "groups", "ses", "nBack"],
    value_vars=["eff"])
acc = df.melt(id_vars=["participant_id", "groups", "ses", "nBack"],
    value_vars=["acc"])
rt = df.melt(id_vars=["participant_id", "groups", "ses", "nBack"],
    value_vars=["respRT"])

sns.violinplot(data=acc, x="nBack", y="value", hue="groups",
               split=True)
sns.despine(left=True)
plt.show()