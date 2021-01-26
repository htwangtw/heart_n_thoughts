from pathlib import Path
from pickle import dump

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn.palettes import dark_palette

from src.util import insert_groups


# execute from the top of dir
data_dir = Path("data/derivatives/nback_derivatives")
results_dir = Path("results/")
df = pd.read_csv(data_dir / "task-nbackmindwandering_performance.tsv",
    sep="\t", index_col=0)
df = insert_groups(df)
# mask = df['ses'].str.contains(r'base', na=True)

df = df.reset_index()
files = []
for type_name in ["acc", "respRT"]:
    d = df.melt(id_vars=["participant_id", "groups", "ses", "nBack"],
        value_vars=[type_name])
    d = d.drop(["variable"], axis=1)
    d = d.rename(columns={"value": type_name})
    files.append(d)

performance = files[0]
performance["rt"] = files[1]["respRT"]

performance.to_csv(results_dir / "task_performance.tsv",
    index=False, sep="\t")