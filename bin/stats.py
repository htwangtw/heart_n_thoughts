import pandas as pd
from pathlib import Path


data = pd.read_csv(Path("results/task-nback_summary.tsv"),
    sep="\t")

# how similar are control's thought to ASD's, vice versa

# for PCA on full sample, is there a component that separates control from patients

#