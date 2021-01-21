from pathlib import Path

import pandas as pd

from src.util import insert_groups


pheno_dir = Path("data/phenotype/")
output_dir = Path("results")

df = pd.concat([pd.read_csv(p, sep="\t", index_col=0) \
    for p in pheno_dir.glob("*.tsv")],
    axis=0)

# get columns with less than 30% missing
# (common between patients and controls)
keep = ["Intervention"] + \
    df.columns[df.isnull().mean() < 0.3].tolist()
data = df[keep].fillna("n/a")

data = insert_groups(data)
data.to_csv(output_dir / "_phenotype.tsv", sep="\t")