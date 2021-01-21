import json
from pathlib import Path

import pandas as pd

from src.util import insert_groups


pheno_dir = Path("data/phenotype/")
output_dir = Path("results") / "phenotypes"

df = pd.concat([pd.read_csv(p, sep="\t", index_col=0) \
    for p in pheno_dir.glob("*.tsv")],
    axis=0)
df = insert_groups(df)

sessions = {
    "BL": "baseline",
    "F": "oneweek",  # confirmed by lisa
    "3mf": "threemonth",
    "FY": "oneyear"
}

# curated session unrelated labels
demographics = ["Age", "GenderID", "GenderBirth", "GenderFit", "Education", "Handedness"]
height_weight = ["Height", "Weight"]
diagnosis = ["Medication", "PriorDx", "AnxietyDx", "DepressionDx",
    "ADHD", "OCD", "PTSD", "CPTSD", "Dyspraxia","Dyslexia", "EatingDisorder",
    "MINIDx", "MINIASCDx", "MINIYes"]  # need to check what they mean
admin = ["Intervention", "Intero_Completion", "PrimOut", "Compliance", "Dropout", "Site"]

assessments = ["BPQ", "TAS", "AQ", "EQ", "STAI",
    "GAD7", "MAIA", "PANAS", "PHQ9", "UCLA", "GSQ"]  # known assessments
template = {"Descriptions": "fill this in",
            "Levels": {"item1": "description; delete if not applied",
                "item2": "description; delete if not applied"}
}
MeasurementToolMetadata = {
            "Description": "A free text description of the measurement tool",
            "TermURL": "A URL to an entity in an ontology corresponding to this tool"
        }
for name, cat in zip(["demographics", "height_weight", "diagnosis", "admin"],
    [demographics, height_weight, diagnosis, admin]):
    desc = {c: template for c in cat}
    # save files
    with open(output_dir / f"{name}.json", "w") as f:
        json.dump(desc, f, indent=2)
    cur_df = df[cat].fillna("n/a")
    cur_df.to_csv(output_dir / f"{name}.tsv", sep="\t")

# get baseline session data only (prefix BL)
for an in assessments:
    subscales = [c for c in df.columns if an in c]
    if subscales:
        desc = {c.replace("BL_", ""): template
            for c in subscales if "BL_" in c}
        desc["MeasurementToolMetadata"] = MeasurementToolMetadata
        # save json
        with open(output_dir / f"{an}.json", "w") as f:
            json.dump(desc, f, indent=2)

        # todo: split file by session tool_name_ses-label.tsv
        cur_df = df[subscales].fillna("n/a")
        cur_df.to_csv(output_dir / f"{an}.tsv", sep="\t")

# # get columns with less than 30% missing
# # (common between patients and controls)
# keep = ["Intervention"] + \
#     df.columns[df.isnull().mean() < 0.3].tolist()
# data = df[keep].fillna("n/a")

# data.to_csv(output_dir / "_phenotype.tsv", sep="\t")