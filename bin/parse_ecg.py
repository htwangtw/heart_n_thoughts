from pathlib import Path

import pandas as pd
import numpy as np


def find_ecg_files(cur_dir):
    ecg_path = list(cur_dir.glob("*physio.tsv.gz"))
    stim_path = list(cur_dir.glob("*stim.tsv.gz"))
    event_path = list(cur_dir.glob("*beh.tsv"))
    paths = {}
    for p in [ecg_path, stim_path, event_path]:
        if not p:
            print("Subject missing files")
            print(cur_dir.parents[1].name)
            continue
        else:
            cp = p[0]
            label = cp.stem.split("_")[-1].split(".")[0]
            paths[label] = str(cp)
    return paths


data_dir = Path("data/")
ecg_dir = data_dir.glob("sub-*/ses-*/beh/*physio.tsv.gz")

for ecg_ses in ecg_dir:
    cur_dir = ecg_ses.parent
    paths = find_ecg_files(cur_dir)

