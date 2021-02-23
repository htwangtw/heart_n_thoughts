import pytest
import numpy as np
import pandas as pd

from heart_n_thoughts.dataset import *
from heart_n_thoughts.utils import insert_groups
from heart_n_thoughts.tests import get_test_data_path

test_data_dir = get_test_data_path()

def test_parse_taskperform():
    df = parse_taskperform(test_data_dir / "test_performance.tsv")
    headers = ["rt", "acc", "nBack"]
    for h in headers:
        assert h in df.columns.tolist()

def test_get_probes():
    bl = get_probes(test_data_dir / "test_probes.tsv")
    assert bl["ses"].unique() == "baseline"

def test_cal_scores():
    test_probes = pd.read_csv(test_data_dir / "test_probes.tsv", sep="\t")
    pattern, scores, evr = cal_scores(test_probes, "abc")
    assert len(evr) == 13
    assert scores.shape == (test_probes.shape[0], 4)
    assert pattern.shape == (13, 4)
    assert np.isnan(pattern.values).sum() == 0
    assert np.isnan(scores.values).sum() == 0

def test_save_pca():
    test_probes = pd.read_csv(test_data_dir / "test_probes.tsv", sep="\t")
    _, scores, _ = cal_scores(test_probes, "abc")
    feature_scores = [scores] * 3
    summary = save_pca(test_probes, feature_scores)
    headers = ["abc_factor01", "participant_id", "nBack"]
    for h in headers:
        assert h in summary.columns.tolist()

def test_sep_adie_group():
    groups_dict = {"sub-CON": "control",
               "sub-ADIE": "patient"}
    df = pd.read_csv(test_data_dir / "test_probes.tsv",
                        sep="\t", index_col=0)
    df = insert_groups(df, groups_dict)
    df_n = sep_adie_group(df)
    assert len(df_n["groups"].unique()) == 2
    df_c = sep_adie_group(df, "control")
    assert len(df_c["groups"].unique()) == 1
    df_p = sep_adie_group(df, "patient")
    assert len(df_p["groups"].unique()) == 1
    with pytest.raises(ValueError):
        sep_adie_group(df, "abc")
