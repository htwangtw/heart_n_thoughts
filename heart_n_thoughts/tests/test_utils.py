import pytest

import pandas as pd

from heart_n_thoughts.utils import (
    _check_subject_index,
    _look_up_group,
    insert_groups,
)
from heart_n_thoughts.tests import get_test_data_path

test_data_dir = get_test_data_path()
df_noidx = pd.read_csv(test_data_dir / "test_probes.tsv", sep="\t")
df_idx = pd.read_csv(test_data_dir / "test_probes.tsv", sep="\t", index_col=0)

groups_dict = {"sub-CON": "control", "sub-ADIE": "patient"}


def test_check_subject_index():
    for df in [df_idx, df_noidx]:
        sub_list = _check_subject_index(df)
        assert "sub-" in sub_list[0]


def test_look_up_group():
    with pytest.raises(ValueError):
        _look_up_group("sub-01", groups_dict)

    assert _look_up_group("sub-CONADIE999", groups_dict) == "control"
    assert _look_up_group("sub-ADIE999", groups_dict) == "patient"


def test_insert_groups():
    for df in [df_idx, df_noidx]:
        df = insert_groups(df, groups_dict)
        assert "groups" in df.columns.tolist()
        assert len(df["groups"].unique()) == 2
