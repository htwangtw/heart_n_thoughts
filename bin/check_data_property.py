import pandas as pd


def main():
    """number of ASD subjects and related session"""
    probes = pd.read_csv(
        "data/derivatives/nback_derivatives/task-nbackmindwandering_probes.tsv",
        sep="\t",
    )

    count_sub = {1: [], 2: [], 3: []}

    for id in probes["participant_id"].unique():
        if "CON" not in id:
            cur_sub = probes["participant_id"] == id
            n = len(probes[cur_sub]["ses"].unique())
            count_sub[n].append(id)

    for k, l in count_sub.items():
        print(f"ASD subject with at most {k} sessions: {len(l)}")


if __name__ == "__main__":
    main()
