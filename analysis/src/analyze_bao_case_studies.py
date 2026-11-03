from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "bao_novelty_complexity.csv"
)

OUTPUT_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "bao_case_studies.csv"
)


def assign_novelty_groups(df):
    df = df.copy()

    q33 = df["novelty"].quantile(1 / 3)
    q67 = df["novelty"].quantile(2 / 3)

    def classify(value):
        if value <= q33:
            return "Low"
        elif value <= q67:
            return "Medium"
        else:
            return "High"

    df["novelty_group"] = (
        df["novelty"]
        .apply(classify)
    )

    return df


def pick_case(df, group, mode):
    subset = df[
        df["novelty_group"] == group
    ].copy()

    if len(subset) == 0:
        return None

    if mode == "best":
        idx = subset[
            "bao_vs_pg_ratio"
        ].idxmin()

    elif mode == "worst":
        idx = subset[
            "bao_vs_pg_ratio"
        ].idxmax()

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    return subset.loc[idx]


def main():

    # =========================================================
    # 1. LOAD
    # =========================================================

    df = pd.read_csv(
        INPUT_FILE
    )

    valid = df[
        (df["bao_status"] == "OK")
        & (df["pg_status"] == "OK")
        & df["novelty"].notna()
        & df["bao_vs_pg_ratio"].notna()
    ].copy()

    print("=== INPUT ===")
    print(f"Total rows      : {len(df)}")
    print(f"Comparable rows : {len(valid)}")

    # =========================================================
    # 2. CHỌN CASE STUDY
    # =========================================================

    split_order = [
        "LOO",
        "Random",
        "BaseQuery",
    ]

    cases = []

    for split in split_order:

        split_df = valid[
            valid["split_type"] == split
        ].copy()

        split_df = assign_novelty_groups(
            split_df
        )

        # 4 archetypes
        selections = [
            ("Low", "best"),
            ("Low", "worst"),
            ("High", "best"),
            ("High", "worst"),
        ]

        for group, mode in selections:

            row = pick_case(
                split_df,
                group,
                mode,
            )

            if row is None:
                continue

            case = row.copy()

            case["case_type"] = (
                f"{group}_{mode}"
            )

            cases.append(case)

    result = pd.DataFrame(
        cases
    )

    # =========================================================
    # 3. CHỈ GIỮ CỘT CẦN CHO CASE STUDY
    # =========================================================

    columns = [
        "split_type",
        "query_id",
        "family",
        "case_type",
        "novelty",
        "novelty_group",

        "nearest_train_query",
        "nearest_train_family",

        "d_table",
        "d_join",
        "d_filter",

        "n_tables",
        "n_joins",
        "n_filters",

        "bao_end_to_end_ms",
        "pg_end_to_end_ms",
        "bao_vs_pg_ratio",
        "winner",
    ]

    result = result[
        columns
    ]

    # =========================================================
    # 4. PRINT
    # =========================================================

    print("\n=== CASE STUDIES ===")

    for split in split_order:

        print(f"\n[{split}]")

        subset = result[
            result["split_type"] == split
        ]

        for _, row in subset.iterrows():

            print(
                f"{row['case_type']:10s} "
                f"| {row['query_id']:5s} "
                f"| novelty={row['novelty']:.6f} "
                f"| ratio={row['bao_vs_pg_ratio']:.3f} "
                f"| winner={row['winner']} "
                f"| tables={int(row['n_tables'])} "
                f"| joins={int(row['n_joins'])} "
                f"| filters={int(row['n_filters'])} "
                f"| nearest={row['nearest_train_query']}"
            )

    # =========================================================
    # 5. SAVE
    # =========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n=== OUTPUT ===")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\n=== RESEARCH NOTE ===")

    print(
        "Low_best vs Low_worst kiểm tra việc "
        "Novelty thấp có đảm bảo Bao tốt hay không."
    )

    print(
        "High_best vs High_worst kiểm tra việc "
        "Novelty cao có luôn làm Bao kém hay không."
    )

    print(
        "Nếu trong cùng Novelty group performance "
        "khác mạnh, Novelty không phải yếu tố duy nhất."
    )


if __name__ == "__main__":
    main()