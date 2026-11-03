from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PERFORMANCE_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "balsa_novelty_performance.csv"
)

GROUP_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "balsa_novelty_group_summary.csv"
)

OUTPUT_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "balsa_case_studies.csv"
)


def assign_group(row, thresholds):

    split_type = row["split_type"]

    q33, q67 = thresholds[split_type]

    novelty = row["novelty"]

    if novelty <= q33:
        return "Low"

    if novelty <= q67:
        return "Medium"

    return "High"


def record_case(row, case_type):

    return {
        "case_type":
            case_type,

        "split_type":
            row["split_type"],

        "split_id":
            row["split_id"],

        "query_id":
            row["query_id"],

        "novelty":
            row["novelty"],

        "novelty_group":
            row["novelty_group"],

        "nearest_train_query":
            row.get(
                "nearest_train_query",
                np.nan,
            ),

        "balsa_status":
            row["balsa_status"],

        "pg_status":
            row["pg_status"],

        "balsa_end_to_end_ms":
            row.get(
                "balsa_end_to_end_ms",
                np.nan,
            ),

        "pg_end_to_end_ms":
            row.get(
                "pg_end_to_end_ms",
                np.nan,
            ),

        "balsa_vs_pg_ratio":
            row.get(
                "balsa_vs_pg_ratio",
                np.nan,
            ),

        "winner":
            row["winner"],

        "interpretation":
            "",
    }


def main():

    df = pd.read_csv(
        PERFORMANCE_FILE
    )

    groups = pd.read_csv(
        GROUP_FILE
    )

    # ========================================================
    # THRESHOLDS
    # ========================================================

    thresholds = {}

    threshold_rows = (
        groups[
            [
                "split_type",
                "q33",
                "q67",
            ]
        ]
        .drop_duplicates()
    )

    for _, row in threshold_rows.iterrows():

        thresholds[
            row["split_type"]
        ] = (
            row["q33"],
            row["q67"],
        )

    df["novelty_group"] = df.apply(
        lambda row:
            assign_group(
                row,
                thresholds,
            ),
        axis=1,
    )

    comparable = df[
        (df["balsa_status"] == "OK")
        & (df["pg_status"] == "OK")
        & df[
            "balsa_vs_pg_ratio"
        ].notna()
    ].copy()

    cases = []

    # ========================================================
    # CASE 1
    # High Novelty + worst slowdown
    # ========================================================

    high = comparable[
        comparable["novelty_group"]
        == "High"
    ]

    if not high.empty:

        row = high.loc[
            high[
                "balsa_vs_pg_ratio"
            ].idxmax()
        ]

        case = record_case(
            row,
            "HIGH_NOVELTY_WORST_SLOWDOWN",
        )

        case["interpretation"] = (
            "High Novelty đi cùng mức "
            "slowdown Balsa/PG rất lớn."
        )

        cases.append(case)

    # ========================================================
    # CASE 2
    # Low Novelty but still poor performance
    # ========================================================

    low = comparable[
        comparable["novelty_group"]
        == "Low"
    ]

    if not low.empty:

        row = low.loc[
            low[
                "balsa_vs_pg_ratio"
            ].idxmax()
        ]

        case = record_case(
            row,
            "LOW_NOVELTY_WORST_SLOWDOWN",
        )

        case["interpretation"] = (
            "Novelty thấp nhưng Balsa vẫn "
            "chậm đáng kể; Novelty không "
            "phải yếu tố duy nhất."
        )

        cases.append(case)

    # ========================================================
    # CASE 3
    # TIMEOUT
    # ========================================================

    timeouts = df[
        df["balsa_status"]
        == "TIMEOUT"
    ].copy()

    for _, row in timeouts.iterrows():

        case = record_case(
            row,
            "BALSA_TIMEOUT",
        )

        case["interpretation"] = (
            "Balsa timeout dù Novelty "
            "không nhất thiết thuộc nhóm cao."
        )

        cases.append(case)

    # ========================================================
    # CASE 4
    # Best Balsa relative performance
    # ========================================================

    if not comparable.empty:

        row = comparable.loc[
            comparable[
                "balsa_vs_pg_ratio"
            ].idxmin()
        ]

        case = record_case(
            row,
            "BEST_BALSA_RELATIVE_PERFORMANCE",
        )

        case["interpretation"] = (
            "Query có Balsa/PG thấp nhất "
            "trong cấu hình 3 iterations."
        )

        cases.append(case)

    # ========================================================
    # PER-SPLIT HIGH NOVELTY REPRESENTATIVE
    # ========================================================

    for split_type in [
        "LOO",
        "Random",
        "BaseQuery",
    ]:

        subset = comparable[
            (
                comparable["split_type"]
                == split_type
            )
            & (
                comparable[
                    "novelty_group"
                ]
                == "High"
            )
        ]

        if subset.empty:
            continue

        row = subset.loc[
            subset[
                "balsa_vs_pg_ratio"
            ].idxmax()
        ]

        case = record_case(
            row,
            (
                "HIGH_NOVELTY_"
                f"{split_type.upper()}"
            ),
        )

        case["interpretation"] = (
            f"High-Novelty representative "
            f"case của {split_type}."
        )

        cases.append(case)

    # ========================================================
    # SAVE
    # ========================================================

    result = pd.DataFrame(
        cases
    )

    result = result.drop_duplicates(
        subset=[
            "case_type",
            "split_type",
            "query_id",
        ]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "=== BALSA CASE STUDIES ==="
    )

    columns = [
        "case_type",
        "split_type",
        "query_id",
        "novelty",
        "novelty_group",
        "balsa_status",
        "balsa_vs_pg_ratio",
    ]

    print(
        result[
            columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()