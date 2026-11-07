from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "balsa_novelty_performance.csv"
)

OUTPUT_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "balsa_novelty_group_summary.csv"
)


def assign_novelty_group(group):
    """
    Split-specific quantile grouping:
    Low     : novelty <= q33
    Medium  : q33 < novelty <= q67
    High    : novelty > q67
    """

    group = group.copy()

    # novelty.csv is stored at 6 decimal places.  Round the quantile cutoffs
    # to the same precision before comparison so values that are textually
    # equal to a cutoff are not split by floating-point representation noise.
    q33 = round(float(group["novelty"].quantile(1 / 3)), 6)
    q67 = round(float(group["novelty"].quantile(2 / 3)), 6)

    conditions = [
        group["novelty"] <= q33,
        (
            (group["novelty"] > q33)
            & (group["novelty"] <= q67)
        ),
        group["novelty"] > q67,
    ]

    choices = [
        "Low",
        "Medium",
        "High",
    ]

    group["novelty_group"] = np.select(
        conditions,
        choices,
        default="Unknown",
    )

    group["q33"] = q33
    group["q67"] = q67

    return group


def summarize_group(group):

    total_n = len(group)

    comparable = group[
        (group["balsa_status"] == "OK")
        & (group["pg_status"] == "OK")
        & group["balsa_vs_pg_ratio"].notna()
    ].copy()

    comparable_n = len(comparable)

    balsa_timeouts = (
        group["balsa_status"]
        == "TIMEOUT"
    ).sum()

    timeout_rate = (
        balsa_timeouts / total_n
        if total_n > 0
        else np.nan
    )

    if comparable_n > 0:

        median_ratio = (
            comparable[
                "balsa_vs_pg_ratio"
            ]
            .median()
        )

        mean_ratio = (
            comparable[
                "balsa_vs_pg_ratio"
            ]
            .mean()
        )

        median_balsa_ms = (
            comparable[
                "balsa_end_to_end_ms"
            ]
            .median()
        )

        median_pg_ms = (
            comparable[
                "pg_end_to_end_ms"
            ]
            .median()
        )

        balsa_wins = (
            comparable["winner"]
            == "Balsa"
        ).sum()

        pg_wins = (
            comparable["winner"]
            == "PostgreSQL"
        ).sum()

        ties = (
            comparable["winner"]
            == "Tie"
        ).sum()

        balsa_win_rate = (
            balsa_wins / comparable_n
        )

    else:

        median_ratio = np.nan
        mean_ratio = np.nan
        median_balsa_ms = np.nan
        median_pg_ms = np.nan
        balsa_wins = 0
        pg_wins = 0
        ties = 0
        balsa_win_rate = np.nan

    return pd.Series(
        {
            "n": total_n,
            "comparable_n": comparable_n,

            "median_novelty":
                group["novelty"].median(),

            "mean_novelty":
                group["novelty"].mean(),

            "median_balsa_pg_ratio":
                median_ratio,

            "mean_balsa_pg_ratio":
                mean_ratio,

            "median_balsa_end_to_end_ms":
                median_balsa_ms,

            "median_pg_end_to_end_ms":
                median_pg_ms,

            "balsa_wins":
                balsa_wins,

            "postgresql_wins":
                pg_wins,

            "ties":
                ties,

            "balsa_win_rate":
                balsa_win_rate,

            "balsa_timeouts":
                balsa_timeouts,

            "timeout_rate":
                timeout_rate,

            "q33":
                group["q33"].iloc[0],

            "q67":
                group["q67"].iloc[0],
        }
    )


def main():

    df = pd.read_csv(INPUT_FILE)

    print("=== INPUT ===")
    print(f"Rows: {len(df)}")

    # ========================================================
    # 1. ASSIGN NOVELTY GROUP PER SPLIT
    # ========================================================

    grouped_frames = []

    for split_type in [
        "LOO",
        "Random",
        "BaseQuery",
    ]:

        split_df = df[
            df["split_type"]
            == split_type
        ].copy()

        if split_df.empty:
            raise ValueError(
                f"No rows for split "
                f"{split_type}"
            )

        split_df = (
            assign_novelty_group(
                split_df
            )
        )

        grouped_frames.append(
            split_df
        )

    grouped_df = pd.concat(
        grouped_frames,
        ignore_index=True,
    )

    # ========================================================
    # 2. SUMMARY PER SPLIT + NOVELTY GROUP
    # ========================================================

    summary = (
        grouped_df
        .groupby(
            [
                "split_type",
                "novelty_group",
            ],
            observed=True,
        )
        .apply(
            summarize_group,
            include_groups=False,
        )
        .reset_index()
    )

    # ========================================================
    # 3. ORDER
    # ========================================================

    split_order = {
        "LOO": 1,
        "Random": 2,
        "BaseQuery": 3,
    }

    group_order = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
    }

    summary["_split_order"] = (
        summary["split_type"]
        .map(split_order)
    )

    summary["_group_order"] = (
        summary["novelty_group"]
        .map(group_order)
    )

    summary = (
        summary
        .sort_values(
            [
                "_split_order",
                "_group_order",
            ]
        )
        .drop(
            columns=[
                "_split_order",
                "_group_order",
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 4. SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # 5. PRINT
    # ========================================================

    print(
        "\n=== NOVELTY GROUP SUMMARY ==="
    )

    display_cols = [
        "split_type",
        "novelty_group",
        "n",
        "comparable_n",
        "median_novelty",
        "median_balsa_pg_ratio",
        "balsa_win_rate",
        "balsa_timeouts",
        "timeout_rate",
    ]

    print(
        summary[
            display_cols
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== THRESHOLDS ==="
    )

    thresholds = (
        summary[
            [
                "split_type",
                "q33",
                "q67",
            ]
        ]
        .drop_duplicates()
    )

    print(
        thresholds.to_string(
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