from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "bao_novelty_performance.csv"
)

OUTPUT_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "bao_novelty_group_summary.csv"
)


def split_thresholds(df):
    """Return canonical split-level thresholds from all test observations."""
    q33 = round(float(df["novelty"].quantile(1 / 3)), 6)
    q67 = round(float(df["novelty"].quantile(2 / 3)), 6)
    return q33, q67


def assign_novelty_groups(df, q33, q67):
    """
    Chia Low / Medium / High Novelty TRONG TỪNG SPLIT.

    Dùng 1/3 và 2/3 quantile của Novelty trong split đó.

    Quan trọng:
    - Không gộp các split.
    - Query có cùng Novelty được giữ cùng nhóm.
    - Vì vậy số query trong 3 nhóm không nhất thiết bằng nhau.
    """

    df = df.copy()

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


def summarize_group(df, split, group):
    subset = df[
        df["novelty_group"] == group
    ].copy()

    n = len(subset)

    bao_wins = (
        subset["winner"] == "Bao"
    ).sum()

    pg_wins = (
        subset["winner"] == "PostgreSQL"
    ).sum()

    comparable = (
        subset["winner"].isin(
            ["Bao", "PostgreSQL", "Tie"]
        )
    ).sum()

    if comparable > 0:
        bao_win_rate = bao_wins / comparable
    else:
        bao_win_rate = np.nan

    return {
        "split_type": split,
        "novelty_group": group,
        "n": n,

        "novelty_min":
            subset["novelty"].min(),

        "novelty_median":
            subset["novelty"].median(),

        "novelty_max":
            subset["novelty"].max(),

        "median_bao_pg_ratio":
            subset["bao_vs_pg_ratio"].median(),

        "mean_bao_pg_ratio":
            subset["bao_vs_pg_ratio"].mean(),

        "bao_wins":
            bao_wins,

        "postgres_wins":
            pg_wins,

        "bao_win_rate":
            bao_win_rate,
    }


def main():
    # =========================================================
    # 1. LOAD DATA
    # =========================================================
    df = pd.read_csv(INPUT_FILE)

    print("=== INPUT ===")
    print(f"Total rows: {len(df)}")

    # =========================================================
    # 2. CHỈ GIỮ QUERY SO SÁNH ĐƯỢC
    # =========================================================
    valid = df[
        (df["bao_status"] == "OK")
        & (df["pg_status"] == "OK")
        & df["novelty"].notna()
        & df["bao_vs_pg_ratio"].notna()
    ].copy()

    print(
        f"Comparable rows: {len(valid)}"
    )

    # =========================================================
    # 3. PHÂN TÍCH RIÊNG TỪNG SPLIT
    # =========================================================
    results = []

    split_order = [
        "LOO",
        "Random",
        "BaseQuery",
    ]

    group_order = [
        "Low",
        "Medium",
        "High",
    ]

    for split in split_order:

        split_all = df[
            df["split_type"] == split
        ].copy()

        split_df = valid[
            valid["split_type"] == split
        ].copy()

        q33, q67 = split_thresholds(split_all)
        grouped_df = assign_novelty_groups(
            split_df,
            q33,
            q67,
        )

        print(
            f"\n=== {split} ==="
        )

        print(
            f"n = {len(grouped_df)}"
        )

        print(
            f"33rd percentile = {q33:.6f}"
        )

        print(
            f"67th percentile = {q67:.6f}"
        )

        print(
            "\nGroup counts:"
        )

        print(
            grouped_df[
                "novelty_group"
            ]
            .value_counts()
            .reindex(
                group_order,
                fill_value=0,
            )
            .to_string()
        )

        # =====================================================
        # Summary từng novelty group
        # =====================================================
        for group in group_order:

            result = summarize_group(
                grouped_df,
                split,
                group,
            )

            results.append(result)

    # =========================================================
    # 4. RESULT DATAFRAME
    # =========================================================
    result_df = pd.DataFrame(
        results
    )

    # =========================================================
    # 5. PRINT SUMMARY
    # =========================================================
    print(
        "\n=== NOVELTY GROUP PERFORMANCE ==="
    )

    for split in split_order:

        print(
            f"\n[{split}]"
        )

        split_result = result_df[
            result_df["split_type"]
            == split
        ]

        for _, row in split_result.iterrows():

            print(
                f"{row['novelty_group']:6s} "
                f"| n={int(row['n']):2d} "
                f"| novelty_med="
                f"{row['novelty_median']:.6f} "
                f"| ratio_med="
                f"{row['median_bao_pg_ratio']:.6f} "
                f"| Bao wins="
                f"{int(row['bao_wins'])} "
                f"| PG wins="
                f"{int(row['postgres_wins'])} "
                f"| Bao win rate="
                f"{row['bao_win_rate']:.3f}"
            )

    # =========================================================
    # 6. SAVE
    # =========================================================
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print(
        "\nResearch note:"
    )

    print(
        "Low / Medium / High được xác định "
        "RIÊNG trong từng split."
    )

    print(
        "Không so sánh trực tiếp mức Novelty "
        "Low của LOO với Low của BaseQuery "
        "như cùng một giá trị tuyệt đối."
    )


if __name__ == "__main__":
    main()