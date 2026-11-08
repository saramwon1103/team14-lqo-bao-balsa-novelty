from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]

PERFORMANCE_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "bao_novelty_performance.csv"
)

FEATURE_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "query_features.csv"
)

CORRELATION_OUTPUT = (
    ROOT
    / "analysis"
    / "data"
    / "bao_complexity_correlations.csv"
)

GROUP_OUTPUT = (
    ROOT
    / "analysis"
    / "data"
    / "bao_complexity_by_novelty_group.csv"
)

ENRICHED_OUTPUT = (
    ROOT
    / "analysis"
    / "data"
    / "bao_novelty_complexity.csv"
)


def safe_spearman(x, y):
    """
    Tính Spearman an toàn.

    Nếu dữ liệu quá ít hoặc một biến không thay đổi,
    trả về NaN thay vì gây warning/error.
    """

    clean = pd.DataFrame({
        "x": x,
        "y": y,
    }).dropna()

    if len(clean) < 3:
        return np.nan, np.nan

    if clean["x"].nunique() < 2:
        return np.nan, np.nan

    if clean["y"].nunique() < 2:
        return np.nan, np.nan

    rho, p_value = spearmanr(
        clean["x"],
        clean["y"],
    )

    return rho, p_value


def classify_strength(rho):
    if pd.isna(rho):
        return "not_available"

    value = abs(rho)

    if value < 0.10:
        return "negligible"
    elif value < 0.30:
        return "weak"
    elif value < 0.50:
        return "moderate"
    else:
        return "strong"


def assign_novelty_groups(df):
    """
    Low / Medium / High được xác định RIÊNG
    trong từng split, giống DM2.
    """

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


def main():

    # =========================================================
    # 1. LOAD
    # =========================================================

    performance = pd.read_csv(
        PERFORMANCE_FILE
    )

    features = pd.read_csv(
        FEATURE_FILE
    )

    print("=== INPUT ===")
    print(
        f"Performance rows: {len(performance)}"
    )
    print(
        f"Feature rows    : {len(features)}"
    )

    # =========================================================
    # 2. VALIDATE QUERY FEATURES
    # =========================================================

    duplicate_features = (
        features
        .duplicated(["query_id"])
        .sum()
    )

    if duplicate_features != 0:
        raise ValueError(
            f"query_features.csv has "
            f"{duplicate_features} duplicate query_id"
        )

    feature_cols = features[
        [
            "query_id",
            "n_tables",
            "n_joins",
            "n_filters",
        ]
    ].copy()

    # =========================================================
    # 3. MERGE PERFORMANCE + COMPLEXITY
    # =========================================================

    df = performance.merge(
        feature_cols,
        on="query_id",
        how="left",
        validate="many_to_one",
    )

    missing_features = (
        df[
            [
                "n_tables",
                "n_joins",
                "n_filters",
            ]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    print(
        f"Missing feature rows: "
        f"{missing_features}"
    )

    if missing_features != 0:
        raise ValueError(
            "Some queries do not have "
            "complexity features"
        )

    # =========================================================
    # 4. CHỈ GIỮ QUERY SO SÁNH ĐƯỢC
    # =========================================================

    valid = df[
        (df["bao_status"] == "OK")
        & (df["pg_status"] == "OK")
        & df["bao_vs_pg_ratio"].notna()
        & df["novelty"].notna()
    ].copy()

    print(
        f"Comparable rows: {len(valid)}"
    )

    # =========================================================
    # 5. SAVE ENRICHED DATA
    # =========================================================

    df.to_csv(
        ENRICHED_OUTPUT,
        index=False,
    )

    # =========================================================
    # 6. CORRELATION ANALYSIS
    # =========================================================

    split_order = [
        "LOO",
        "Random",
        "BaseQuery",
    ]

    complexity_features = [
        "n_tables",
        "n_joins",
        "n_filters",
    ]

    correlation_results = []

    for split in split_order:

        split_df = valid[
            valid["split_type"] == split
        ].copy()

        print(
            f"\n=== {split}: CORRELATIONS ==="
        )

        for feature in complexity_features:

            # ---------------------------------------------
            # Novelty vs Complexity
            # ---------------------------------------------

            rho_novelty, p_novelty = (
                safe_spearman(
                    split_df["novelty"],
                    split_df[feature],
                )
            )

            correlation_results.append({
                "split_type": split,
                "relationship":
                    f"novelty_vs_{feature}",
                "n": len(split_df),
                "rho": rho_novelty,
                "p_value": p_novelty,
                "strength":
                    classify_strength(
                        rho_novelty
                    ),
            })

            print(
                f"Novelty vs {feature:9s} "
                f"| rho={rho_novelty:.6f} "
                f"| p={p_novelty:.6f}"
            )

            # ---------------------------------------------
            # Performance Ratio vs Complexity
            # ---------------------------------------------

            rho_perf, p_perf = (
                safe_spearman(
                    split_df[
                        "bao_vs_pg_ratio"
                    ],
                    split_df[feature],
                )
            )

            correlation_results.append({
                "split_type": split,
                "relationship":
                    f"bao_pg_ratio_vs_{feature}",
                "n": len(split_df),
                "rho": rho_perf,
                "p_value": p_perf,
                "strength":
                    classify_strength(
                        rho_perf
                    ),
            })

            print(
                f"Ratio   vs {feature:9s} "
                f"| rho={rho_perf:.6f} "
                f"| p={p_perf:.6f}"
            )

    correlation_df = pd.DataFrame(
        correlation_results
    )

    # =========================================================
    # 7. COMPLEXITY BY NOVELTY GROUP
    # =========================================================

    group_results = []

    group_order = [
        "Low",
        "Medium",
        "High",
    ]

    for split in split_order:

        split_df = valid[
            valid["split_type"] == split
        ].copy()

        split_df = assign_novelty_groups(
            split_df
        )

        print(
            f"\n=== {split}: "
            f"COMPLEXITY BY NOVELTY GROUP ==="
        )

        for group in group_order:

            subset = split_df[
                split_df[
                    "novelty_group"
                ] == group
            ].copy()

            if len(subset) == 0:
                continue

            result = {
                "split_type": split,
                "novelty_group": group,
                "n": len(subset),

                "median_novelty":
                    subset[
                        "novelty"
                    ].median(),

                "median_n_tables":
                    subset[
                        "n_tables"
                    ].median(),

                "median_n_joins":
                    subset[
                        "n_joins"
                    ].median(),

                "median_n_filters":
                    subset[
                        "n_filters"
                    ].median(),

                "median_bao_pg_ratio":
                    subset[
                        "bao_vs_pg_ratio"
                    ].median(),
            }

            group_results.append(result)

            print(
                f"{group:6s} "
                f"| n={len(subset):2d} "
                f"| tables_med="
                f"{result['median_n_tables']:.2f} "
                f"| joins_med="
                f"{result['median_n_joins']:.2f} "
                f"| filters_med="
                f"{result['median_n_filters']:.2f} "
                f"| ratio_med="
                f"{result['median_bao_pg_ratio']:.3f}"
            )

    group_df = pd.DataFrame(
        group_results
    )

    # =========================================================
    # 8. SAVE
    # =========================================================

    correlation_df.to_csv(
        CORRELATION_OUTPUT,
        index=False,
    )

    group_df.to_csv(
        GROUP_OUTPUT,
        index=False,
    )

    print("\n=== OUTPUT ===")

    print(
        f"Correlations:\n"
        f"{CORRELATION_OUTPUT}"
    )

    print(
        f"\nNovelty groups:\n"
        f"{GROUP_OUTPUT}"
    )

    print(
        f"\nEnriched dataset:\n"
        f"{ENRICHED_OUTPUT}"
    )

    # =========================================================
    # 9. RESEARCH NOTE
    # =========================================================

    print("\n=== RESEARCH NOTE ===")

    print(
        "Nếu High Novelty đồng thời có nhiều "
        "tables/joins/filters hơn, performance "
        "degradation có thể liên quan cả query "
        "complexity chứ không chỉ Novelty."
    )

    print(
        "Nếu complexity gần tương đương giữa "
        "các Novelty groups nhưng Bao/PG ratio "
        "vẫn tăng mạnh, Novelty trở thành tín hiệu "
        "đáng chú ý hơn."
    )


if __name__ == "__main__":
    main()