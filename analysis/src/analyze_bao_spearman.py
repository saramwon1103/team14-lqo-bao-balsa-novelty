from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


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
    / "bao_spearman_summary.csv"
)


def classify_strength(rho):
    """
    Diễn giải độ mạnh của tương quan theo |rho|.

    Lưu ý:
    Đây chỉ là cách mô tả mức độ tương quan,
    không phải ngưỡng thống kê tuyệt đối.
    """
    value = abs(rho)

    if value < 0.10:
        return "negligible"
    elif value < 0.30:
        return "weak"
    elif value < 0.50:
        return "moderate"
    else:
        return "strong"


def calculate_spearman(df, split_name):
    """
    Tính Spearman correlation giữa:

        novelty
        và
        bao_vs_pg_ratio

    trong MỘT split cụ thể.
    """

    clean = df[
        [
            "novelty",
            "bao_vs_pg_ratio",
        ]
    ].dropna()

    n = len(clean)

    if n < 3:
        return {
            "split_type": split_name,
            "n": n,
            "rho": None,
            "p_value": None,
            "strength": "insufficient_data",
            "direction": "insufficient_data",
        }

    rho, p_value = spearmanr(
        clean["novelty"],
        clean["bao_vs_pg_ratio"],
    )

    if rho > 0:
        direction = "positive"
    elif rho < 0:
        direction = "negative"
    else:
        direction = "none"

    return {
        "split_type": split_name,
        "n": n,
        "rho": rho,
        "p_value": p_value,
        "strength": classify_strength(rho),
        "direction": direction,
    }


def main():
    # =========================================================
    # 1. LOAD DATA
    # =========================================================
    df = pd.read_csv(INPUT_FILE)

    print("=== INPUT ===")
    print(f"Total rows: {len(df)}")

    # =========================================================
    # 2. CHỈ GIỮ QUERY CÓ THỂ SO SÁNH BAO VS POSTGRESQL
    # =========================================================
    valid = df[
        (df["bao_status"] == "OK")
        & (df["pg_status"] == "OK")
        & df["novelty"].notna()
        & df["bao_vs_pg_ratio"].notna()
    ].copy()

    print(f"Comparable rows: {len(valid)}")

    # =========================================================
    # 3. KIỂM TRA SỐ ROW THEO SPLIT
    # =========================================================
    print("\n=== COMPARABLE ROWS BY SPLIT ===")

    comparable_counts = (
        valid.groupby("split_type")
        .size()
    )

    print(comparable_counts.to_string())

    # =========================================================
    # 4. SPEARMAN RIÊNG CHO TỪNG SPLIT
    # =========================================================
    results = []

    split_order = [
        "LOO",
        "Random",
        "BaseQuery",
    ]

    for split in split_order:
        subset = valid[
            valid["split_type"] == split
        ].copy()

        result = calculate_spearman(
            subset,
            split,
        )

        results.append(result)

    result_df = pd.DataFrame(results)

    # =========================================================
    # 5. PRINT RESULT
    # =========================================================
    print(
        "\n=== SPEARMAN: NOVELTY vs BAO/PG RATIO "
        "BY SPLIT ==="
    )

    for _, row in result_df.iterrows():

        if pd.isna(row["rho"]):
            print(
                f"{row['split_type']:10s} "
                f"| n={int(row['n']):2d} "
                f"| insufficient data"
            )
            continue

        print(
            f"{row['split_type']:10s} "
            f"| n={int(row['n']):2d} "
            f"| rho={row['rho']:.6f} "
            f"| p={row['p_value']:.6f} "
            f"| direction={row['direction']} "
            f"| strength={row['strength']}"
        )

    # =========================================================
    # 6. INTERPRETATION THEO TỪNG SPLIT
    # =========================================================
    print("\n=== INTERPRETATION BY SPLIT ===")

    for _, row in result_df.iterrows():

        split = row["split_type"]
        rho = row["rho"]
        p_value = row["p_value"]

        print(f"\n[{split}]")

        if pd.isna(rho):
            print(
                "Không đủ dữ liệu để tính Spearman."
            )
            continue

        if rho > 0:
            print(
                "Novelty tăng thì Bao/PG ratio "
                "có xu hướng tăng."
            )
            print(
                "=> Bao có xu hướng tương đối kém hơn "
                "PostgreSQL khi Novelty tăng."
            )

        elif rho < 0:
            print(
                "Novelty tăng thì Bao/PG ratio "
                "có xu hướng giảm."
            )
            print(
                "=> Bao có xu hướng tương đối tốt hơn "
                "PostgreSQL khi Novelty tăng."
            )

        else:
            print(
                "Không thấy quan hệ đơn điệu "
                "giữa Novelty và Bao/PG ratio."
            )

        print(
            f"Spearman rho = {rho:.6f}"
        )

        print(
            f"p-value = {p_value:.6f}"
        )

        if p_value < 0.05:
            print(
                "=> Có tín hiệu thống kê ở ngưỡng p < 0.05."
            )
        else:
            print(
                "=> Chưa có đủ bằng chứng thống kê "
                "ở ngưỡng p < 0.05."
            )

    # =========================================================
    # 7. SAVE RESULT
    # =========================================================
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n=== OUTPUT ===")
    print(f"Saved to: {OUTPUT_FILE}")

    # =========================================================
    # 8. RESEARCH REMINDER
    # =========================================================
    print("\n=== RESEARCH NOTE ===")

    print(
        "Không gộp LOO, Random và BaseQuery thành một "
        "Spearman chung để trả lời RQ2."
    )

    print(
        "Mỗi split có Train/Test context khác nhau, "
        "nên Novelty phải được phân tích riêng theo split."
    )

    print(
        "Bước tiếp theo: chia Novelty thành "
        "Low / Medium / High trong từng split."
    )


if __name__ == "__main__":
    main()