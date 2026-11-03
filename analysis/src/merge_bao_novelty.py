from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

NOVELTY_FILE = ROOT / "analysis" / "data" / "novelty.csv"
BAO_FILE = ROOT / "bao" / "results" / "ALL_evaluation_results.csv"
PG_FILE = ROOT / "bao" / "results" / "ALL_postgres_results.csv"

OUTPUT_FILE = ROOT / "analysis" / "data" / "bao_novelty_performance.csv"


def main():
    # =========================================================
    # 1. LOAD DATA
    # =========================================================
    novelty = pd.read_csv(NOVELTY_FILE)
    bao = pd.read_csv(BAO_FILE)
    pg = pd.read_csv(PG_FILE)

    print("=== INPUT ===")
    print(f"Novelty rows   : {len(novelty)}")
    print(f"Bao rows       : {len(bao)}")
    print(f"PostgreSQL rows: {len(pg)}")

    # =========================================================
    # 2. MERGE KEY
    # =========================================================
    keys = ["split_type", "split_id", "query_id"]

    # Kiểm tra duplicate trước khi merge
    for name, df in [
        ("Novelty", novelty),
        ("Bao", bao),
        ("PostgreSQL", pg),
    ]:
        duplicates = df.duplicated(keys).sum()

        if duplicates != 0:
            raise ValueError(
                f"{name} has {duplicates} duplicated merge keys"
            )

    # =========================================================
    # 3. CHỌN CỘT BAO
    # =========================================================
    bao = bao[
        [
            "split_type",
            "split_id",
            "query_id",
            "planning_ms",
            "execution_ms",
            "end_to_end_ms",
            "status",
            "timeout_ms",
            "chosen_plan_id",
            "error",
        ]
    ].rename(
        columns={
            "planning_ms": "bao_planning_ms",
            "execution_ms": "bao_execution_ms",
            "end_to_end_ms": "bao_end_to_end_ms",
            "status": "bao_status",
            "timeout_ms": "bao_timeout_ms",
            "chosen_plan_id": "bao_chosen_plan_id",
            "error": "bao_error",
        }
    )

    # =========================================================
    # 4. CHỌN CỘT POSTGRESQL
    # =========================================================
    pg = pg[
        [
            "split_type",
            "split_id",
            "query_id",
            "planning_ms",
            "execution_ms",
            "end_to_end_ms",
            "status",
            "timeout_ms",
        ]
    ].rename(
        columns={
            "planning_ms": "pg_planning_ms",
            "execution_ms": "pg_execution_ms",
            "end_to_end_ms": "pg_end_to_end_ms",
            "status": "pg_status",
            "timeout_ms": "pg_timeout_ms",
        }
    )

    # =========================================================
    # 5. MERGE NOVELTY + BAO
    # =========================================================
    merged = novelty.merge(
        bao,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # =========================================================
    # 6. MERGE + POSTGRESQL
    # =========================================================
    merged = merged.merge(
        pg,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # =========================================================
    # 7. VALIDATE MATCH
    # =========================================================
    missing_bao = merged["bao_status"].isna().sum()
    missing_pg = merged["pg_status"].isna().sum()

    print("\n=== MERGE VALIDATION ===")
    print(f"Merged rows : {len(merged)}")
    print(f"Missing Bao : {missing_bao}")
    print(f"Missing PG  : {missing_pg}")

    if len(merged) != len(novelty):
        raise ValueError("Merged row count differs from novelty.csv")

    if missing_bao != 0:
        raise ValueError("Some Novelty rows do not match Bao")

    if missing_pg != 0:
        raise ValueError("Some Novelty rows do not match PostgreSQL")

    # =========================================================
    # 8. PERFORMANCE RATIO
    # =========================================================
    valid = (
        (merged["bao_status"] == "OK")
        & (merged["pg_status"] == "OK")
        & merged["bao_end_to_end_ms"].notna()
        & merged["pg_end_to_end_ms"].notna()
        & (merged["pg_end_to_end_ms"] > 0)
    )

    merged["bao_vs_pg_ratio"] = np.nan

    merged.loc[valid, "bao_vs_pg_ratio"] = (
        merged.loc[valid, "bao_end_to_end_ms"]
        / merged.loc[valid, "pg_end_to_end_ms"]
    )

    # =========================================================
    # 9. WINNER
    # =========================================================
    merged["winner"] = ""

    merged.loc[
        valid & (merged["bao_vs_pg_ratio"] < 1),
        "winner",
    ] = "Bao"

    merged.loc[
        valid & (merged["bao_vs_pg_ratio"] > 1),
        "winner",
    ] = "PostgreSQL"

    merged.loc[
        valid & (merged["bao_vs_pg_ratio"] == 1),
        "winner",
    ] = "Tie"

    # Timeout / error giữ riêng, không giả runtime
    merged.loc[
        ~valid,
        "winner",
    ] = "Not comparable"

    # =========================================================
    # 10. SORT + SAVE
    # =========================================================
    split_order = {
        "LOO": 1,
        "Random": 2,
        "BaseQuery": 3,
    }

    merged["_split_order"] = (
        merged["split_type"]
        .map(split_order)
        .fillna(99)
    )

    merged = (
        merged.sort_values(
            ["_split_order", "query_id"]
        )
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # =========================================================
    # 11. SUMMARY
    # =========================================================
    comparable = valid.sum()

    print("\n=== RESULT ===")
    print(f"Output rows       : {len(merged)}")
    print(f"Comparable queries: {comparable}")

    print("\nRows by split:")
    print(
        merged.groupby("split_type")
        .size()
        .to_string()
    )

    print("\nWinner:")
    print(
        merged["winner"]
        .value_counts()
        .to_string()
    )

    print(f"\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()