from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

NOVELTY_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "novelty.csv"
)

OUTPUT_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "balsa_novelty_performance.csv"
)

# ============================================================
# CURRENT 3-ITERATION BALSA BASELINE RUNS
# ============================================================

BALSA_RUNS = {
    "LOO": (
        ROOT
        / "balsa"
        / "runs"
        / "LOO-1"
        / "balsa-v2-run-51121"
    ),
    "Random": (
        ROOT
        / "balsa"
        / "runs"
        / "Random-1"
        / "balsa-v2-run-698900"
    ),
    "BaseQuery": (
        ROOT
        / "balsa"
        / "runs"
        / "BaseQuery-1"
        / "balsa-v2-run-1308903"
    ),
}

MAIN_TRAINING_ITERATIONS = 3
MAIN_TRAINING_LABEL = "resource_constrained_3_iterations"


def read_training_iterations(run_dir):
    """Read and validate the training budget recorded by the run manifest."""
    manifest_file = run_dir / "run_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Missing Balsa run manifest: {manifest_file}"
        )

    manifest = json.loads(
        manifest_file.read_text(encoding="utf-8")
    )
    command = str(
        manifest.get("training_command", "")
    )
    match = re.search(
        r"(?:^|\s)--iterations\s+(\d+)(?:\s|$)",
        command,
    )
    if not match:
        raise ValueError(
            f"Cannot determine --iterations from {manifest_file}"
        )

    iterations = int(match.group(1))
    if iterations != MAIN_TRAINING_ITERATIONS:
        raise ValueError(
            f"Main Balsa analysis expects {MAIN_TRAINING_ITERATIONS} "
            f"iterations, but {manifest_file} records {iterations}. "
            "Keep 3/5/10 milestone runs in the sensitivity analysis, "
            "not in the main Novelty merge."
        )
    return iterations


def read_baseline_provenance(run_dir):
    manifest_file = run_dir / "run_manifest.json"
    manifest = json.loads(
        manifest_file.read_text(encoding="utf-8")
    )
    info = manifest.get(
        "postgres_baseline_session",
        {},
    )
    connection = str(
        info.get("connection", "unknown")
    )
    return connection, connection in {
        "fresh",
        "fresh_per_query",
    }


def load_balsa_runs():
    evaluation_frames = []
    postgres_frames = []

    for split_type, run_dir in BALSA_RUNS.items():

        training_iterations = read_training_iterations(
            run_dir
        )

        (
            pg_baseline_session,
            pg_baseline_isolated,
        ) = read_baseline_provenance(
            run_dir
        )

        evaluation_file = (
            run_dir
            / "evaluation_results.csv"
        )

        postgres_file = (
            run_dir
            / "postgres_results.csv"
        )

        if not evaluation_file.exists():
            raise FileNotFoundError(
                f"Missing Balsa evaluation file: "
                f"{evaluation_file}"
            )

        if not postgres_file.exists():
            raise FileNotFoundError(
                f"Missing PostgreSQL file: "
                f"{postgres_file}"
            )

        evaluation = pd.read_csv(
            evaluation_file
        )

        evaluation[
            "balsa_training_iterations"
        ] = training_iterations

        evaluation[
            "balsa_training_label"
        ] = MAIN_TRAINING_LABEL

        postgres = pd.read_csv(
            postgres_file
        )

        postgres[
            "pg_baseline_session"
        ] = pg_baseline_session

        postgres[
            "pg_baseline_isolated"
        ] = pg_baseline_isolated

        # Validate split
        if not (
            evaluation["split_type"]
            == split_type
        ).all():
            raise ValueError(
                f"Unexpected split in "
                f"{evaluation_file}"
            )

        if not (
            postgres["split_type"]
            == split_type
        ).all():
            raise ValueError(
                f"Unexpected PostgreSQL split in "
                f"{postgres_file}"
            )

        evaluation_frames.append(
            evaluation
        )

        postgres_frames.append(
            postgres
        )

    balsa = pd.concat(
        evaluation_frames,
        ignore_index=True,
    )

    postgres = pd.concat(
        postgres_frames,
        ignore_index=True,
    )

    return balsa, postgres


def main():

    # ========================================================
    # 1. LOAD
    # ========================================================

    novelty = pd.read_csv(
        NOVELTY_FILE
    )

    balsa, postgres = (
        load_balsa_runs()
    )

    print("=== INPUT ===")
    print(
        f"Novelty rows   : {len(novelty)}"
    )
    print(
        f"Balsa rows     : {len(balsa)}"
    )
    print(
        f"PostgreSQL rows: {len(postgres)}"
    )

    # ========================================================
    # 2. MERGE KEY
    # ========================================================

    keys = [
        "split_type",
        "split_id",
        "query_id",
    ]

    for name, df in [
        ("Novelty", novelty),
        ("Balsa", balsa),
        ("PostgreSQL", postgres),
    ]:

        duplicates = (
            df.duplicated(keys).sum()
        )

        if duplicates != 0:
            raise ValueError(
                f"{name} has "
                f"{duplicates} duplicate keys"
            )

    # ========================================================
    # 3. SELECT BALSA COLUMNS
    # ========================================================

    balsa = balsa[
        [
            "split_type",
            "split_id",
            "query_id",
            "run_id",
            "checkpoint_id",
            "planning_ms",
            "inference_ms",
            "execution_ms",
            "end_to_end_ms",
            "status",
            "timeout_ms",
            "chosen_plan_id",
            "error",
            "balsa_training_iterations",
            "balsa_training_label",
        ]
    ].rename(
        columns={
            "run_id":
                "balsa_run_id",

            "checkpoint_id":
                "balsa_checkpoint_id",

            "planning_ms":
                "balsa_planning_ms",

            "inference_ms":
                "balsa_inference_ms",

            "execution_ms":
                "balsa_execution_ms",

            "end_to_end_ms":
                "balsa_end_to_end_ms",

            "status":
                "balsa_status",

            "timeout_ms":
                "balsa_timeout_ms",

            "chosen_plan_id":
                "balsa_chosen_plan_id",

            "error":
                "balsa_error",
        }
    )

    # ========================================================
    # 4. SELECT POSTGRESQL COLUMNS
    # ========================================================

    postgres = postgres[
        [
            "split_type",
            "split_id",
            "query_id",
            "planning_ms",
            "execution_ms",
            "end_to_end_ms",
            "status",
            "timeout_ms",
            "pg_baseline_session",
            "pg_baseline_isolated",
        ]
    ].rename(
        columns={
            "planning_ms":
                "pg_planning_ms",

            "execution_ms":
                "pg_execution_ms",

            "end_to_end_ms":
                "pg_end_to_end_ms",

            "status":
                "pg_status",

            "timeout_ms":
                "pg_timeout_ms",
        }
    )

    # ========================================================
    # 5. MERGE NOVELTY + BALSA
    # ========================================================

    merged = novelty.merge(
        balsa,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # 6. MERGE POSTGRESQL
    # ========================================================

    merged = merged.merge(
        postgres,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # 7. VALIDATE
    # ========================================================

    missing_balsa = (
        merged["balsa_status"]
        .isna()
        .sum()
    )

    missing_pg = (
        merged["pg_status"]
        .isna()
        .sum()
    )

    print(
        "\n=== MERGE VALIDATION ==="
    )

    print(
        f"Merged rows  : {len(merged)}"
    )

    print(
        f"Missing Balsa: {missing_balsa}"
    )

    print(
        f"Missing PG   : {missing_pg}"
    )

    if len(merged) != len(novelty):
        raise ValueError(
            "Merged row count differs "
            "from novelty.csv"
        )

    if missing_balsa != 0:
        raise ValueError(
            "Some Novelty rows "
            "do not match Balsa"
        )

    if missing_pg != 0:
        raise ValueError(
            "Some Novelty rows "
            "do not match PostgreSQL"
        )

    # ========================================================
    # 8. BALSA / POSTGRESQL RATIO
    # ========================================================

    valid = (
        (merged["balsa_status"] == "OK")
        & (merged["pg_status"] == "OK")
        & merged[
            "balsa_end_to_end_ms"
        ].notna()
        & merged[
            "pg_end_to_end_ms"
        ].notna()
        & (
            merged[
                "pg_end_to_end_ms"
            ] > 0
        )
    )

    merged[
        "balsa_vs_pg_ratio"
    ] = np.nan

    merged.loc[
        valid,
        "balsa_vs_pg_ratio",
    ] = (
        merged.loc[
            valid,
            "balsa_end_to_end_ms",
        ]
        /
        merged.loc[
            valid,
            "pg_end_to_end_ms",
        ]
    )

    # ========================================================
    # 9. WINNER
    # ========================================================

    merged["winner"] = (
        "Not comparable"
    )

    merged.loc[
        valid
        & (
            merged[
                "balsa_vs_pg_ratio"
            ] < 1
        ),
        "winner",
    ] = "Balsa"

    merged.loc[
        valid
        & (
            merged[
                "balsa_vs_pg_ratio"
            ] > 1
        ),
        "winner",
    ] = "PostgreSQL"

    merged.loc[
        valid
        & (
            merged[
                "balsa_vs_pg_ratio"
            ] == 1
        ),
        "winner",
    ] = "Tie"

    # ========================================================
    # 10. SORT
    # ========================================================

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
        merged
        .sort_values(
            [
                "_split_order",
                "query_id",
            ]
        )
        .drop(
            columns="_split_order"
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 11. SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # 12. SUMMARY
    # ========================================================

    comparable = int(
        valid.sum()
    )

    print("\n=== RESULT ===")

    print(
        f"Output rows       : "
        f"{len(merged)}"
    )

    print(
        f"Comparable queries: "
        f"{comparable}"
    )

    print(
        "\nRows by split:"
    )

    print(
        merged
        .groupby("split_type")
        .size()
        .to_string()
    )

    print(
        "\nBalsa status:"
    )

    print(
        merged[
            "balsa_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nPostgreSQL status:"
    )

    print(
        merged[
            "pg_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nWinner:"
    )

    print(
        merged[
            "winner"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()