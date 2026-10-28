from pathlib import Path
import argparse
import csv
import re


SPLITS = {
    "LOO": "job__leave_one_out_split_1",
    "Random": "job__random_split_1",
    "BaseQuery": "job__base_query_split_1",
}


def natural_key(name):
    match = re.match(r"(\d+)([a-z]+)", name)
    if not match:
        return (9999, name)

    return (int(match.group(1)), match.group(2))


def normalize_query_id(filename):
    """
    2a.sql -> q2a
    17c.sql -> q17c
    """
    stem = Path(filename).stem
    return f"q{stem}"


def get_family(query_id):
    """
    q2a  -> q2
    q17c -> q17
    """
    match = re.match(r"q(\d+)", query_id)

    if not match:
        raise ValueError(f"Invalid query id: {query_id}")

    return f"q{match.group(1)}"


def sql_names(folder):
    return {
        p.name
        for p in folder.glob("*.sql")
        if p.is_file()
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--upstream-root",
        required=True,
        help="Path to cloned lqo_ml_perspective repository",
    )

    parser.add_argument(
        "--output",
        default="analysis/data/split_master.csv",
    )

    args = parser.parse_args()

    upstream = Path(args.upstream_root)

    queries_root = (
        upstream
        / "bao"
        / "queries"
    )

    canonical_dir = (
        queries_root
        / "join-order-benchmark"
    )

    # -----------------------------
    # Canonical JOB queries
    # -----------------------------

    canonical_queries = sql_names(canonical_dir)

    print(
        f"Canonical JOB queries: "
        f"{len(canonical_queries)}"
    )

    if len(canonical_queries) != 113:
        raise RuntimeError(
            f"Expected 113 JOB queries, "
            f"found {len(canonical_queries)}"
        )

    rows = []

    # -----------------------------
    # Process each official split
    # -----------------------------

    for split_type, dirname in SPLITS.items():

        split_dir = queries_root / dirname

        train_dir = split_dir / "train"
        test_dir = split_dir / "test"

        if not train_dir.exists():
            raise FileNotFoundError(train_dir)

        if not test_dir.exists():
            raise FileNotFoundError(test_dir)

        train = sql_names(train_dir)
        test = sql_names(test_dir)

        # -------------------------
        # Validation
        # -------------------------

        overlap = train & test

        if overlap:
            raise RuntimeError(
                f"{split_type}: "
                f"Train/Test overlap: "
                f"{sorted(overlap)}"
            )

        union = train | test

        missing = canonical_queries - union
        extra = union - canonical_queries

        if missing:
            raise RuntimeError(
                f"{split_type}: "
                f"Missing queries: "
                f"{sorted(missing)}"
            )

        if extra:
            raise RuntimeError(
                f"{split_type}: "
                f"Unexpected queries: "
                f"{sorted(extra)}"
            )

        print()
        print(f"=== {split_type}-1 ===")
        print(f"Train : {len(train)}")
        print(f"Test  : {len(test)}")
        print(f"Total : {len(union)}")
        print(f"Overlap: {len(overlap)}")

        # -------------------------
        # Build CSV rows
        # -------------------------

        for role, query_set in [
            ("train", train),
            ("test", test),
        ]:

            for sql_file in sorted(
                query_set,
                key=lambda x: natural_key(Path(x).stem),
            ):

                query_id = normalize_query_id(sql_file)

                rows.append(
                    {
                        "split_type": split_type,
                        "split_id": 1,
                        "query_id": query_id,
                        "family": get_family(query_id),
                        "role": role,
                        "sql_file": sql_file,
                        "source": (
                            "edualc/"
                            "lqo_ml_perspective"
                        ),
                    }
                )

    # -----------------------------
    # Write output
    # -----------------------------

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "split_type",
        "split_id",
        "query_id",
        "family",
        "role",
        "sql_file",
        "source",
    ]

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Rows written: {len(rows)}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()