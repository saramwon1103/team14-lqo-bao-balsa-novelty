"""
compute_novelty.py

Compute structural Novelty Score for JOB test queries.

Inputs:
    analysis/data/split_master.csv
    analysis/data/query_features.csv

Outputs:
    analysis/data/novelty.csv
    analysis/data/novelty_pairs.csv

Novelty definition
------------------

For each feature type:

    Jaccard distance:

        d(A, B) = 1 - |A ∩ B| / |A ∪ B|

    If both sets are empty:

        d(A, B) = 0

For query q and training query h:

    D(q, h)
      = (d_table + d_join + d_filter) / 3

For split s:

    Novelty_s(q)
      = min D(q, h)
        for every h in Train_s

Important:
    A test query is compared ONLY with the Train set
    belonging to the same split.

TRACE MODE
----------

Example:

    python analysis/src/compute_novelty.py \
        --trace-query q7c \
        --trace-split LOO

Trace mode prints:

    STEP 1
        Test-query structural features

    STEP 2
        Distance from the test query to EVERY Train query

    STEP 3
        Detailed Jaccard calculation for the nearest query:
            intersection
            union
            Jaccard formula
            component distances
            combined D(q,h)

    STEP 4
        MIN operation and final Novelty

Optional:

    --trace-detail all

prints the full Jaccard derivation for EVERY Train candidate.
This is very verbose, so "nearest" is the default.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# Constants
# ============================================================

EPS = 1e-12


# ============================================================
# Query helpers
# ============================================================


def query_sort_key(
    query_id: str,
) -> Tuple[int, str]:
    """
    Natural JOB ordering:

        q1a
        q1b
        q2a
        q10a
    """

    match = re.fullmatch(
        r"q(\d+)([a-z]+)",
        query_id.lower(),
    )

    if not match:
        return (
            999999,
            query_id,
        )

    return (
        int(match.group(1)),
        match.group(2),
    )


def family_from_query_id(
    query_id: str,
) -> str:
    """
    q7c  -> q7
    q15a -> q15
    """

    match = re.fullmatch(
        r"q(\d+)[a-z]+",
        query_id.lower(),
    )

    if not match:
        raise ValueError(
            f"Invalid JOB query id: {query_id}"
        )

    return f"q{match.group(1)}"


# ============================================================
# Feature loading
# ============================================================


def parse_json_set(
    value: str,
    field_name: str,
    query_id: str,
) -> Set[str]:
    """
    Parse JSON list stored inside query_features.csv
    into a Python set.
    """

    try:
        parsed = json.loads(value)

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"{query_id}: invalid JSON "
            f"in field '{field_name}': "
            f"{exc}"
        ) from exc

    if not isinstance(
        parsed,
        list,
    ):
        raise ValueError(
            f"{query_id}: field "
            f"'{field_name}' must be "
            f"a JSON list."
        )

    return set(parsed)


def load_features(
    path: Path,
) -> Dict[str, dict]:
    """
    Load query_features.csv.

    Result:

        features["q7c"] = {
            "family": ...,
            "tables": set(...),
            "joins": set(...),
            "filters": set(...)
        }
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: "
            f"{path}"
        )

    features: Dict[str, dict] = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "query_id",
            "family",
            "tables",
            "joins",
            "filters",
            "n_tables",
            "n_joins",
            "n_filters",
            "parse_status",
        }

        actual = set(
            reader.fieldnames
            or []
        )

        missing = required - actual

        if missing:
            raise RuntimeError(
                "query_features.csv is missing "
                f"columns: {sorted(missing)}"
            )

        for row in reader:

            query_id = (
                row["query_id"]
                .strip()
                .lower()
            )

            if query_id in features:
                raise RuntimeError(
                    f"Duplicate feature row: "
                    f"{query_id}"
                )

            if (
                row["parse_status"]
                .strip()
                .upper()
                != "OK"
            ):
                raise RuntimeError(
                    f"{query_id}: "
                    f"parse_status is not OK."
                )

            tables = parse_json_set(
                row["tables"],
                "tables",
                query_id,
            )

            joins = parse_json_set(
                row["joins"],
                "joins",
                query_id,
            )

            filters = parse_json_set(
                row["filters"],
                "filters",
                query_id,
            )

            # ------------------------------------
            # Feature-count validation
            # ------------------------------------

            if (
                len(tables)
                != int(row["n_tables"])
            ):
                raise RuntimeError(
                    f"{query_id}: "
                    f"n_tables does not "
                    f"match tables."
                )

            if (
                len(joins)
                != int(row["n_joins"])
            ):
                raise RuntimeError(
                    f"{query_id}: "
                    f"n_joins does not "
                    f"match joins."
                )

            if (
                len(filters)
                != int(row["n_filters"])
            ):
                raise RuntimeError(
                    f"{query_id}: "
                    f"n_filters does not "
                    f"match filters."
                )

            features[query_id] = {
                "family":
                    row["family"]
                    .strip()
                    .lower(),

                "tables":
                    tables,

                "joins":
                    joins,

                "filters":
                    filters,
            }

    return features


# ============================================================
# Split loading
# ============================================================


def load_splits(
    path: Path,
) -> Dict[
    Tuple[str, int],
    Dict[str, Set[str]],
]:
    """
    Load split_master.csv.

    Example:

        {
            ("LOO", 1): {
                "train": {...},
                "test": {...}
            },

            ("Random", 1): {
                ...
            }
        }
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: "
            f"{path}"
        )

    splits = defaultdict(
        lambda: {
            "train": set(),
            "test": set(),
        }
    )

    seen_rows = set()

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "split_type",
            "split_id",
            "query_id",
            "family",
            "role",
        }

        actual = set(
            reader.fieldnames
            or []
        )

        missing = required - actual

        if missing:
            raise RuntimeError(
                "split_master.csv is missing "
                f"columns: {sorted(missing)}"
            )

        for row in reader:

            split_type = (
                row["split_type"]
                .strip()
            )

            split_id = int(
                row["split_id"]
            )

            query_id = (
                row["query_id"]
                .strip()
                .lower()
            )

            role = (
                row["role"]
                .strip()
                .lower()
            )

            if role not in {
                "train",
                "test",
            }:
                raise RuntimeError(
                    f"Invalid role: {role}"
                )

            row_key = (
                split_type,
                split_id,
                query_id,
            )

            if row_key in seen_rows:
                raise RuntimeError(
                    "Duplicate split row: "
                    f"{row_key}"
                )

            seen_rows.add(
                row_key
            )

            splits[
                (
                    split_type,
                    split_id,
                )
            ][role].add(
                query_id
            )

    # ----------------------------------------
    # Validate Train/Test separation
    # ----------------------------------------

    for (
        split_key,
        roles,
    ) in splits.items():

        train = roles["train"]
        test = roles["test"]

        overlap = (
            train
            & test
        )

        if overlap:
            raise RuntimeError(
                f"{split_key}: "
                f"Train/Test overlap: "
                f"{sorted(overlap)}"
            )

        if not train:
            raise RuntimeError(
                f"{split_key}: "
                f"empty Train set."
            )

        if not test:
            raise RuntimeError(
                f"{split_key}: "
                f"empty Test set."
            )

    return dict(splits)


# ============================================================
# Jaccard calculation
# ============================================================


def jaccard_details(
    a: Set[str],
    b: Set[str],
) -> dict:
    """
    Calculate Jaccard distance while retaining
    all intermediate values needed for tracing.

        d(A,B)
          = 1 - |A intersection B|
                -------------------
                  |A union B|

    Returns:
        {
            "a": ...,
            "b": ...,
            "intersection": ...,
            "union": ...,
            "n_a": ...,
            "n_b": ...,
            "n_intersection": ...,
            "n_union": ...,
            "similarity": ...,
            "distance": ...
        }
    """

    intersection = (
        a & b
    )

    union = (
        a | b
    )

    n_a = len(a)
    n_b = len(b)

    n_intersection = len(
        intersection
    )

    n_union = len(
        union
    )

    # Both empty sets are treated
    # as structurally identical.
    if n_union == 0:

        similarity = 1.0
        distance = 0.0

    else:

        similarity = (
            n_intersection
            / n_union
        )

        distance = (
            1.0
            - similarity
        )

    return {
        "a":
            a,

        "b":
            b,

        "intersection":
            intersection,

        "union":
            union,

        "n_a":
            n_a,

        "n_b":
            n_b,

        "n_intersection":
            n_intersection,

        "n_union":
            n_union,

        "similarity":
            similarity,

        "distance":
            distance,
    }


def query_distance_details(
    query_features: dict,
    train_features: dict,
) -> dict:
    """
    Calculate all three structural distances.

        d_table
        d_join
        d_filter

    Then:

        D(q,h)
          = (d_table
             + d_join
             + d_filter) / 3
    """

    table = jaccard_details(
        query_features["tables"],
        train_features["tables"],
    )

    join = jaccard_details(
        query_features["joins"],
        train_features["joins"],
    )

    filter_ = jaccard_details(
        query_features["filters"],
        train_features["filters"],
    )

    total = (
        table["distance"]
        + join["distance"]
        + filter_["distance"]
    ) / 3.0

    return {
        "table":
            table,

        "join":
            join,

        "filter":
            filter_,

        "distance":
            total,
    }


# ============================================================
# Novelty calculation
# ============================================================


def compute_novelty(
    features: Dict[str, dict],
    splits: Dict[
        Tuple[str, int],
        Dict[str, Set[str]],
    ],
) -> Tuple[
    List[dict],
    List[dict],
]:
    """
    Compare every Test query against every Train query
    of the corresponding split.

    Returns:

        novelty_rows
        pair_rows
    """

    novelty_rows: List[dict] = []
    pair_rows: List[dict] = []

    for (
        split_type,
        split_id,
    ) in sorted(
        splits.keys(),
        key=lambda item: (
            item[0],
            item[1],
        ),
    ):

        split = splits[
            (
                split_type,
                split_id,
            )
        ]

        train_queries = sorted(
            split["train"],
            key=query_sort_key,
        )

        test_queries = sorted(
            split["test"],
            key=query_sort_key,
        )

        print()
        print(
            f"=== "
            f"{split_type}-"
            f"{split_id} ==="
        )

        print(
            "Train candidates : "
            f"{len(train_queries)}"
        )

        print(
            "Test queries     : "
            f"{len(test_queries)}"
        )

        # ------------------------------------
        # Verify all query features exist
        # ------------------------------------

        all_queries = (
            set(train_queries)
            | set(test_queries)
        )

        missing_features = (
            all_queries
            - set(features.keys())
        )

        if missing_features:
            raise RuntimeError(
                f"{split_type}-"
                f"{split_id}: "
                f"missing features for "
                f"{sorted(missing_features)}"
            )

        # ------------------------------------
        # Process each Test query
        # ------------------------------------

        for test_query in (
            test_queries
        ):

            test_feature = (
                features[
                    test_query
                ]
            )

            candidates = []

            # --------------------------------
            # Compare against EVERY Train q
            # --------------------------------

            for train_query in (
                train_queries
            ):

                train_feature = (
                    features[
                        train_query
                    ]
                )

                details = (
                    query_distance_details(
                        test_feature,
                        train_feature,
                    )
                )

                distance = (
                    details[
                        "distance"
                    ]
                )

                d_table = (
                    details[
                        "table"
                    ][
                        "distance"
                    ]
                )

                d_join = (
                    details[
                        "join"
                    ][
                        "distance"
                    ]
                )

                d_filter = (
                    details[
                        "filter"
                    ][
                        "distance"
                    ]
                )

                candidate = {
                    "train_query":
                        train_query,

                    "distance":
                        distance,

                    "details":
                        details,
                }

                candidates.append(
                    candidate
                )

                # --------------------------------
                # Save all pair distances
                # --------------------------------

                pair_rows.append(
                    {
                        "split_type":
                            split_type,

                        "split_id":
                            split_id,

                        "test_query":
                            test_query,

                        "test_family":
                            test_feature[
                                "family"
                            ],

                        "train_query":
                            train_query,

                        "train_family":
                            train_feature[
                                "family"
                            ],

                        "distance":
                            f"{distance:.6f}",

                        "d_table":
                            f"{d_table:.6f}",

                        "d_join":
                            f"{d_join:.6f}",

                        "d_filter":
                            f"{d_filter:.6f}",

                        "table_intersection_count":
                            details[
                                "table"
                            ][
                                "n_intersection"
                            ],

                        "table_union_count":
                            details[
                                "table"
                            ][
                                "n_union"
                            ],

                        "join_intersection_count":
                            details[
                                "join"
                            ][
                                "n_intersection"
                            ],

                        "join_union_count":
                            details[
                                "join"
                            ][
                                "n_union"
                            ],

                        "filter_intersection_count":
                            details[
                                "filter"
                            ][
                                "n_intersection"
                            ],

                        "filter_union_count":
                            details[
                                "filter"
                            ][
                                "n_union"
                            ],
                    }
                )

            # ------------------------------------
            # Find minimum distance
            # ------------------------------------

            min_distance = min(
                candidate[
                    "distance"
                ]
                for candidate
                in candidates
            )

            tied = [
                candidate
                for candidate
                in candidates
                if abs(
                    candidate[
                        "distance"
                    ]
                    - min_distance
                )
                <= EPS
            ]

            # Deterministic tie break.
            tied.sort(
                key=lambda item:
                    query_sort_key(
                        item[
                            "train_query"
                        ]
                    )
            )

            nearest = tied[0]

            nearest_query = (
                nearest[
                    "train_query"
                ]
            )

            nearest_details = (
                nearest[
                    "details"
                ]
            )

            nearest_queries = [
                candidate[
                    "train_query"
                ]
                for candidate
                in tied
            ]

            novelty_rows.append(
                {
                    "query_id":
                        test_query,

                    "family":
                        test_feature[
                            "family"
                        ],

                    "split_type":
                        split_type,

                    "split_id":
                        split_id,

                    "novelty":
                        (
                            f"{min_distance:.6f}"
                        ),

                    "nearest_train_query":
                        nearest_query,

                    "nearest_train_family":
                        features[
                            nearest_query
                        ][
                            "family"
                        ],

                    "d_table":
                        (
                            f"{nearest_details['table']['distance']:.6f}"
                        ),

                    "d_join":
                        (
                            f"{nearest_details['join']['distance']:.6f}"
                        ),

                    "d_filter":
                        (
                            f"{nearest_details['filter']['distance']:.6f}"
                        ),

                    "n_train_candidates":
                        len(
                            train_queries
                        ),

                    "nearest_tie_count":
                        len(tied),

                    "nearest_train_queries":
                        json.dumps(
                            nearest_queries,
                            ensure_ascii=False,
                        ),
                }
            )

            print(
                f"{test_query:>5} | "
                f"Novelty="
                f"{min_distance:.6f} | "
                f"nearest="
                f"{nearest_query} | "
                f"T="
                f"{nearest_details['table']['distance']:.3f} | "
                f"J="
                f"{nearest_details['join']['distance']:.3f} | "
                f"F="
                f"{nearest_details['filter']['distance']:.3f}"
            )

    return (
        novelty_rows,
        pair_rows,
    )


# ============================================================
# CSV output
# ============================================================


def write_novelty(
    path: Path,
    rows: List[dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "query_id",
        "family",
        "split_type",
        "split_id",
        "novelty",
        "nearest_train_query",
        "nearest_train_family",
        "d_table",
        "d_join",
        "d_filter",
        "n_train_candidates",
        "nearest_tie_count",
        "nearest_train_queries",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def write_pairs(
    path: Path,
    rows: List[dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "split_type",
        "split_id",
        "test_query",
        "test_family",
        "train_query",
        "train_family",
        "distance",
        "d_table",
        "d_join",
        "d_filter",

        # Diagnostic evidence for Jaccard
        "table_intersection_count",
        "table_union_count",

        "join_intersection_count",
        "join_union_count",

        "filter_intersection_count",
        "filter_union_count",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# Summary
# ============================================================


def print_summary(
    rows: List[dict],
) -> None:

    grouped = defaultdict(
        list
    )

    for row in rows:

        key = (
            row["split_type"],
            row["split_id"],
        )

        grouped[
            key
        ].append(
            float(
                row[
                    "novelty"
                ]
            )
        )

    print()
    print("=" * 72)
    print(
        "NOVELTY SUMMARY"
    )
    print("=" * 72)

    for (
        split_type,
        split_id,
    ) in sorted(
        grouped.keys(),
        key=lambda item: (
            item[0],
            item[1],
        ),
    ):

        values = grouped[
            (
                split_type,
                split_id,
            )
        ]

        zero_count = sum(
            abs(value)
            <= EPS
            for value
            in values
        )

        print(
            f"{split_type}-"
            f"{split_id}: "
            f"n={len(values)} | "
            f"min="
            f"{min(values):.6f} | "
            f"median="
            f"{statistics.median(values):.6f} | "
            f"mean="
            f"{statistics.mean(values):.6f} | "
            f"max="
            f"{max(values):.6f} | "
            f"zero="
            f"{zero_count}"
        )

    print("=" * 72)


# ============================================================
# TRACE / DEMO helpers
# ============================================================


def print_feature_set(
    title: str,
    values: Set[str],
) -> None:
    """
    Pretty-print one feature set.
    """

    print()
    print(
        f"{title} "
        f"({len(values)}):"
    )

    if not values:

        print(
            "  ∅"
        )

        return

    for value in sorted(
        values
    ):
        print(
            f"  - {value}"
        )


def print_jaccard_trace(
    label: str,
    details: dict,
) -> None:
    """
    Print every intermediate step of one
    Jaccard distance calculation.
    """

    print()
    print(
        "-" * 70
    )

    print(
        f"{label}"
    )

    print(
        "-" * 70
    )

    print(
        f"|A| = "
        f"{details['n_a']}"
    )

    print(
        f"|B| = "
        f"{details['n_b']}"
    )

    print()

    print(
        "A ∩ B:"
    )

    if details[
        "intersection"
    ]:

        for value in sorted(
            details[
                "intersection"
            ]
        ):

            print(
                f"  - {value}"
            )

    else:

        print(
            "  ∅"
        )

    print()

    print(
        "|A ∩ B| = "
        f"{details['n_intersection']}"
    )

    print()

    print(
        "A ∪ B:"
    )

    if details[
        "union"
    ]:

        for value in sorted(
            details[
                "union"
            ]
        ):

            print(
                f"  - {value}"
            )

    else:

        print(
            "  ∅"
        )

    print()

    print(
        "|A ∪ B| = "
        f"{details['n_union']}"
    )

    print()

    if (
        details[
            "n_union"
        ]
        == 0
    ):

        print(
            "Both sets are empty."
        )

        print(
            "By definition:"
        )

        print(
            "d(A,B) = 0"
        )

    else:

        print(
            "Jaccard similarity"
        )

        print(
            "  = |A ∩ B| / |A ∪ B|"
        )

        print(
            "  = "
            f"{details['n_intersection']} "
            "/ "
            f"{details['n_union']}"
        )

        print(
            "  = "
            f"{details['similarity']:.6f}"
        )

        print()

        print(
            "Jaccard distance"
        )

        print(
            "  = 1 - "
            "|A ∩ B| / |A ∪ B|"
        )

        print(
            "  = 1 - "
            f"{details['n_intersection']} "
            "/ "
            f"{details['n_union']}"
        )

        print(
            "  = "
            f"{details['distance']:.6f}"
        )


def print_candidate_detail(
    test_query: str,
    train_query: str,
    details: dict,
) -> None:
    """
    Print full distance derivation for a
    single Test-vs-Train pair.
    """

    print()
    print("=" * 72)

    print(
        "DETAILED PAIR CALCULATION"
    )

    print("=" * 72)

    print(
        f"Test query  : "
        f"{test_query}"
    )

    print(
        f"Train query : "
        f"{train_query}"
    )

    # ----------------------------------------
    # Tables
    # ----------------------------------------

    print_jaccard_trace(
        "[1] TABLE DISTANCE",
        details[
            "table"
        ],
    )

    # ----------------------------------------
    # Joins
    # ----------------------------------------

    print_jaccard_trace(
        "[2] JOIN DISTANCE",
        details[
            "join"
        ],
    )

    # ----------------------------------------
    # Filters
    # ----------------------------------------

    print_jaccard_trace(
        "[3] FILTER DISTANCE",
        details[
            "filter"
        ],
    )

    # ----------------------------------------
    # Combined query distance
    # ----------------------------------------

    d_table = (
        details[
            "table"
        ][
            "distance"
        ]
    )

    d_join = (
        details[
            "join"
        ][
            "distance"
        ]
    )

    d_filter = (
        details[
            "filter"
        ][
            "distance"
        ]
    )

    print()
    print("-" * 70)
    print(
        "[4] COMBINED QUERY DISTANCE"
    )
    print("-" * 70)

    print(
        f"D({test_query}, "
        f"{train_query})"
    )

    print(
        "  = "
        "(d_table + "
        "d_join + "
        "d_filter) / 3"
    )

    print()

    print(
        "  = "
        f"({d_table:.6f} "
        f"+ {d_join:.6f} "
        f"+ {d_filter:.6f}) / 3"
    )

    print()

    print(
        "  = "
        f"{details['distance']:.6f}"
    )


def find_split_key(
    splits: Dict[
        Tuple[str, int],
        Dict[str, Set[str]],
    ],
    requested_type: str,
    requested_id: int,
) -> Tuple[str, int]:
    """
    Find split key case-insensitively.

    Allows:

        loo
        LOO
        BaseQuery
        basequery
        random
    """

    requested_type_lower = (
        requested_type
        .strip()
        .lower()
    )

    for (
        split_type,
        split_id,
    ) in splits.keys():

        if (
            split_type.lower()
            == requested_type_lower
            and split_id
            == requested_id
        ):

            return (
                split_type,
                split_id,
            )

    available = [
        f"{split_type}-{split_id}"
        for (
            split_type,
            split_id,
        )
        in splits.keys()
    ]

    raise ValueError(
        "Requested trace split "
        f"'{requested_type}-"
        f"{requested_id}' "
        f"does not exist. "
        f"Available: "
        f"{sorted(available)}"
    )


def trace_novelty(
    features: Dict[str, dict],
    splits: Dict[
        Tuple[str, int],
        Dict[str, Set[str]],
    ],
    trace_query: str,
    trace_split: str,
    trace_split_id: int,
    trace_detail: str,
) -> None:
    """
    Explain exactly how one Novelty Score
    is produced.
    """

    test_query = (
        trace_query
        .strip()
        .lower()
    )

    # ----------------------------------------
    # Find requested split
    # ----------------------------------------

    split_key = find_split_key(
        splits,
        trace_split,
        trace_split_id,
    )

    split_type, split_id = (
        split_key
    )

    split = splits[
        split_key
    ]

    train_queries = sorted(
        split[
            "train"
        ],
        key=query_sort_key,
    )

    test_queries = split[
        "test"
    ]

    # ----------------------------------------
    # Validation
    # ----------------------------------------

    if (
        test_query
        not in features
    ):
        raise ValueError(
            f"Unknown query: "
            f"{test_query}"
        )

    if (
        test_query
        not in test_queries
    ):

        role = (
            "TRAIN"
            if test_query
            in split["train"]
            else "NOT PRESENT"
        )

        raise ValueError(
            f"{test_query} is not a "
            f"Test query in "
            f"{split_type}-"
            f"{split_id}. "
            f"Current role: "
            f"{role}."
        )

    test_feature = (
        features[
            test_query
        ]
    )

    # ========================================================
    # TRACE HEADER
    # ========================================================

    print()
    print()
    print("#" * 72)
    print(
        "TRACE NOVELTY CALCULATION"
    )
    print("#" * 72)

    print(
        f"Test query       : "
        f"{test_query}"
    )

    print(
        f"Query family     : "
        f"{test_feature['family']}"
    )

    print(
        f"Split            : "
        f"{split_type}-"
        f"{split_id}"
    )

    print(
        f"Train candidates : "
        f"{len(train_queries)}"
    )

    print(
        f"Trace detail     : "
        f"{trace_detail}"
    )

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print("=" * 72)
    print(
        "STEP 1 - LOAD TEST QUERY FEATURES"
    )
    print("=" * 72)

    print_feature_set(
        "TABLES",
        test_feature[
            "tables"
        ],
    )

    print_feature_set(
        "JOINS",
        test_feature[
            "joins"
        ],
    )

    print_feature_set(
        "FILTERS",
        test_feature[
            "filters"
        ],
    )

    # ========================================================
    # STEP 2 - compute every Test-vs-Train distance
    # ========================================================

    print()
    print("=" * 72)
    print(
        "STEP 2 - COMPARE TEST QUERY "
        "WITH EVERY TRAIN QUERY"
    )
    print("=" * 72)

    candidates = []

    for train_query in (
        train_queries
    ):

        train_feature = (
            features[
                train_query
            ]
        )

        details = (
            query_distance_details(
                test_feature,
                train_feature,
            )
        )

        candidates.append(
            {
                "train_query":
                    train_query,

                "train_family":
                    train_feature[
                        "family"
                    ],

                "details":
                    details,

                "distance":
                    details[
                        "distance"
                    ],
            }
        )

    # Sort by distance for human-readable demo.
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate[
                "distance"
            ],
            query_sort_key(
                candidate[
                    "train_query"
                ]
            ),
        ),
    )

    print()
    print(
        "Rank | Train | Family | "
        "d_table  | d_join   | "
        "d_filter | D(q,h)"
    )

    print(
        "-" * 72
    )

    for rank, candidate in enumerate(
        ranked,
        start=1,
    ):

        details = (
            candidate[
                "details"
            ]
        )

        print(
            f"{rank:>4} | "
            f"{candidate['train_query']:<5} | "
            f"{candidate['train_family']:<6} | "
            f"{details['table']['distance']:>8.6f} | "
            f"{details['join']['distance']:>8.6f} | "
            f"{details['filter']['distance']:>8.6f} | "
            f"{details['distance']:>8.6f}"
        )

    # ----------------------------------------
    # Minimum
    # ----------------------------------------

    min_distance = (
        ranked[0][
            "distance"
        ]
    )

    tied = [
        candidate
        for candidate
        in ranked
        if abs(
            candidate[
                "distance"
            ]
            - min_distance
        )
        <= EPS
    ]

    tied.sort(
        key=lambda candidate:
            query_sort_key(
                candidate[
                    "train_query"
                ]
            )
    )

    nearest = tied[0]

    # ========================================================
    # STEP 3
    # ========================================================

    print()
    print("=" * 72)
    print(
        "STEP 3 - SHOW JACCARD "
        "CALCULATION"
    )
    print("=" * 72)

    if (
        trace_detail
        == "all"
    ):

        print()
        print(
            "Detailed calculation "
            "for ALL Train candidates."
        )

        for candidate in ranked:

            print_candidate_detail(
                test_query,
                candidate[
                    "train_query"
                ],
                candidate[
                    "details"
                ],
            )

    else:

        print()
        print(
            "For demo readability, "
            "full intersection/union "
            "is shown for the nearest "
            "Train query."
        )

        print(
            "Use "
            "--trace-detail all "
            "to print this derivation "
            "for every candidate."
        )

        print_candidate_detail(
            test_query,
            nearest[
                "train_query"
            ],
            nearest[
                "details"
            ],
        )

    # ========================================================
    # STEP 4 - MIN
    # ========================================================

    print()
    print("=" * 72)
    print(
        "STEP 4 - TAKE THE MINIMUM"
    )
    print("=" * 72)

    print()
    print(
        f"Novelty_"
        f"{split_type}"
        f"({test_query})"
    )

    print(
        "  = min "
        "{ D(q,h) | "
        f"h ∈ Train_{split_type} "
        "}"
    )

    print()

    print(
        f"Number of Train "
        f"candidates = "
        f"{len(train_queries)}"
    )

    print()

    print(
        "Smallest candidate distance:"
    )

    print(
        f"  D("
        f"{test_query}, "
        f"{nearest['train_query']}"
        f")"
    )

    print(
        "  = "
        f"{nearest['distance']:.6f}"
    )

    print()

    print(
        f"Nearest Train query : "
        f"{nearest['train_query']}"
    )

    print(
        f"Nearest family      : "
        f"{nearest['train_family']}"
    )

    if (
        len(tied)
        > 1
    ):

        print(
            f"Tie count           : "
            f"{len(tied)}"
        )

        print(
            "All nearest queries : "
            + ", ".join(
                candidate[
                    "train_query"
                ]
                for candidate
                in tied
            )
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("#" * 72)
    print(
        "FINAL NOVELTY RESULT"
    )
    print("#" * 72)

    print()

    print(
        f"Novelty_"
        f"{split_type}"
        f"({test_query}) "
        f"= "
        f"{min_distance:.6f}"
    )

    print()

    print(
        "Derivation:"
    )

    print(
        f"  nearest = "
        f"{nearest['train_query']}"
    )

    print(
        f"  d_table = "
        f"{nearest['details']['table']['distance']:.6f}"
    )

    print(
        f"  d_join = "
        f"{nearest['details']['join']['distance']:.6f}"
    )

    print(
        f"  d_filter = "
        f"{nearest['details']['filter']['distance']:.6f}"
    )

    print()

    print(
        "  Novelty"
    )

    print(
        "  = "
        "(d_table + "
        "d_join + "
        "d_filter) / 3"
    )

    print(
        "  = "
        f"("
        f"{nearest['details']['table']['distance']:.6f} "
        f"+ "
        f"{nearest['details']['join']['distance']:.6f} "
        f"+ "
        f"{nearest['details']['filter']['distance']:.6f}"
        f") / 3"
    )

    print(
        "  = "
        f"{min_distance:.6f}"
    )

    print()
    print("#" * 72)


# ============================================================
# Main
# ============================================================


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Compute structural Novelty "
            "Score for JOB test queries."
        )
    )

    parser.add_argument(
        "--splits",
        default=(
            r"analysis\data"
            r"\split_master.csv"
        ),
        help=(
            "Path to "
            "split_master.csv"
        ),
    )

    parser.add_argument(
        "--features",
        default=(
            r"analysis\data"
            r"\query_features.csv"
        ),
        help=(
            "Path to "
            "query_features.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            r"analysis\data"
            r"\novelty.csv"
        ),
        help=(
            "Output novelty CSV."
        ),
    )

    parser.add_argument(
        "--pairs-output",
        default=(
            r"analysis\data"
            r"\novelty_pairs.csv"
        ),
        help=(
            "Output all "
            "Test-vs-Train distances."
        ),
    )

    # ========================================================
    # Trace arguments
    # ========================================================

    parser.add_argument(
        "--trace-query",
        default=None,
        help=(
            "Show detailed Novelty "
            "calculation for one query. "
            "Example: q7c"
        ),
    )

    parser.add_argument(
        "--trace-split",
        default=None,
        help=(
            "Split used for trace mode. "
            "Example: LOO, Random, "
            "BaseQuery"
        ),
    )

    parser.add_argument(
        "--trace-split-id",
        type=int,
        default=1,
        help=(
            "Split ID used for trace "
            "mode. Default: 1"
        ),
    )

    parser.add_argument(
        "--trace-detail",
        choices=[
            "nearest",
            "all",
        ],
        default="nearest",
        help=(
            "nearest: print full "
            "Jaccard derivation only "
            "for nearest candidate. "
            "all: print full derivation "
            "for every Train candidate."
        ),
    )

    args = parser.parse_args()

    split_path = Path(
        args.splits
    )

    feature_path = Path(
        args.features
    )

    novelty_path = Path(
        args.output
    )

    pair_path = Path(
        args.pairs_output
    )

    try:

        # ------------------------------------
        # Input loading
        # ------------------------------------

        features = load_features(
            feature_path
        )

        splits = load_splits(
            split_path
        )

        print(
            f"Loaded features : "
            f"{len(features)}"
        )

        print(
            f"Loaded splits   : "
            f"{len(splits)}"
        )

        if (
            len(features)
            != 113
        ):
            raise RuntimeError(
                "Expected 113 JOB query "
                f"features, found "
                f"{len(features)}."
            )

        # ------------------------------------
        # Compute complete Novelty dataset
        # ------------------------------------

        (
            novelty_rows,
            pair_rows,
        ) = compute_novelty(
            features,
            splits,
        )

        # ------------------------------------
        # Write output
        # ------------------------------------

        write_novelty(
            novelty_path,
            novelty_rows,
        )

        write_pairs(
            pair_path,
            pair_rows,
        )

        # ------------------------------------
        # Summary
        # ------------------------------------

        print_summary(
            novelty_rows
        )

        print()

        print(
            f"Novelty rows    : "
            f"{len(novelty_rows)}"
        )

        print(
            f"Pair rows       : "
            f"{len(pair_rows)}"
        )

        print(
            f"Novelty output  : "
            f"{novelty_path}"
        )

        print(
            f"Pairs output    : "
            f"{pair_path}"
        )

        # ------------------------------------
        # Expected counts based directly
        # on split_master.csv
        # ------------------------------------

        expected_novelty_rows = sum(
            len(
                split[
                    "test"
                ]
            )
            for split
            in splits.values()
        )

        expected_pair_rows = sum(
            len(
                split[
                    "test"
                ]
            )
            *
            len(
                split[
                    "train"
                ]
            )
            for split
            in splits.values()
        )

        if (
            len(novelty_rows)
            != expected_novelty_rows
        ):

            raise RuntimeError(
                "Novelty-row count mismatch: "
                f"expected "
                f"{expected_novelty_rows}, "
                f"got "
                f"{len(novelty_rows)}."
            )

        if (
            len(pair_rows)
            != expected_pair_rows
        ):

            raise RuntimeError(
                "Pair-row count mismatch: "
                f"expected "
                f"{expected_pair_rows}, "
                f"got "
                f"{len(pair_rows)}."
            )

        print()

        print(
            "SUCCESS: Novelty computation "
            "completed."
        )

        # ====================================================
        # Optional trace/demo
        # ====================================================

        if (
            args.trace_query
            is not None
            or args.trace_split
            is not None
        ):

            if (
                args.trace_query
                is None
                or args.trace_split
                is None
            ):

                raise ValueError(
                    "Trace mode requires BOTH "
                    "--trace-query and "
                    "--trace-split."
                )

            trace_novelty(
                features=features,
                splits=splits,
                trace_query=(
                    args.trace_query
                ),
                trace_split=(
                    args.trace_split
                ),
                trace_split_id=(
                    args.trace_split_id
                ),
                trace_detail=(
                    args.trace_detail
                ),
            )

        return 0

    except Exception as exc:

        print(
            (
                "ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )