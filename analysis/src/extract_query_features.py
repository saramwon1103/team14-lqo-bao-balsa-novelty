"""
extract_query_features.py

Extract structural SQL features from JOB queries for Novelty analysis.

Input:
    Canonical JOB SQL files, e.g.
    D:\lqo_ml_perspective\bao\queries\join-order-benchmark

Output:
    analysis\data\query_features.csv

Features:
    - tables
    - joins
    - filters
    - n_tables
    - n_joins
    - n_filters

Design choices:
    1. SQL aliases are resolved back to base table names.
    2. Self-joins preserve repeated table references:
         aka_name AS an1, aka_name AS an2
       becomes:
         aka_name#1, aka_name#2
    3. Literal values are removed from filters:
         t.production_year > 2000
       becomes:
         title.production_year > <LITERAL>
    4. Both:
         JOIN ... ON ...
       and implicit JOB joins in WHERE are supported.
    5. Feature collections are serialized as JSON arrays in CSV so
       they can be safely loaded later for Jaccard distance.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------


def natural_query_key(path: Path) -> Tuple[int, str]:
    """
    Sort:
        1a.sql, 1b.sql, 2a.sql, 10a.sql
    instead of lexicographic:
        10a.sql, 1a.sql, ...
    """

    match = re.fullmatch(
        r"(\d+)([a-z]+)",
        path.stem.lower(),
    )

    if not match:
        return (999999, path.stem.lower())

    return int(match.group(1)), match.group(2)


def normalize_name(value: str) -> str:
    """Normalize SQL identifier."""
    return value.strip().lower()


def query_id_from_path(path: Path) -> str:
    """
    7c.sql  -> q7c
    15a.sql -> q15a
    """
    return f"q{path.stem.lower()}"


def family_from_query_id(query_id: str) -> str:
    """
    q7c  -> q7
    q15a -> q15
    """

    match = re.fullmatch(
        r"q(\d+)[a-z]+",
        query_id,
    )

    if not match:
        raise ValueError(
            f"Invalid JOB query id: {query_id}"
        )

    return f"q{match.group(1)}"


# ---------------------------------------------------------------------
# Table / alias handling
# ---------------------------------------------------------------------


def build_table_metadata(
    tree: exp.Expression,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Build canonical table labels and alias mapping.

    Example:

        title AS t
        movie_info AS mi

    becomes:

        tables:
            ["movie_info", "title"]

        alias_map:
            {
                "t": "title",
                "mi": "movie_info"
            }

    Self-join:

        aka_name AS an1,
        aka_name AS an2

    becomes:

        ["aka_name#1", "aka_name#2"]

    and:

        an1 -> aka_name#1
        an2 -> aka_name#2
    """

    table_nodes = list(
        tree.find_all(exp.Table)
    )

    base_names = [
        normalize_name(table.name)
        for table in table_nodes
        if table.name
    ]

    counts = Counter(base_names)
    occurrence = defaultdict(int)

    table_labels: List[str] = []
    alias_map: Dict[str, str] = {}

    for table in table_nodes:

        if not table.name:
            continue

        base = normalize_name(table.name)

        occurrence[base] += 1

        if counts[base] > 1:
            label = (
                f"{base}"
                f"#{occurrence[base]}"
            )
        else:
            label = base

        table_labels.append(label)

        alias = table.alias_or_name

        if alias:
            alias_map[
                normalize_name(alias)
            ] = label

        # Mapping physical table name itself is safe
        # only when there is no self-join ambiguity.
        if counts[base] == 1:
            alias_map[base] = label

    return (
        sorted(set(table_labels)),
        alias_map,
    )


def resolve_column(
    column: exp.Column,
    alias_map: Dict[str, str],
) -> str:
    """
    Resolve:
        t.id -> title.id

    For self-join:
        an1.id -> aka_name#1.id
    """

    column_name = normalize_name(
        column.name
    )

    qualifier = column.table

    if qualifier:

        qualifier = normalize_name(
            qualifier
        )

        relation = alias_map.get(
            qualifier,
            qualifier,
        )

        return (
            f"{relation}."
            f"{column_name}"
        )

    # Keep unqualified column rather than guessing.
    return column_name


# ---------------------------------------------------------------------
# Predicate handling
# ---------------------------------------------------------------------


COMPARISON_OPERATORS = {
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.LT: "<",
    exp.LTE: "<=",
}


def unwrap_parens(
    node: exp.Expression,
) -> exp.Expression:

    while isinstance(
        node,
        exp.Paren,
    ):
        node = node.this

    return node


def split_conjunction(
    expression: Optional[
        exp.Expression
    ],
) -> List[exp.Expression]:
    """
    Split:

        A AND B AND C

    into:

        [A, B, C]

    OR is kept intact as one logical filter group.
    """

    if expression is None:
        return []

    expression = unwrap_parens(
        expression
    )

    if isinstance(
        expression,
        exp.And,
    ):
        return (
            split_conjunction(
                expression.left
            )
            +
            split_conjunction(
                expression.right
            )
        )

    return [expression]


def get_relation_of_column(
    column: exp.Column,
    alias_map: Dict[str, str],
) -> Optional[str]:

    qualifier = column.table

    if not qualifier:
        return None

    qualifier = normalize_name(
        qualifier
    )

    return alias_map.get(
        qualifier,
        qualifier,
    )


def comparison_operator(
    node: exp.Expression,
) -> Optional[str]:

    for cls, symbol in (
        COMPARISON_OPERATORS.items()
    ):
        if isinstance(node, cls):
            return symbol

    return None


def is_join_predicate(
    predicate: exp.Expression,
    alias_map: Dict[str, str],
) -> bool:
    """
    Treat comparison between columns from
    different relation references as a join.

    Example:
        t.id = mi.movie_id

    Self-join is handled because:
        an1.id -> aka_name#1.id
        an2.id -> aka_name#2.id
    """

    predicate = unwrap_parens(
        predicate
    )

    if (
        comparison_operator(predicate)
        is None
    ):
        return False

    left = unwrap_parens(
        predicate.left
    )

    right = unwrap_parens(
        predicate.right
    )

    if not isinstance(
        left,
        exp.Column,
    ):
        return False

    if not isinstance(
        right,
        exp.Column,
    ):
        return False

    left_relation = (
        get_relation_of_column(
            left,
            alias_map,
        )
    )

    right_relation = (
        get_relation_of_column(
            right,
            alias_map,
        )
    )

    if (
        not left_relation
        or not right_relation
    ):
        return False

    return (
        left_relation
        != right_relation
    )


def normalize_join(
    predicate: exp.Expression,
    alias_map: Dict[str, str],
) -> str:
    """
    Canonicalize join predicate.

    Equality joins are treated as symmetric:

        t.id = mi.movie_id

    and:

        mi.movie_id = t.id

    become the same feature.
    """

    predicate = unwrap_parens(
        predicate
    )

    operator = comparison_operator(
        predicate
    )

    if operator is None:
        raise ValueError(
            "Unsupported join predicate: "
            f"{predicate.sql()}"
        )

    left = resolve_column(
        predicate.left,
        alias_map,
    )

    right = resolve_column(
        predicate.right,
        alias_map,
    )

    if operator in {"=", "!="}:

        left, right = sorted(
            [left, right]
        )

    return (
        f"{left} "
        f"{operator} "
        f"{right}"
    )


# ---------------------------------------------------------------------
# Filter normalization
# ---------------------------------------------------------------------


def is_literal_like(
    node: exp.Expression,
) -> bool:

    node = unwrap_parens(node)

    return isinstance(
        node,
        (
            exp.Literal,
            exp.Null,
            exp.Boolean,
        ),
    )


def invert_operator(
    operator: str,
) -> str:

    return {
        "=": "=",
        "!=": "!=",
        ">": "<",
        ">=": "<=",
        "<": ">",
        "<=": ">=",
    }.get(
        operator,
        operator,
    )


def normalize_operand(
    node: exp.Expression,
    alias_map: Dict[str, str],
) -> str:
    """
    Convert one expression operand to
    structural representation.
    """

    node = unwrap_parens(node)

    # -----------------------------
    # Column
    # -----------------------------

    if isinstance(
        node,
        exp.Column,
    ):
        return resolve_column(
            node,
            alias_map,
        )

    # -----------------------------
    # Literal
    # -----------------------------

    if isinstance(
        node,
        exp.Literal,
    ):
        return "<LITERAL>"

    # -----------------------------
    # NULL
    # -----------------------------

    # NULL is kept explicitly because:
    # IS NULL and IS NOT NULL
    # are structural predicates.
    if isinstance(
        node,
        exp.Null,
    ):
        return "NULL"

    # -----------------------------
    # Boolean
    # -----------------------------

    if isinstance(
        node,
        exp.Boolean,
    ):
        return "<BOOLEAN>"

    # -----------------------------
    # Function
    # -----------------------------

    if isinstance(
        node,
        exp.Func,
    ):

        args = [
            normalize_operand(
                arg,
                alias_map,
            )
            for arg
            in node.iter_expressions()
        ]

        function_name = (
            node.sql_name().upper()
        )

        return (
            f"{function_name}"
            f"({', '.join(args)})"
        )

    # -----------------------------
    # Fallback
    # -----------------------------

    text = node.sql(
        dialect="postgres",
        pretty=False,
    )

    # Hide string literals.
    text = re.sub(
        r"'(?:''|[^'])*'",
        "<LITERAL>",
        text,
    )

    # Hide numeric literals.
    text = re.sub(
        (
            r"(?<![\w.])"
            r"-?\d+(?:\.\d+)?"
            r"(?![\w.])"
        ),
        "<LITERAL>",
        text,
    )

    return " ".join(
        text.split()
    )


def normalize_filter(
    predicate: exp.Expression,
    alias_map: Dict[str, str],
) -> str:
    """
    Produce structural filter feature.

    Literal VALUES are discarded.

    Examples:

        t.production_year > 2000

    becomes:

        title.production_year > <LITERAL>


        an.name IS NOT NULL

    becomes:

        aka_name.name IS NOT NULL
    """

    predicate = unwrap_parens(
        predicate
    )

    # -------------------------------------------------
    # AND
    # -------------------------------------------------

    if isinstance(
        predicate,
        exp.And,
    ):

        parts = sorted(
            normalize_filter(
                part,
                alias_map,
            )
            for part
            in split_conjunction(
                predicate
            )
        )

        return (
            "("
            + " AND ".join(parts)
            + ")"
        )

    # -------------------------------------------------
    # OR
    # -------------------------------------------------

    if isinstance(
        predicate,
        exp.Or,
    ):

        left = normalize_filter(
            predicate.left,
            alias_map,
        )

        right = normalize_filter(
            predicate.right,
            alias_map,
        )

        # OR is commutative for structural matching.
        parts = sorted(
            [left, right]
        )

        return (
            f"({parts[0]} "
            f"OR {parts[1]})"
        )

    # -------------------------------------------------
    # NOT
    # -------------------------------------------------

    if isinstance(
        predicate,
        exp.Not,
    ):

        return (
            "NOT ("
            + normalize_filter(
                predicate.this,
                alias_map,
            )
            + ")"
        )

    # -------------------------------------------------
    # Standard comparisons
    # -------------------------------------------------

    operator = comparison_operator(
        predicate
    )

    if operator is not None:

        left_node = unwrap_parens(
            predicate.left
        )

        right_node = unwrap_parens(
            predicate.right
        )

        # Normalize:
        #
        #   2000 < t.production_year
        #
        # into:
        #
        #   title.production_year > <LITERAL>
        #
        if (
            is_literal_like(
                left_node
            )
            and isinstance(
                right_node,
                exp.Column,
            )
        ):

            left_node, right_node = (
                right_node,
                left_node,
            )

            operator = (
                invert_operator(
                    operator
                )
            )

        left = normalize_operand(
            left_node,
            alias_map,
        )

        right = normalize_operand(
            right_node,
            alias_map,
        )

        return (
            f"{left} "
            f"{operator} "
            f"{right}"
        )

    # -------------------------------------------------
    # LIKE
    # -------------------------------------------------

    if isinstance(
        predicate,
        exp.Like,
    ):

        left = normalize_operand(
            predicate.this,
            alias_map,
        )

        right = normalize_operand(
            predicate.expression,
            alias_map,
        )

        return (
            f"{left} "
            f"LIKE "
            f"{right}"
        )

    # -------------------------------------------------
    # ILIKE
    # -------------------------------------------------

    if (
        hasattr(exp, "ILike")
        and isinstance(
            predicate,
            exp.ILike,
        )
    ):

        left = normalize_operand(
            predicate.this,
            alias_map,
        )

        right = normalize_operand(
            predicate.expression,
            alias_map,
        )

        return (
            f"{left} "
            f"ILIKE "
            f"{right}"
        )

    # -------------------------------------------------
    # IN (...)
    # -------------------------------------------------

    if isinstance(
        predicate,
        exp.In,
    ):

        left = normalize_operand(
            predicate.this,
            alias_map,
        )

        expressions = (
            predicate.expressions
            or []
        )

        if (
            expressions
            and all(
                is_literal_like(item)
                for item
                in expressions
            )
        ):

            return (
                f"{left} "
                f"IN "
                f"(<LITERAL_SET>)"
            )

        if expressions:

            normalized_values = sorted(
                normalize_operand(
                    item,
                    alias_map,
                )
                for item
                in expressions
            )

            return (
                f"{left} "
                f"IN "
                f"("
                f"{', '.join(normalized_values)}"
                f")"
            )

        query = predicate.args.get(
            "query"
        )

        if query is not None:
            return (
                f"{left} "
                f"IN "
                f"(<SUBQUERY>)"
            )

        return (
            f"{left} "
            f"IN "
            f"(<UNKNOWN>)"
        )

    # -------------------------------------------------
    # BETWEEN
    # -------------------------------------------------

    if isinstance(
        predicate,
        exp.Between,
    ):

        target = normalize_operand(
            predicate.this,
            alias_map,
        )

        low = normalize_operand(
            predicate.args["low"],
            alias_map,
        )

        high = normalize_operand(
            predicate.args["high"],
            alias_map,
        )

        return (
            f"{target} "
            f"BETWEEN "
            f"{low} "
            f"AND "
            f"{high}"
        )

    # -------------------------------------------------
    # IS / IS NOT
    # -------------------------------------------------

    if isinstance(
        predicate,
        exp.Is,
    ):

        left = normalize_operand(
            predicate.this,
            alias_map,
        )

        right = normalize_operand(
            predicate.expression,
            alias_map,
        )

        # SQLGlot may represent:
        #
        #   IS NOT NULL
        #
        # using the "negate" argument.
        #
        operator = (
            "IS NOT"
            if predicate.args.get(
                "negate"
            )
            else "IS"
        )

        return (
            f"{left} "
            f"{operator} "
            f"{right}"
        )

    # -------------------------------------------------
    # Parentheses fallback
    # -------------------------------------------------

    if isinstance(
        predicate,
        exp.Paren,
    ):

        return (
            "("
            + normalize_filter(
                predicate.this,
                alias_map,
            )
            + ")"
        )

    # -------------------------------------------------
    # Generic fallback
    # -------------------------------------------------

    text = predicate.sql(
        dialect="postgres",
        pretty=False,
    )

    # Hide string literals.
    text = re.sub(
        r"'(?:''|[^'])*'",
        "<LITERAL>",
        text,
    )

    # Hide numeric literals.
    text = re.sub(
        (
            r"(?<![\w.])"
            r"-?\d+(?:\.\d+)?"
            r"(?![\w.])"
        ),
        "<LITERAL>",
        text,
    )

    return " ".join(
        text.split()
    )


# ---------------------------------------------------------------------
# Predicate collection
# ---------------------------------------------------------------------


def collect_predicates(
    tree: exp.Expression,
) -> List[exp.Expression]:
    """
    Collect predicates from:

        WHERE ...
        JOIN ... ON ...

    JOB commonly expresses joins inside WHERE,
    so WHERE predicates are essential.
    """

    predicates: List[
        exp.Expression
    ] = []

    # -----------------------------
    # WHERE
    # -----------------------------

    for where in tree.find_all(
        exp.Where
    ):

        predicates.extend(
            split_conjunction(
                where.this
            )
        )

    # -----------------------------
    # Explicit JOIN ... ON
    # -----------------------------

    for join in tree.find_all(
        exp.Join
    ):

        on_expression = (
            join.args.get("on")
        )

        if (
            on_expression
            is not None
        ):

            predicates.extend(
                split_conjunction(
                    on_expression
                )
            )

    return predicates


# ---------------------------------------------------------------------
# One-query extraction
# ---------------------------------------------------------------------


def extract_features(
    sql: str,
    dialect: str = "postgres",
) -> Tuple[
    List[str],
    List[str],
    List[str],
]:
    """
    Parse one SQL statement and return:

        tables
        joins
        filters
    """

    tree = sqlglot.parse_one(
        sql,
        read=dialect,
    )

    tables, alias_map = (
        build_table_metadata(tree)
    )

    joins: Set[str] = set()
    filters: Set[str] = set()

    for predicate in (
        collect_predicates(tree)
    ):

        predicate = unwrap_parens(
            predicate
        )

        if is_join_predicate(
            predicate,
            alias_map,
        ):

            joins.add(
                normalize_join(
                    predicate,
                    alias_map,
                )
            )

        else:

            filters.add(
                normalize_filter(
                    predicate,
                    alias_map,
                )
            )

    return (
        sorted(tables),
        sorted(joins),
        sorted(filters),
    )


# ---------------------------------------------------------------------
# Dataset extraction
# ---------------------------------------------------------------------


def process_query(
    path: Path,
    dialect: str,
) -> dict:

    query_id = query_id_from_path(
        path
    )

    try:

        sql = path.read_text(
            encoding="utf-8",
        )

        (
            tables,
            joins,
            filters,
        ) = extract_features(
            sql,
            dialect=dialect,
        )

        return {
            "query_id":
                query_id,

            "family":
                family_from_query_id(
                    query_id
                ),

            "tables":
                json.dumps(
                    tables,
                    ensure_ascii=False,
                ),

            "joins":
                json.dumps(
                    joins,
                    ensure_ascii=False,
                ),

            "filters":
                json.dumps(
                    filters,
                    ensure_ascii=False,
                ),

            "n_tables":
                len(tables),

            "n_joins":
                len(joins),

            "n_filters":
                len(filters),

            "parse_status":
                "OK",

            "parse_error":
                "",

            "sql_file":
                path.name,
        }

    except Exception as exc:

        return {
            "query_id":
                query_id,

            "family":
                family_from_query_id(
                    query_id
                ),

            "tables":
                "[]",

            "joins":
                "[]",

            "filters":
                "[]",

            "n_tables":
                0,

            "n_joins":
                0,

            "n_filters":
                0,

            "parse_status":
                "ERROR",

            "parse_error":
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),

            "sql_file":
                path.name,
        }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Extract structural JOB query "
            "features for Novelty analysis."
        )
    )

    parser.add_argument(
        "--sql-dir",
        default=(
            r"D:\lqo_ml_perspective"
            r"\bao\queries"
            r"\join-order-benchmark"
        ),
        help=(
            "Directory containing "
            "canonical JOB *.sql files."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            r"analysis\data"
            r"\query_features.csv"
        ),
        help="Output CSV path.",
    )

    parser.add_argument(
        "--dialect",
        default="postgres",
        help="SQLGlot input dialect.",
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=113,
        help=(
            "Expected number "
            "of JOB SQL files."
        ),
    )

    args = parser.parse_args()

    sql_dir = Path(
        args.sql_dir
    )

    output = Path(
        args.output
    )

    if not sql_dir.exists():

        print(
            (
                "ERROR: SQL directory "
                "does not exist: "
                f"{sql_dir}"
            ),
            file=sys.stderr,
        )

        return 1

    query_files = sorted(
        [
            p
            for p
            in sql_dir.glob("*.sql")
            if re.fullmatch(
                r"\d+[a-z]+\.sql",
                p.name.lower(),
            )
        ],
        key=natural_query_key,
    )

    print(
        f"SQL directory : "
        f"{sql_dir}"
    )

    print(
        f"JOB SQL files : "
        f"{len(query_files)}"
    )

    if (
        len(query_files)
        != args.expected_count
    ):

        print(
            (
                "ERROR: expected "
                f"{args.expected_count} "
                "SQL files, found "
                f"{len(query_files)}."
            ),
            file=sys.stderr,
        )

        return 1

    rows = []

    for (
        index,
        query_path,
    ) in enumerate(
        query_files,
        start=1,
    ):

        row = process_query(
            query_path,
            dialect=args.dialect,
        )

        rows.append(row)

        print(
            f"[{index:03d}/"
            f"{len(query_files)}] "
            f"{row['query_id']}: "
            f"{row['parse_status']} | "
            f"tables="
            f"{row['n_tables']} | "
            f"joins="
            f"{row['n_joins']} | "
            f"filters="
            f"{row['n_filters']}"
        )

        if (
            row["parse_status"]
            != "OK"
        ):

            print(
                "    "
                + row[
                    "parse_error"
                ]
            )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "query_id",
        "family",
        "tables",
        "joins",
        "filters",
        "n_tables",
        "n_joins",
        "n_filters",
        "parse_status",
        "parse_error",
        "sql_file",
    ]

    with output.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    ok_count = sum(
        row["parse_status"]
        == "OK"
        for row
        in rows
    )

    error_count = (
        len(rows)
        - ok_count
    )

    print()
    print("=" * 60)

    print(
        "FEATURE EXTRACTION SUMMARY"
    )

    print("=" * 60)

    print(
        f"Queries total : "
        f"{len(rows)}"
    )

    print(
        f"Parsed OK     : "
        f"{ok_count}"
    )

    print(
        f"Errors        : "
        f"{error_count}"
    )

    print(
        f"Output        : "
        f"{output}"
    )

    print("=" * 60)

    if error_count:

        print()

        print(
            "WARNING: Some queries failed "
            "to parse. Do not compute "
            "Novelty until they are "
            "inspected."
        )

        return 2

    print()

    print(
        "SUCCESS: all JOB queries "
        "were parsed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )