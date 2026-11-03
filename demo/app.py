from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Độ mới cấu trúc (Novelty) và khả năng khái quát hóa của LQO",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

# app.py:
#
# team14-lqo-bao-balsa-novelty/
# └── demo/
#     └── app.py
#
# parents[0] -> demo/
# parents[1] -> repository root

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    REPO_ROOT
    / "analysis"
    / "data"
)

SPLIT_MASTER_PATH = (
    DATA_DIR
    / "split_master.csv"
)

QUERY_FEATURES_PATH = (
    DATA_DIR
    / "query_features.csv"
)

NOVELTY_PATH = (
    DATA_DIR
    / "novelty.csv"
)


# ------------------------------------------------------------
# Bao analysis outputs
# ------------------------------------------------------------

BAO_PERFORMANCE_PATH = (
    DATA_DIR
    / "bao_novelty_performance.csv"
)

BAO_SPEARMAN_PATH = (
    DATA_DIR
    / "bao_spearman_summary.csv"
)

BAO_GROUP_PATH = (
    DATA_DIR
    / "bao_novelty_group_summary.csv"
)

BAO_COMPLEXITY_CORR_PATH = (
    DATA_DIR
    / "bao_complexity_correlations.csv"
)

BAO_COMPLEXITY_GROUP_PATH = (
    DATA_DIR
    / "bao_complexity_by_novelty_group.csv"
)

BAO_CASE_STUDIES_PATH = (
    DATA_DIR
    / "bao_case_studies.csv"
)


# ------------------------------------------------------------
# Balsa analysis outputs
# ------------------------------------------------------------

BALSA_PERFORMANCE_PATH = (
    DATA_DIR
    / "balsa_novelty_performance.csv"
)

BALSA_SPEARMAN_PATH = (
    DATA_DIR
    / "balsa_spearman_summary.csv"
)

BALSA_GROUP_PATH = (
    DATA_DIR
    / "balsa_novelty_group_summary.csv"
)

BALSA_CASE_STUDIES_PATH = (
    DATA_DIR
    / "balsa_case_studies.csv"
)


# ------------------------------------------------------------
# Canonical JOB SQL source
# ------------------------------------------------------------

# Prefer the copy already tracked in this repository.
# Keep the old local Bao path as a fallback for the author's machine.
JOB_SQL_CANDIDATES = [
    REPO_ROOT
    / "balsa"
    / "source"
    / "queries"
    / "join-order-benchmark",
    Path(
        r"D:\lqo_ml_perspective"
        r"\bao\queries"
        r"\join-order-benchmark"
    ),
]

JOB_SQL_DIR = next(
    (
        path
        for path in JOB_SQL_CANDIDATES
        if path.exists()
    ),
    JOB_SQL_CANDIDATES[0],
)


# ============================================================
# BASIC HELPERS
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


def parse_json_feature(
    value: str,
) -> Set[str]:
    """
    Parse a JSON array stored in query_features.csv
    into a Python set.
    """

    if pd.isna(value):
        return set()

    parsed = json.loads(
        str(value)
    )

    return set(parsed)


# ============================================================
# LOAD ORIGINAL SQL
# ============================================================


@st.cache_data
def load_query_sql(
    query_id: str,
) -> Tuple[str, str]:
    """
    Convert:

        q7c  -> 7c.sql
        q32b -> 32b.sql

    Returns:

        sql_text
        sql_path
    """

    query_id = (
        query_id
        .strip()
        .lower()
    )

    filename = (
        query_id.removeprefix("q")
        + ".sql"
    )

    sql_path = (
        JOB_SQL_DIR
        / filename
    )

    if not sql_path.exists():

        return (
            (
                "-- SQL file not found\n"
                f"-- Expected path: {sql_path}"
            ),
            str(sql_path),
        )

    sql = sql_path.read_text(
        encoding="utf-8"
    )

    return (
        sql,
        str(sql_path),
    )


# ============================================================
# JACCARD
# ============================================================


def jaccard_details(
    a: Set[str],
    b: Set[str],
) -> dict:
    """
    Jaccard similarity:

        J(A,B)
        = |A ∩ B| / |A ∪ B|

    Jaccard distance:

        d(A,B)
        = 1 - J(A,B)

    If both sets are empty:

        d(A,B) = 0
    """

    intersection = (
        a & b
    )

    union = (
        a | b
    )

    only_a = (
        a - b
    )

    only_b = (
        b - a
    )

    n_intersection = len(
        intersection
    )

    n_union = len(
        union
    )

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

        "only_a":
            only_a,

        "only_b":
            only_b,

        "n_a":
            len(a),

        "n_b":
            len(b),

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
    test_feature: dict,
    train_feature: dict,
) -> dict:
    """
    Compute:

        d_table
        d_join
        d_filter

    then:

        D(q,h)
        =
        (
            d_table
            + d_join
            + d_filter
        ) / 3
    """

    table = jaccard_details(
        test_feature[
            "tables"
        ],
        train_feature[
            "tables"
        ],
    )

    join = jaccard_details(
        test_feature[
            "joins"
        ],
        train_feature[
            "joins"
        ],
    )

    filter_ = jaccard_details(
        test_feature[
            "filters"
        ],
        train_feature[
            "filters"
        ],
    )

    distance = (
        table[
            "distance"
        ]
        +
        join[
            "distance"
        ]
        +
        filter_[
            "distance"
        ]
    ) / 3.0

    return {
        "table":
            table,

        "join":
            join,

        "filter":
            filter_,

        "distance":
            distance,
    }


# ============================================================
# DATA LOADING
# ============================================================


@st.cache_data
def load_split_master(
    path: str,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    df["query_id"] = (
        df["query_id"]
        .astype(str)
        .str.lower()
    )

    df["role"] = (
        df["role"]
        .astype(str)
        .str.lower()
    )

    df["split_id"] = (
        df["split_id"]
        .astype(int)
    )

    return df


@st.cache_data
def load_query_features(
    path: str,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    df["query_id"] = (
        df["query_id"]
        .astype(str)
        .str.lower()
    )

    return df


@st.cache_data
def load_novelty(
    path: str,
) -> pd.DataFrame | None:

    file_path = Path(
        path
    )

    if not file_path.exists():
        return None

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
    )

    df["query_id"] = (
        df["query_id"]
        .astype(str)
        .str.lower()
    )

    df["split_id"] = (
        df["split_id"]
        .astype(int)
    )

    return df


@st.cache_data
def load_optional_csv(
    path: str,
) -> pd.DataFrame | None:
    """
    Load an analysis CSV when it exists.

    Bao/Balsa result files are treated as optional so that the
    original Novelty demo can still start even when a later
    analysis stage has not been generated yet.
    """

    file_path = Path(
        path
    )

    if not file_path.exists():
        return None

    return pd.read_csv(
        file_path,
        encoding="utf-8-sig",
    )


def build_feature_map(
    feature_df: pd.DataFrame,
) -> Dict[str, dict]:
    """
    Convert query_features.csv into:

        feature_map["q7c"] = {
            "family": "q7",
            "tables": {...},
            "joins": {...},
            "filters": {...}
        }
    """

    result = {}

    for _, row in (
        feature_df.iterrows()
    ):

        query_id = (
            row[
                "query_id"
            ]
        )

        result[
            query_id
        ] = {
            "family":
                row[
                    "family"
                ],

            "tables":
                parse_json_feature(
                    row[
                        "tables"
                    ]
                ),

            "joins":
                parse_json_feature(
                    row[
                        "joins"
                    ]
                ),

            "filters":
                parse_json_feature(
                    row[
                        "filters"
                    ]
                ),

            "n_tables":
                int(
                    row[
                        "n_tables"
                    ]
                ),

            "n_joins":
                int(
                    row[
                        "n_joins"
                    ]
                ),

            "n_filters":
                int(
                    row[
                        "n_filters"
                    ]
                ),
        }

    return result


# ============================================================
# SPLIT HELPERS
# ============================================================


def get_split_queries(
    split_df: pd.DataFrame,
    split_type: str,
    split_id: int,
) -> Tuple[
    List[str],
    List[str],
]:
    """
    Return:

        train_queries
        test_queries
    """

    selected = split_df[
        (
            split_df[
                "split_type"
            ]
            == split_type
        )
        &
        (
            split_df[
                "split_id"
            ]
            == split_id
        )
    ]

    train_queries = sorted(
        selected[
            selected[
                "role"
            ]
            == "train"
        ][
            "query_id"
        ].tolist(),
        key=query_sort_key,
    )

    test_queries = sorted(
        selected[
            selected[
                "role"
            ]
            == "test"
        ][
            "query_id"
        ].tolist(),
        key=query_sort_key,
    )

    return (
        train_queries,
        test_queries,
    )


def get_test_splits_for_query(
    split_df: pd.DataFrame,
    query_id: str,
) -> List[
    Tuple[str, int]
]:
    """
    Return all splits in which query_id
    belongs to Test.
    """

    rows = split_df[
        (
            split_df[
                "query_id"
            ]
            == query_id
        )
        &
        (
            split_df[
                "role"
            ]
            == "test"
        )
    ]

    result = []

    for _, row in (
        rows.iterrows()
    ):

        result.append(
            (
                row[
                    "split_type"
                ],

                int(
                    row[
                        "split_id"
                    ]
                ),
            )
        )

    return result


# ============================================================
# NOVELTY CALCULATION
# ============================================================


def calculate_candidates(
    test_query: str,
    train_queries: List[str],
    feature_map: Dict[str, dict],
) -> Tuple[
    pd.DataFrame,
    Dict[str, dict],
]:
    """
    Calculate:

        D(q,h)

    for every:

        h ∈ Train
    """

    test_feature = (
        feature_map[
            test_query
        ]
    )

    records = []

    detail_map = {}

    for train_query in (
        train_queries
    ):

        train_feature = (
            feature_map[
                train_query
            ]
        )

        details = (
            query_distance_details(
                test_feature,
                train_feature,
            )
        )

        detail_map[
            train_query
        ] = details

        records.append(
            {
                "Train Query":
                    train_query,

                "Family":
                    train_feature[
                        "family"
                    ],

                "d_table":
                    details[
                        "table"
                    ][
                        "distance"
                    ],

                "d_join":
                    details[
                        "join"
                    ][
                        "distance"
                    ],

                "d_filter":
                    details[
                        "filter"
                    ][
                        "distance"
                    ],

                "D(q,h)":
                    details[
                        "distance"
                    ],
            }
        )

    # Sort:
    #
    # 1. smallest D(q,h)
    # 2. natural query ID order

    records = sorted(
        records,
        key=lambda row: (
            row[
                "D(q,h)"
            ],
            query_sort_key(
                row[
                    "Train Query"
                ]
            ),
        ),
    )

    for rank, record in enumerate(
        records,
        start=1,
    ):

        record[
            "Rank"
        ] = rank

    df = pd.DataFrame(
        records
    )

    df = df[
        [
            "Rank",
            "Train Query",
            "Family",
            "d_table",
            "d_join",
            "d_filter",
            "D(q,h)",
        ]
    ]

    return (
        df,
        detail_map,
    )


def calculate_novelty_for_split(
    query_id: str,
    split_type: str,
    split_id: int,
    split_df: pd.DataFrame,
    feature_map: Dict[str, dict],
) -> dict:
    """
    Compute Novelty for one Test Query
    under one split.
    """

    (
        train_queries,
        test_queries,
    ) = get_split_queries(
        split_df,
        split_type,
        split_id,
    )

    if (
        query_id
        not in test_queries
    ):

        raise ValueError(
            f"{query_id} is not Test in "
            f"{split_type}-{split_id}"
        )

    (
        candidate_df,
        detail_map,
    ) = calculate_candidates(
        query_id,
        train_queries,
        feature_map,
    )

    nearest_row = (
        candidate_df.iloc[0]
    )

    nearest_query = (
        nearest_row[
            "Train Query"
        ]
    )

    details = (
        detail_map[
            nearest_query
        ]
    )

    min_distance = float(
        nearest_row[
            "D(q,h)"
        ]
    )

    ties = candidate_df[
        (
            candidate_df[
                "D(q,h)"
            ]
            - min_distance
        ).abs()
        <= 1e-12
    ]

    return {
        "split_type":
            split_type,

        "split_id":
            split_id,

        "train_size":
            len(
                train_queries
            ),

        "test_size":
            len(
                test_queries
            ),

        "nearest":
            nearest_query,

        "nearest_family":
            feature_map[
                nearest_query
            ][
                "family"
            ],

        "novelty":
            min_distance,

        "d_table":
            details[
                "table"
            ][
                "distance"
            ],

        "d_join":
            details[
                "join"
            ][
                "distance"
            ],

        "d_filter":
            details[
                "filter"
            ][
                "distance"
            ],

        "tie_count":
            len(
                ties
            ),

        "candidate_df":
            candidate_df,

        "detail_map":
            detail_map,
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================


def feature_dataframe(
    values: Set[str],
) -> pd.DataFrame:

    return pd.DataFrame(
        {
            "Đặc trưng":
                sorted(
                    values
                )
        }
    )


def display_feature_column(
    title: str,
    values: Set[str],
) -> None:

    st.markdown(
        f"#### {title}"
    )

    st.metric(
        "Số lượng",
        len(values),
    )

    st.dataframe(
        feature_dataframe(
            values
        ),
        hide_index=True,
        width="stretch",
    )


def display_set(
    title: str,
    values: Set[str],
) -> None:

    st.markdown(
        f"**{title} ({len(values)})**"
    )

    if not values:

        st.code(
            "∅",
            language=None,
        )

        return

    st.dataframe(
        feature_dataframe(
            values
        ),
        hide_index=True,
        width="stretch",
    )


def display_jaccard_derivation(
    label: str,
    details: dict,
) -> None:
    """
    Display:

        intersection
        union
        Jaccard similarity
        Jaccard distance
    """

    st.markdown(
        f"### {label}"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "|A|",
        details[
            "n_a"
        ],
    )

    c2.metric(
        "|B|",
        details[
            "n_b"
        ],
    )

    c3.metric(
        "|A ∩ B|",
        details[
            "n_intersection"
        ],
    )

    c4.metric(
        "|A ∪ B|",
        details[
            "n_union"
        ],
    )

    intersection_col, union_col = (
        st.columns(2)
    )

    with intersection_col:

        display_set(
            "A ∩ B",
            details[
                "intersection"
            ],
        )

    with union_col:

        display_set(
            "A ∪ B",
            details[
                "union"
            ],
        )

    with st.expander(
        "Xem các đặc trưng khác nhau giữa hai truy vấn"
    ):

        left, right = (
            st.columns(2)
        )

        with left:

            display_set(
                "Chỉ có trong Test Query",
                details[
                    "only_a"
                ],
            )

        with right:

            display_set(
                "Chỉ có trong Train Query",
                details[
                    "only_b"
                ],
            )

    if (
        details[
            "n_union"
        ]
        == 0
    ):

        st.latex(
            r"A=B=\varnothing"
        )

        st.latex(
            r"d_J(A,B)=0"
        )

    else:

        n_intersection = (
            details[
                "n_intersection"
            ]
        )

        n_union = (
            details[
                "n_union"
            ]
        )

        similarity = (
            details[
                "similarity"
            ]
        )

        distance = (
            details[
                "distance"
            ]
        )

        st.markdown(
            "**Jaccard Similarity**"
        )

        st.latex(
            rf"""
            J(A,B)
            =
            \frac{{|A \cap B|}}
                 {{|A \cup B|}}
            =
            \frac{{{n_intersection}}}
                 {{{n_union}}}
            =
            {similarity:.6f}
            """
        )

        st.markdown(
            "**Jaccard Distance**"
        )

        st.latex(
            rf"""
            d_J(A,B)
            =
            1-J(A,B)
            =
            1-
            \frac{{{n_intersection}}}
                 {{{n_union}}}
            =
            {distance:.6f}
            """
        )


# ============================================================
# FILE VALIDATION
# ============================================================


missing_files = []

for file_path in [
    SPLIT_MASTER_PATH,
    QUERY_FEATURES_PATH,
]:

    if not file_path.exists():

        missing_files.append(
            str(
                file_path
            )
        )


if missing_files:

    st.error(
        "Thiếu file dữ liệu cần thiết."
    )

    st.code(
        "\n".join(
            missing_files
        )
    )

    st.stop()


if not JOB_SQL_DIR.exists():

    st.error(
        "Không tìm thấy thư mục canonical JOB SQL."
    )

    st.code(
        str(
            JOB_SQL_DIR
        )
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================


split_df = load_split_master(
    str(
        SPLIT_MASTER_PATH
    )
)

feature_df = load_query_features(
    str(
        QUERY_FEATURES_PATH
    )
)

novelty_df = load_novelty(
    str(
        NOVELTY_PATH
    )
)

bao_performance_df = load_optional_csv(
    str(BAO_PERFORMANCE_PATH)
)

bao_spearman_df = load_optional_csv(
    str(BAO_SPEARMAN_PATH)
)

bao_group_df = load_optional_csv(
    str(BAO_GROUP_PATH)
)

bao_complexity_corr_df = load_optional_csv(
    str(BAO_COMPLEXITY_CORR_PATH)
)

bao_complexity_group_df = load_optional_csv(
    str(BAO_COMPLEXITY_GROUP_PATH)
)

bao_case_studies_df = load_optional_csv(
    str(BAO_CASE_STUDIES_PATH)
)

balsa_performance_df = load_optional_csv(
    str(BALSA_PERFORMANCE_PATH)
)

balsa_spearman_df = load_optional_csv(
    str(BALSA_SPEARMAN_PATH)
)

balsa_group_df = load_optional_csv(
    str(BALSA_GROUP_PATH)
)

balsa_case_studies_df = load_optional_csv(
    str(BALSA_CASE_STUDIES_PATH)
)

feature_map = build_feature_map(
    feature_df
)



# ============================================================
# PRESENTATION-FOCUSED UI
# ============================================================


def get_comparable_split_rows(
    performance_df: pd.DataFrame | None,
    split_type: str,
    model_key: str,
    split_id: int | None = None,
) -> pd.DataFrame:
    """Return model/PostgreSQL rows valid for ratio analysis."""

    if performance_df is None:
        return pd.DataFrame()

    status_col = f"{model_key}_status"
    ratio_col = f"{model_key}_vs_pg_ratio"

    required = {
        "split_type",
        status_col,
        "pg_status",
        "novelty",
        ratio_col,
    }

    if not required.issubset(
        performance_df.columns
    ):
        return pd.DataFrame()

    mask = (
        (performance_df["split_type"] == split_type)
        & (performance_df[status_col] == "OK")
        & (performance_df["pg_status"] == "OK")
        & performance_df["novelty"].notna()
        & performance_df[ratio_col].notna()
    )

    if (
        split_id is not None
        and "split_id" in performance_df.columns
    ):
        mask &= (
            performance_df["split_id"]
            == split_id
        )

    return performance_df[
        mask
    ].copy()


def get_novelty_thresholds(
    split_type: str,
    split_id: int,
) -> Tuple[float | None, float | None]:
    """
    Shared split-level Novelty buckets:

        Low    <= q33
        Medium <= q67
        High   >  q67

    Important: thresholds come from novelty.csv, not from a model's
    comparable subset. Therefore Bao and Balsa use the same structural
    Novelty level for the same query.
    """

    if novelty_df is None:
        return None, None

    required = {
        "split_type",
        "split_id",
        "novelty",
    }

    if not required.issubset(
        novelty_df.columns
    ):
        return None, None

    rows = novelty_df[
        (
            novelty_df["split_type"]
            == split_type
        )
        & (
            novelty_df["split_id"]
            == split_id
        )
        & novelty_df["novelty"].notna()
    ]

    if rows.empty:
        return None, None

    q33 = float(
        rows["novelty"].quantile(1 / 3)
    )

    q67 = float(
        rows["novelty"].quantile(2 / 3)
    )

    return q33, q67


def novelty_level(
    value: float,
    q33: float | None,
    q67: float | None,
) -> str:

    if q33 is None or q67 is None:
        return "N/A"

    if value <= q33:
        return "Low"

    if value <= q67:
        return "Medium"

    return "High"


def novelty_level_label(
    level: str,
) -> str:

    mapping = {
        "Low": "THẤP",
        "Medium": "TRUNG BÌNH",
        "High": "CAO",
        "N/A": "CHƯA XÁC ĐỊNH",
    }

    return mapping.get(
        str(level),
        str(level),
    )


def correlation_strength_label(
    strength: str,
) -> str:

    mapping = {
        "negligible": "Không đáng kể",
        "very weak": "Rất yếu",
        "weak": "Yếu",
        "moderate": "Trung bình",
        "strong": "Mạnh",
        "very strong": "Rất mạnh",
        "not_available": "Chưa xác định",
        "insufficient_data": "Không đủ dữ liệu",
    }

    return mapping.get(
        str(strength).lower(),
        str(strength),
    )


def winner_label(
    winner: str,
) -> str:

    if str(winner) == "Not comparable":
        return "Không so sánh được"

    return str(winner)


def get_selected_performance_row(
    performance_df: pd.DataFrame | None,
    split_type: str,
    split_id: int,
    query_id: str,
) -> pd.Series | None:

    if performance_df is None:
        return None

    required = {
        "split_type",
        "split_id",
        "query_id",
    }

    if not required.issubset(
        performance_df.columns
    ):
        return None

    rows = performance_df[
        (
            performance_df["split_type"]
            == split_type
        )
        & (
            performance_df["split_id"]
            == split_id
        )
        & (
            performance_df["query_id"]
            .astype(str)
            .str.lower()
            == query_id.lower()
        )
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def get_spearman_row(
    spearman_df: pd.DataFrame | None,
    split_type: str,
) -> pd.Series | None:

    if spearman_df is None:
        return None

    key = (
        "split_type"
        if "split_type" in spearman_df.columns
        else "group"
        if "group" in spearman_df.columns
        else None
    )

    if key is None:
        return None

    rows = spearman_df[
        spearman_df[key]
        == split_type
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def get_case_study_labels(
    case_df: pd.DataFrame | None,
    split_type: str,
    query_id: str,
) -> List[str]:

    if case_df is None:
        return []

    required = {
        "split_type",
        "query_id",
        "case_type",
    }

    if not required.issubset(
        case_df.columns
    ):
        return []

    rows = case_df[
        (
            case_df["split_type"]
            == split_type
        )
        & (
            case_df["query_id"]
            .astype(str)
            .str.lower()
            == query_id.lower()
        )
    ]

    return (
        rows["case_type"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


def get_group_metrics(
    performance_df: pd.DataFrame | None,
    split_type: str,
    split_id: int,
    level: str,
    model_key: str,
    q33: float | None,
    q67: float | None,
) -> dict | None:
    """Build Low/Medium/High metrics with the shared Novelty thresholds."""

    if (
        performance_df is None
        or q33 is None
        or q67 is None
    ):
        return None

    required = {
        "split_type",
        "split_id",
        "novelty",
        f"{model_key}_status",
        "pg_status",
        f"{model_key}_vs_pg_ratio",
    }

    if not required.issubset(
        performance_df.columns
    ):
        return None

    rows = performance_df[
        (
            performance_df["split_type"]
            == split_type
        )
        & (
            performance_df["split_id"]
            == split_id
        )
    ].copy()

    if rows.empty:
        return None

    rows["novelty_group_ui"] = rows[
        "novelty"
    ].apply(
        lambda value: novelty_level(
            float(value),
            q33,
            q67,
        )
        if pd.notna(value)
        else "N/A"
    )

    group = rows[
        rows["novelty_group_ui"]
        == level
    ].copy()

    if group.empty:
        return None

    status_col = f"{model_key}_status"
    ratio_col = f"{model_key}_vs_pg_ratio"

    comparable = group[
        (group[status_col] == "OK")
        & (group["pg_status"] == "OK")
        & group[ratio_col].notna()
    ].copy()

    n = len(group)
    comparable_n = len(comparable)

    model_wins = int(
        (comparable[ratio_col] < 1).sum()
    )

    timeout_n = int(
        (group[status_col] == "TIMEOUT").sum()
    )

    return {
        "n": n,
        "comparable_n": comparable_n,
        "median_ratio": (
            float(comparable[ratio_col].median())
            if comparable_n > 0
            else None
        ),
        "win_rate": (
            model_wins / comparable_n
            if comparable_n > 0
            else None
        ),
        "timeouts": timeout_n,
        "timeout_rate": (
            timeout_n / n
            if n > 0
            else None
        ),
    }


def get_valid_ratio(
    row: pd.Series | None,
    model_key: str,
) -> float | None:

    if row is None:
        return None

    status = str(
        row.get(
            f"{model_key}_status",
            "N/A",
        )
    )

    pg_status = str(
        row.get(
            "pg_status",
            "N/A",
        )
    )

    ratio = row.get(
        f"{model_key}_vs_pg_ratio"
    )

    if (
        status != "OK"
        or pg_status != "OK"
        or pd.isna(ratio)
    ):
        return None

    return float(ratio)


def normalized_lqo_winner(
    bao_row: pd.Series | None,
    balsa_row: pd.Series | None,
) -> str:
    """Compare normalized model/PG ratios, not raw cross-pipeline latency."""

    bao_ratio = get_valid_ratio(
        bao_row,
        "bao",
    )

    balsa_ratio = get_valid_ratio(
        balsa_row,
        "balsa",
    )

    if (
        bao_ratio is None
        or balsa_ratio is None
    ):
        return "N/A"

    if abs(bao_ratio - balsa_ratio) <= 1e-12:
        return "Tương đương"

    return (
        "Bao"
        if bao_ratio < balsa_ratio
        else "Balsa"
    )


def format_ms(value) -> str:

    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):,.3f} ms"


def format_ratio(value) -> str:

    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.3f}×"


def render_spearman_summary(
    spearman_row: pd.Series | None,
    model_label: str,
) -> None:

    if spearman_row is None:
        st.warning(
            f"Không tìm thấy kết quả Spearman của {model_label}."
        )
        return

    rho = float(
        spearman_row["rho"]
    )

    p_value = float(
        spearman_row["p_value"]
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Spearman ρ",
        f"{rho:.3f}",
    )

    c2.metric(
        "p-value",
        f"{p_value:.3f}",
    )

    c3.metric(
        "Mức tương quan",
        correlation_strength_label(
            spearman_row.get(
                "strength",
                "N/A",
            )
        ),
    )

    if p_value < 0.05 and rho > 0:
        st.success(
            f"Trong `{selected_split_type}`, Novelty cao hơn đi cùng "
            f"{model_label}/PG ratio cao hơn với tín hiệu thống kê ở ngưỡng p < 0.05."
        )
    elif p_value < 0.05 and rho < 0:
        st.success(
            f"Trong `{selected_split_type}`, có tín hiệu thống kê cho quan hệ âm giữa "
            f"Novelty và {model_label}/PG ratio."
        )
    else:
        st.info(
            f"Trong `{selected_split_type}`, chưa có đủ bằng chứng thống kê tại ngưỡng "
            f"p < 0.05 cho quan hệ Novelty ↔ {model_label}/PG."
        )


def render_group_metrics(
    metrics: dict | None,
    model_label: str,
) -> None:

    if metrics is None:
        st.warning(
            f"Không đủ dữ liệu nhóm Novelty cho {model_label}."
        )
        return

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Số truy vấn",
        int(metrics["n"]),
    )

    c2.metric(
        "Có thể so sánh",
        int(metrics["comparable_n"]),
    )

    c3.metric(
        f"Trung vị {model_label}/PG",
        format_ratio(
            metrics["median_ratio"]
        ),
    )

    win_rate = metrics["win_rate"]

    c4.metric(
        f"Tỷ lệ {model_label} thắng",
        (
            f"{100 * win_rate:.1f}%"
            if win_rate is not None
            else "N/A"
        ),
    )

    if metrics["timeouts"] > 0:
        st.caption(
            f"Nhóm này có {metrics['timeouts']} timeout của {model_label} "
            f"({100 * metrics['timeout_rate']:.1f}% số truy vấn trong nhóm)."
        )


def render_scatter_plot(
    performance_df: pd.DataFrame | None,
    model_key: str,
    model_label: str,
) -> None:

    comparable = get_comparable_split_rows(
        performance_df,
        selected_split_type,
        model_key,
        selected_split_id,
    )

    if comparable.empty:
        st.warning(
            f"Không có đủ dữ liệu comparable để vẽ scatter cho {model_label}."
        )
        return

    ratio_col = f"{model_key}_vs_pg_ratio"

    scatter_df = comparable[
        [
            "query_id",
            "novelty",
            ratio_col,
        ]
    ].copy()

    scatter_df["Point"] = scatter_df[
        "query_id"
    ].apply(
        lambda query:
            "Truy vấn đang chọn"
            if str(query).lower()
            == selected_query.lower()
            else "Các truy vấn kiểm thử khác"
    )

    selected_filter = (
        "datum.Point == 'Truy vấn đang chọn'"
    )

    chart_spec = {
        "layer": [
            {
                "mark": {
                    "type": "rule",
                    "strokeDash": [6, 4],
                },
                "encoding": {
                    "y": {
                        "datum": 1,
                    },
                },
            },
            {
                "mark": {
                    "type": "point",
                    "filled": True,
                },
                "encoding": {
                    "x": {
                        "field": "novelty",
                        "type": "quantitative",
                        "title": "Novelty cấu trúc",
                    },
                    "y": {
                        "field": ratio_col,
                        "type": "quantitative",
                        "title": f"Tỷ lệ {model_label} / PostgreSQL",
                    },
                    "color": {
                        "field": "Point",
                        "type": "nominal",
                        "title": None,
                    },
                    "size": {
                        "condition": {
                            "test": selected_filter,
                            "value": 180,
                        },
                        "value": 70,
                    },
                    "tooltip": [
                        {
                            "field": "query_id",
                            "type": "nominal",
                            "title": "Truy vấn",
                        },
                        {
                            "field": "novelty",
                            "type": "quantitative",
                            "format": ".6f",
                            "title": "Novelty",
                        },
                        {
                            "field": ratio_col,
                            "type": "quantitative",
                            "format": ".3f",
                            "title": f"{model_label}/PG",
                        },
                    ],
                },
            },
            {
                "transform": [
                    {
                        "filter": selected_filter,
                    }
                ],
                "mark": {
                    "type": "text",
                    "dy": -14,
                    "fontWeight": "bold",
                },
                "encoding": {
                    "x": {
                        "field": "novelty",
                        "type": "quantitative",
                    },
                    "y": {
                        "field": ratio_col,
                        "type": "quantitative",
                    },
                    "text": {
                        "field": "query_id",
                    },
                },
            },
        ]
    }

    st.vega_lite_chart(
        scatter_df,
        chart_spec,
        use_container_width=True,
    )

    st.caption(
        f"Đường ngang y = 1: phía dưới là {model_label} nhanh hơn PostgreSQL; "
        "phía trên là PostgreSQL nhanh hơn."
    )



def get_dm_dataset_summary(
    performance_df: pd.DataFrame | None,
    model_key: str,
    split_type: str,
    split_id: int,
) -> dict | None:
    """Summarize the split-level dataset used for Data Mining."""

    if performance_df is None:
        return None

    required = {
        "split_type",
        "split_id",
        f"{model_key}_status",
        "pg_status",
        f"{model_key}_vs_pg_ratio",
    }

    if not required.issubset(performance_df.columns):
        return None

    rows = performance_df[
        (performance_df["split_type"] == split_type)
        & (performance_df["split_id"] == split_id)
    ].copy()

    if rows.empty:
        return None

    status_col = f"{model_key}_status"
    ratio_col = f"{model_key}_vs_pg_ratio"

    comparable = rows[
        (rows[status_col] == "OK")
        & (rows["pg_status"] == "OK")
        & rows[ratio_col].notna()
    ]

    return {
        "observations": len(rows),
        "comparable": len(comparable),
        "timeouts": int((rows[status_col] == "TIMEOUT").sum()),
        "non_ok_pg": int((rows["pg_status"] != "OK").sum()),
    }


def render_dm_dataset_summary(
    performance_df: pd.DataFrame | None,
    model_key: str,
    model_label: str,
) -> None:
    """DM1: show the dataset that feeds the mining steps."""

    summary = get_dm_dataset_summary(
        performance_df,
        model_key,
        selected_split_type,
        selected_split_id,
    )

    if summary is None:
        st.warning(f"Không đủ dữ liệu DM1 cho {model_label}.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quan sát", summary["observations"])
    c2.metric("So sánh hợp lệ", summary["comparable"])
    c3.metric(f"{model_label} timeout", summary["timeouts"])
    c4.metric("PG không hợp lệ", summary["non_ok_pg"])



def build_group_pattern_table(
    performance_df: pd.DataFrame | None,
    model_key: str,
    model_label: str,
) -> pd.DataFrame:
    """DM3: build Low/Medium/High metrics under the shared thresholds."""

    records = []

    for level in ["Low", "Medium", "High"]:
        metrics = get_group_metrics(
            performance_df,
            selected_split_type,
            selected_split_id,
            level,
            model_key,
            q33,
            q67,
        )

        if metrics is None:
            continue

        win_rate = metrics["win_rate"]
        timeout_rate = metrics["timeout_rate"]

        records.append(
            {
                "Nhóm Novelty": novelty_level_label(level),
                "n": int(metrics["n"]),
                "Hợp lệ": int(metrics["comparable_n"]),
                f"Trung vị {model_label}/PG": metrics["median_ratio"],
                f"Tỷ lệ {model_label} thắng (%)": (
                    100 * win_rate if win_rate is not None else None
                ),
                "Timeout rate (%)": (
                    100 * timeout_rate if timeout_rate is not None else None
                ),
            }
        )

    return pd.DataFrame(records)


def render_group_pattern_mining(
    performance_df: pd.DataFrame | None,
    model_key: str,
    model_label: str,
) -> None:
    """DM3: render all Novelty groups, not only the selected query's group."""

    table = build_group_pattern_table(
        performance_df,
        model_key,
        model_label,
    )

    if table.empty:
        st.warning(f"Không đủ dữ liệu DM3 cho {model_label}.")
        return

    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            f"Trung vị {model_label}/PG": st.column_config.NumberColumn(format="%.3f×"),
            f"Tỷ lệ {model_label} thắng (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Timeout rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    selected_row = table[
        table["Nhóm Novelty"] == novelty_level_label(selected_level)
    ]

    if not selected_row.empty:
        row = selected_row.iloc[0]
        st.info(
            f"Query `{selected_query}` thuộc nhóm **{novelty_level_label(selected_level)}**. "
            f"Trung vị {model_label}/PG của nhóm = "
            f"**{format_ratio(row[f'Trung vị {model_label}/PG'])}**."
        )


def case_type_label(value: str) -> str:
    mapping = {
        "HIGH_NOVELTY_WORST_SLOWDOWN": "Novelty cao · slowdown lớn nhất",
        "LOW_NOVELTY_WORST_SLOWDOWN": "Novelty thấp · slowdown lớn",
        "BALSA_TIMEOUT": "Balsa timeout",
        "BAO_TIMEOUT": "Bao timeout",
        "BEST_BALSA_RELATIVE_PERFORMANCE": "Balsa tốt nhất tương đối",
        "BEST_BAO_RELATIVE_PERFORMANCE": "Bao tốt nhất tương đối",
        "HIGH_NOVELTY_LOO": "Novelty cao · LOO",
        "HIGH_NOVELTY_RANDOM": "Novelty cao · Random",
        "HIGH_NOVELTY_BASEQUERY": "Novelty cao · BaseQuery",
    }
    return mapping.get(str(value), str(value).replace("_", " ").title())


def render_case_mining(
    case_df: pd.DataFrame | None,
    model_key: str,
    model_label: str,
) -> None:
    """DM4: show supportive/counterexample/timeout cases for the selected split."""

    if case_df is None:
        st.warning(f"Không tìm thấy dữ liệu case study của {model_label}.")
        return

    required = {"split_type", "query_id", "case_type"}
    if not required.issubset(case_df.columns):
        st.warning(f"Case-study file của {model_label} thiếu cột cần thiết.")
        return

    rows = case_df[
        case_df["split_type"] == selected_split_type
    ].copy()

    if "split_id" in rows.columns:
        rows = rows[
            rows["split_id"] == selected_split_id
        ]

    if rows.empty:
        st.info(f"Không có case study được gắn nhãn cho {model_label} trong split này.")
        return

    ratio_col = f"{model_key}_vs_pg_ratio"

    wanted = [
        "case_type",
        "query_id",
        "novelty",
        "novelty_group",
        f"{model_key}_status",
        ratio_col,
    ]
    display_cols = [c for c in wanted if c in rows.columns]

    display = rows[display_cols].copy()
    display["case_type"] = display["case_type"].apply(case_type_label)
    rename = {
        "case_type": "Loại case",
        "query_id": "Query",
        "novelty": "Novelty",
        "novelty_group": "Mức Novelty",
        f"{model_key}_status": "Status",
        ratio_col: f"{model_label}/PG",
    }
    display = display.rename(columns=rename)

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Novelty": st.column_config.NumberColumn(format="%.6f"),
            f"{model_label}/PG": st.column_config.NumberColumn(format="%.3f×"),
        },
    )

    selected_cases = rows[
        rows["query_id"].astype(str).str.lower() == selected_query.lower()
    ]

    if not selected_cases.empty:
        labels = [
            case_type_label(value)
            for value in selected_cases["case_type"].astype(str).drop_duplicates().tolist()
        ]
        st.success(
            f"`{selected_query}`: " + "; ".join(labels)
        )


def render_query_performance_card(
    performance_row: pd.Series | None,
    model_key: str,
    model_label: str,
    caveat: str | None = None,
) -> float | None:
    """Render the selected query's model-vs-PG result and return a valid ratio."""

    if performance_row is None:
        st.warning(f"Không tìm thấy dữ liệu {model_label} cho query đang chọn.")
        return None

    model_status = str(performance_row.get(f"{model_key}_status", "N/A"))
    pg_status = str(performance_row.get("pg_status", "N/A"))
    ratio = get_valid_ratio(performance_row, model_key)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        f"{model_label} E2E",
        format_ms(performance_row.get(f"{model_key}_end_to_end_ms")),
    )
    c2.metric(
        "PostgreSQL E2E",
        format_ms(performance_row.get("pg_end_to_end_ms")),
    )
    c3.metric(
        f"{model_label} / PG",
        format_ratio(performance_row.get(f"{model_key}_vs_pg_ratio")),
    )
    c4.metric(
        "Kết quả",
        winner_label(performance_row.get("winner", "N/A")),
    )

    if caveat:
        st.caption(caveat)

    if model_status != "OK" or pg_status != "OK":
        st.warning(
            f"{model_label} = `{model_status}`, PostgreSQL = `{pg_status}`. "
            "Query này không được dùng như một latency ratio hợp lệ."
        )
        return None

    if ratio is not None:
        if ratio < 1:
            st.success(f"{model_label} nhanh hơn PostgreSQL · **{ratio:.3f}×**")
        elif ratio > 1:
            st.warning(f"PostgreSQL nhanh hơn {model_label} · **{ratio:.3f}×**")
        else:
            st.info(f"{model_label} ≈ PostgreSQL")

    return ratio


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔎 Độ mới cấu trúc (Novelty) và khả năng khái quát hóa của LQO"
)

st.caption(
    "JOB · Cách chia dữ liệu số 1 chính thức · Bao và Balsa so với PostgreSQL"
)

st.caption(
    "Structural Novelty → hiệu năng tương đối của Bao/Balsa → phân tích Data Mining"
)


# ============================================================
# SIDEBAR — ONLY TWO CONTROLS
# ============================================================

st.sidebar.header(
    "⚙️ Điều khiển trình bày"
)

available_split_types = (
    split_df["split_type"]
    .drop_duplicates()
    .tolist()
)

preferred_order = [
    "Random",
    "LOO",
    "BaseQuery",
]

split_types = [
    item
    for item in preferred_order
    if item in available_split_types
]

split_types += [
    item
    for item in available_split_types
    if item not in split_types
]

selected_split_type = st.sidebar.selectbox(
    "1. Chọn kiểu chia dữ liệu",
    split_types,
)

available_split_ids = sorted(
    split_df[
        split_df["split_type"]
        == selected_split_type
    ]["split_id"]
    .unique()
    .tolist()
)

# Project analysis currently uses official Split-1.
selected_split_id = int(
    available_split_ids[0]
)

(
    train_queries,
    test_queries,
) = get_split_queries(
    split_df,
    selected_split_type,
    selected_split_id,
)

# Presentation-friendly defaults.
default_demo_queries = {
    "Random": "q6a",
    "LOO": "q11b",
    "BaseQuery": "q7a",
}

preferred_query = default_demo_queries.get(
    selected_split_type
)

if preferred_query in test_queries:
    default_query_index = test_queries.index(
        preferred_query
    )
else:
    default_query_index = 0

selected_query = st.sidebar.selectbox(
    "2. Chọn truy vấn kiểm thử",
    test_queries,
    index=default_query_index,
)

st.sidebar.caption(
    f"Cách chia chính thức: {selected_split_type}-{selected_split_id}"
)

st.sidebar.divider()

st.sidebar.markdown(
    "### Thông tin tập dữ liệu"
)

side_c1, side_c2 = st.sidebar.columns(2)

side_c1.metric(
    "Tập huấn luyện",
    len(train_queries),
)

side_c2.metric(
    "Tập kiểm thử",
    len(test_queries),
)

st.sidebar.metric(
    "Tổng số truy vấn JOB",
    len(feature_map),
)



# ============================================================
# CALCULATE SELECTED QUERY
# ============================================================

result = calculate_novelty_for_split(
    query_id=selected_query,
    split_type=selected_split_type,
    split_id=selected_split_id,
    split_df=split_df,
    feature_map=feature_map,
)

test_feature = feature_map[
    selected_query
]

query_sql, query_sql_path = load_query_sql(
    selected_query
)

bao_performance_row = get_selected_performance_row(
    bao_performance_df,
    selected_split_type,
    selected_split_id,
    selected_query,
)

balsa_performance_row = get_selected_performance_row(
    balsa_performance_df,
    selected_split_type,
    selected_split_id,
    selected_query,
)

q33, q67 = get_novelty_thresholds(
    selected_split_type,
    selected_split_id,
)

selected_level = novelty_level(
    float(result["novelty"]),
    q33,
    q67,
)

bao_spearman_row = get_spearman_row(
    bao_spearman_df,
    selected_split_type,
)

balsa_spearman_row = get_spearman_row(
    balsa_spearman_df,
    selected_split_type,
)

bao_group_metrics = get_group_metrics(
    bao_performance_df,
    selected_split_type,
    selected_split_id,
    selected_level,
    "bao",
    q33,
    q67,
)

balsa_group_metrics = get_group_metrics(
    balsa_performance_df,
    selected_split_type,
    selected_split_id,
    selected_level,
    "balsa",
    q33,
    q67,
)

bao_case_labels = get_case_study_labels(
    bao_case_studies_df,
    selected_split_type,
    selected_query,
)

balsa_case_labels = get_case_study_labels(
    balsa_case_studies_df,
    selected_split_type,
    selected_query,
)


# ============================================================
# QUERY SUMMARY
# ============================================================

st.divider()

st.subheader(
    f"Truy vấn kiểm thử `{selected_query}` — "
    f"{selected_split_type}-{selected_split_id}"
)

summary_1, summary_2, summary_3, summary_4 = (
    st.columns(4)
)

summary_1.metric(
    "Novelty cấu trúc",
    f"{result['novelty']:.6f}",
)

summary_2.metric(
    "Mức Novelty",
    novelty_level_label(selected_level),
)

summary_3.metric(
    "Truy vấn huấn luyện gần nhất",
    result["nearest"],
)

selected_lqo_ratio_winner = normalized_lqo_winner(
    bao_performance_row,
    balsa_performance_row,
)

summary_4.metric(
    "Normalized ratio thấp hơn",
    selected_lqo_ratio_winner,
)


# ============================================================
# STEP 1 — ORIGINAL SQL
# ============================================================

st.markdown(
    "## ① Truy vấn SQL gốc"
)

with st.expander(
    f"📄 Xem SQL gốc của {selected_query}",
    expanded=False,
):

    st.code(
        query_sql,
        language="sql",
    )



# ============================================================
# STEP 2 — QUERY STRUCTURE
# ============================================================

st.markdown(
    "## ② Cấu trúc truy vấn"
)

feature_m1, feature_m2, feature_m3 = (
    st.columns(3)
)

feature_m1.metric(
    "Bảng",
    test_feature["n_tables"],
)

feature_m2.metric(
    "Điều kiện nối",
    test_feature["n_joins"],
)

feature_m3.metric(
    "Điều kiện lọc",
    test_feature["n_filters"],
)

with st.expander("Xem chi tiết đặc trưng cấu trúc", expanded=False):
    feature_c1, feature_c2, feature_c3 = (
        st.columns(3)
    )

    with feature_c1:
        display_feature_column(
            "Bảng",
            test_feature["tables"],
        )

    with feature_c2:
        display_feature_column(
            "Điều kiện nối",
            test_feature["joins"],
        )

    with feature_c3:
        display_feature_column(
            "Điều kiện lọc",
            test_feature["filters"],
        )



# ============================================================
# STEP 3 — COMPARE WITH TRAIN
# ============================================================

st.markdown(
    "## ③ So sánh với tập huấn luyện"
)

st.latex(
    r"D(q,h)=\frac{d_{table}(q,h)+d_{join}(q,h)+d_{filter}(q,h)}{3}"
)

nearest_preview = (
    result["candidate_df"]
    .head(5)
    .copy()
    .rename(
        columns={
            "Rank": "Hạng",
            "Train Query": "Truy vấn huấn luyện",
            "Family": "Nhóm truy vấn",
        }
    )
)

st.markdown(
    "### 5 truy vấn huấn luyện gần nhất"
)

st.dataframe(
    nearest_preview,
    hide_index=True,
    width="stretch",
    column_config={
        "d_table": st.column_config.NumberColumn(
            format="%.6f"
        ),
        "d_join": st.column_config.NumberColumn(
            format="%.6f"
        ),
        "d_filter": st.column_config.NumberColumn(
            format="%.6f"
        ),
        "D(q,h)": st.column_config.NumberColumn(
            format="%.6f"
        ),
    },
)

st.success(
    f"Truy vấn huấn luyện gần nhất = **`{result['nearest']}`** "
    f"với khoảng cách nhỏ nhất **{result['novelty']:.6f}**."
)


# ============================================================
# STEP 4 — NOVELTY SCORE + LEVEL
# ============================================================

st.markdown(
    "## ④ Điểm Novelty cấu trúc"
)

nov_c1, nov_c2, nov_c3 = st.columns(3)

nov_c1.metric(
    "d_table",
    f"{result['d_table']:.6f}",
)

nov_c2.metric(
    "d_join",
    f"{result['d_join']:.6f}",
)

nov_c3.metric(
    "d_filter",
    f"{result['d_filter']:.6f}",
)

st.latex(
    rf"""
    Novelty({selected_query})
    =
    \min_{{h\in Train}}
    D({selected_query},h)
    =
    \frac{{
        {result['d_table']:.6f}
        +
        {result['d_join']:.6f}
        +
        {result['d_filter']:.6f}
    }}{{3}}
    =
    {result['novelty']:.6f}
    """
)

level_c1, level_c2 = st.columns(
    [1, 2]
)

with level_c1:
    st.metric(
        "Mức Novelty",
        novelty_level_label(selected_level),
    )

with level_c2:

    if q33 is not None and q67 is not None:

        threshold_c1, threshold_c2 = st.columns(2)
        threshold_c1.metric("Ngưỡng q33", f"{q33:.6f}")
        threshold_c2.metric("Ngưỡng q67", f"{q67:.6f}")

    else:
        st.warning(
            "Chưa đủ dữ liệu Novelty để xác định "
            "mức Low / Medium / High cho cách chia dữ liệu hiện tại."
        )


# ============================================================
# STEP 5 — LQO ANALYSIS TABS
# ============================================================

st.markdown(
    "## ⑤ Hiệu năng và Data Mining"
)

bao_ratio = get_valid_ratio(
    bao_performance_row,
    "bao",
)

balsa_ratio = get_valid_ratio(
    balsa_performance_row,
    "balsa",
)

bao_tab, balsa_tab = st.tabs(
    [
        "🔵 Bao",
        "🟠 Balsa",
    ]
)

with bao_tab:

    st.markdown(
        f"### Bao · `{selected_split_type}-{selected_split_id}` · `{selected_query}`"
    )

    bao_ratio = render_query_performance_card(
        bao_performance_row,
        "bao",
        "Bao",
        caveat="Author checkpoint · Team evaluation",
    )

    st.markdown("### Data Mining")
    dm1, dm2, dm3, dm4 = st.tabs(
        [
            "DM1 · Dữ liệu",
            "DM2 · Tương quan",
            "DM3 · Nhóm Novelty",
            "DM4 · Case study",
        ]
    )

    with dm1:
        render_dm_dataset_summary(
            bao_performance_df,
            "bao",
            "Bao",
        )

    with dm2:
        render_spearman_summary(
            bao_spearman_row,
            "Bao",
        )
        render_scatter_plot(
            bao_performance_df,
            "bao",
            "Bao",
        )

    with dm3:
        render_group_pattern_mining(
            bao_performance_df,
            "bao",
            "Bao",
        )

    with dm4:
        render_case_mining(
            bao_case_studies_df,
            "bao",
            "Bao",
        )

with balsa_tab:

    st.markdown(
        f"### Balsa · `{selected_split_type}-{selected_split_id}` · `{selected_query}`"
    )

    balsa_ratio = render_query_performance_card(
        balsa_performance_row,
        "balsa",
        "Balsa",
        caveat="Team-trained · Resource-constrained · 3 iterations",
    )

    st.markdown("### Data Mining")
    dm1, dm2, dm3, dm4 = st.tabs(
        [
            "DM1 · Dữ liệu",
            "DM2 · Tương quan",
            "DM3 · Nhóm Novelty",
            "DM4 · Case study",
        ]
    )

    with dm1:
        render_dm_dataset_summary(
            balsa_performance_df,
            "balsa",
            "Balsa",
        )

    with dm2:
        render_spearman_summary(
            balsa_spearman_row,
            "Balsa",
        )
        render_scatter_plot(
            balsa_performance_df,
            "balsa",
            "Balsa",
        )

    with dm3:
        render_group_pattern_mining(
            balsa_performance_df,
            "balsa",
            "Balsa",
        )

    with dm4:
        render_case_mining(
            balsa_case_studies_df,
            "balsa",
            "Balsa",
        )


# ============================================================
# STEP 6 — CROSS-LQO INTERPRETATION
# ============================================================

st.markdown(
    "## ⑥ Bao và Balsa trên cùng truy vấn"
)

compare_c1, compare_c2, compare_c3, compare_c4 = st.columns(4)

compare_c1.metric(
    "Novelty",
    f"{result['novelty']:.6f}",
)

compare_c2.metric(
    "Mức Novelty",
    novelty_level_label(selected_level),
)

compare_c3.metric(
    "Bao / PG",
    format_ratio(bao_ratio),
)

compare_c4.metric(
    "Balsa / PG",
    format_ratio(balsa_ratio),
)

st.caption(
    "So sánh theo tỷ lệ model/PG của từng pipeline; không phải benchmark raw latency trực tiếp Bao ↔ Balsa."
)

if bao_ratio is not None and balsa_ratio is not None:
    normalized_winner = normalized_lqo_winner(
        bao_performance_row,
        balsa_performance_row,
    )

    if normalized_winner == "Bao":
        st.success(
            f"Normalized ratio thấp hơn: **Bao** · {bao_ratio:.3f}× vs {balsa_ratio:.3f}×"
        )
    elif normalized_winner == "Balsa":
        st.success(
            f"Normalized ratio thấp hơn: **Balsa** · {balsa_ratio:.3f}× vs {bao_ratio:.3f}×"
        )
    else:
        st.info("Normalized ratio của hai LQO xấp xỉ nhau.")


# ============================================================
# STEP 7 — PRESENTATION TAKEAWAY
# ============================================================

st.divider()

st.markdown(
    "## 🎯 Kết luận"
)

st.markdown(
    """
- **Structural Novelty** định lượng mức khác biệt cấu trúc giữa test query và tập huấn luyện.
- **Data Mining** kiểm tra mối liên hệ Novelty ↔ performance bằng tương quan, phân nhóm và case study.
- Kết quả cho thấy Novelty là **một tín hiệu về generalization**, nhưng mức liên hệ phụ thuộc split và không đủ để suy ra quan hệ nhân quả.
"""
)

st.caption(
    "Balsa hiện dùng cấu hình team-trained, resource-constrained, 3 iterations."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Team14 — Novelty cấu trúc / Bao / Balsa / PostgreSQL"
)
