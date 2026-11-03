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
# Canonical JOB SQL source
# ------------------------------------------------------------

JOB_SQL_DIR = Path(
    r"D:\lqo_ml_perspective"
    r"\bao\queries"
    r"\join-order-benchmark"
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

feature_map = build_feature_map(
    feature_df
)



# ============================================================
# PRESENTATION-FOCUSED UI
# ============================================================


def get_comparable_split_rows(
    performance_df: pd.DataFrame | None,
    split_type: str,
) -> pd.DataFrame:
    """Return Bao/PostgreSQL rows that are valid for ratio analysis."""

    if performance_df is None:
        return pd.DataFrame()

    required = {
        "split_type",
        "bao_status",
        "pg_status",
        "novelty",
        "bao_vs_pg_ratio",
    }

    if not required.issubset(
        performance_df.columns
    ):
        return pd.DataFrame()

    return performance_df[
        (
            performance_df["split_type"]
            == split_type
        )
        & (
            performance_df["bao_status"]
            == "OK"
        )
        & (
            performance_df["pg_status"]
            == "OK"
        )
        & performance_df["novelty"].notna()
        & performance_df["bao_vs_pg_ratio"].notna()
    ].copy()


def get_novelty_thresholds(
    split_type: str,
) -> Tuple[float | None, float | None]:
    """
    Reproduce DM2 grouping:

        Low    <= q33
        Medium <= q67
        High   >  q67

    Thresholds are computed separately inside each split.
    """

    comparable = get_comparable_split_rows(
        bao_performance_df,
        split_type,
    )

    if comparable.empty:
        return None, None

    q33 = float(
        comparable["novelty"].quantile(1 / 3)
    )

    q67 = float(
        comparable["novelty"].quantile(2 / 3)
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
        "weak": "Yếu",
        "moderate": "Trung bình",
        "strong": "Mạnh",
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
    split_type: str,
    split_id: int,
    query_id: str,
) -> pd.Series | None:

    if bao_performance_df is None:
        return None

    rows = bao_performance_df[
        (
            bao_performance_df["split_type"]
            == split_type
        )
        & (
            bao_performance_df["split_id"]
            == split_id
        )
        & (
            bao_performance_df["query_id"]
            .astype(str)
            .str.lower()
            == query_id.lower()
        )
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def get_spearman_row(
    split_type: str,
) -> pd.Series | None:

    if bao_spearman_df is None:
        return None

    key = (
        "split_type"
        if "split_type" in bao_spearman_df.columns
        else "group"
    )

    rows = bao_spearman_df[
        bao_spearman_df[key]
        == split_type
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def get_group_summary_row(
    split_type: str,
    level: str,
) -> pd.Series | None:

    if bao_group_df is None:
        return None

    rows = bao_group_df[
        (
            bao_group_df["split_type"]
            == split_type
        )
        & (
            bao_group_df["novelty_group"]
            == level
        )
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def format_ms(value) -> str:

    if pd.isna(value):
        return "N/A"

    return f"{float(value):,.3f} ms"


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔎 Độ mới cấu trúc (Novelty) và khả năng khái quát hóa của LQO"
)

st.caption(
    "JOB · Cách chia dữ liệu số 1 chính thức · Bao so với PostgreSQL"
)

st.markdown(
    """
**Hướng mở rộng của nhóm:** từ một truy vấn kiểm thử, đo truy vấn đó **khác tập huấn luyện đến mức nào**
bằng Novelty cấu trúc, sau đó đối chiếu mức Novelty với **hiệu năng của Bao so với PostgreSQL**.

**Luồng trình bày:** `SQL → Đặc trưng cấu trúc → Khoảng cách tới tập huấn luyện → Mức Novelty → Bao so với PostgreSQL → Nhận xét về khả năng khái quát hóa`
"""
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

st.sidebar.info(
    "Đổi kiểu chia dữ liệu hoặc truy vấn kiểm thử để toàn bộ quá trình "
    "tính Novelty và hiệu năng cập nhật theo truy vấn đó."
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

performance_row = get_selected_performance_row(
    selected_split_type,
    selected_split_id,
    selected_query,
)

q33, q67 = get_novelty_thresholds(
    selected_split_type
)

selected_level = novelty_level(
    float(result["novelty"]),
    q33,
    q67,
)

spearman_row = get_spearman_row(
    selected_split_type
)

group_summary_row = get_group_summary_row(
    selected_split_type,
    selected_level,
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

if performance_row is not None:
    selected_winner = str(
        performance_row.get(
            "winner",
            "N/A",
        )
    )
else:
    selected_winner = "N/A"

summary_4.metric(
    "Bộ tối ưu tốt hơn",
    winner_label(selected_winner),
)


# ============================================================
# STEP 1 — ORIGINAL SQL
# ============================================================

st.markdown(
    "## ① Truy vấn SQL gốc"
)

st.markdown(
    f"Truy vấn **`{selected_query}`** thuộc tập kiểm thử của "
    f"**{selected_split_type}-{selected_split_id}**. "
    "Đây là SQL gốc trước khi nhóm trích các đặc trưng cấu trúc."
)

with st.expander(
    f"📄 Xem SQL gốc của {selected_query}",
    expanded=False,
):

    st.code(
        query_sql,
        language="sql",
    )

    st.caption(
        f"Nguồn: {query_sql_path}"
    )


# ============================================================
# STEP 2 — QUERY STRUCTURE
# ============================================================

st.markdown(
    "## ② Cấu trúc truy vấn"
)

st.markdown(
    "Từ SQL gốc, nhóm trích ba nhóm đặc trưng cấu trúc. "
    "Các giá trị hằng (literal) được chuẩn hóa nên Novelty tập trung vào "
    "**cấu trúc truy vấn**, không phải giá trị literal cụ thể."
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

st.info(
    "Các giá trị hằng được chuẩn hóa thành `<LITERAL>` hoặc "
    "`<LITERAL_SET>`. Vì vậy điểm hiện tại phản ánh "
    "**Novelty cấu trúc**."
)


# ============================================================
# STEP 3 — COMPARE WITH TRAIN
# ============================================================

st.markdown(
    "## ③ So sánh với tập huấn luyện"
)

st.markdown(
    f"`{selected_query}` được so sánh với **{result['train_size']} truy vấn trong tập huấn luyện** "
    f"trong `{selected_split_type}-{selected_split_id}`. "
    "Với mỗi cặp kiểm thử–huấn luyện, nhóm tính ba khoảng cách Jaccard rồi lấy trung bình."
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

        st.markdown(
            f"""
**Ngưỡng phân nhóm trong `{selected_split_type}`**

- **Thấp:** Novelty ≤ `{q33:.6f}`
- **Trung bình:** `{q33:.6f}` < Novelty ≤ `{q67:.6f}`
- **Cao:** Novelty > `{q67:.6f}`

`{selected_query}` = **{result['novelty']:.6f} → Novelty {novelty_level_label(selected_level)}**
"""
        )

        st.caption(
            "Ba mức Thấp / Trung bình / Cao được xác định riêng trong từng cách chia dữ liệu "
            "theo cùng quy tắc đã dùng ở DM2."
        )

    else:
        st.warning(
            "Chưa đủ dữ liệu hiệu năng của Bao để xác định "
            "mức Novelty theo cách chia dữ liệu hiện tại."
        )


# ============================================================
# STEP 5 — BAO VS POSTGRESQL
# ============================================================

st.markdown(
    "## ⑤ So sánh hiệu năng Bao và PostgreSQL"
)

st.latex(
    r"R_{Bao}(q)=\frac{T_{Bao}(q)}{T_{PostgreSQL}(q)}"
)

ratio_info_1, ratio_info_2 = st.columns(2)

ratio_info_1.info(
    "**Tỷ lệ < 1** → Bao nhanh hơn"
)

ratio_info_2.info(
    "**Tỷ lệ > 1** → PostgreSQL nhanh hơn"
)

if performance_row is None:

    st.warning(
        "Không tìm thấy dữ liệu hiệu năng của truy vấn này trong "
        "bao_novelty_performance.csv."
    )

else:

    bao_status = str(
        performance_row.get(
            "bao_status",
            "N/A",
        )
    )

    pg_status = str(
        performance_row.get(
            "pg_status",
            "N/A",
        )
    )

    perf_c1, perf_c2, perf_c3, perf_c4 = (
        st.columns(4)
    )

    perf_c1.metric(
        "Tổng thời gian Bao",
        format_ms(
            performance_row.get(
                "bao_end_to_end_ms"
            )
        ),
    )

    perf_c2.metric(
        "Tổng thời gian PostgreSQL",
        format_ms(
            performance_row.get(
                "pg_end_to_end_ms"
            )
        ),
    )

    ratio_value = performance_row.get(
        "bao_vs_pg_ratio"
    )

    if pd.notna(ratio_value):
        ratio_label = f"{float(ratio_value):.3f}×"
    else:
        ratio_label = "N/A"

    perf_c3.metric(
        "Bao / PostgreSQL",
        ratio_label,
    )

    perf_c4.metric(
        "Bộ tối ưu tốt hơn",
        winner_label(
            performance_row.get(
                "winner",
                "N/A",
            )
        ),
    )

    if (
        bao_status != "OK"
        or pg_status != "OK"
    ):

        st.warning(
            f"Không thể so sánh truy vấn này: trạng thái Bao = `{bao_status}`, "
            f"trạng thái PostgreSQL = `{pg_status}`. "
            "Timeout được giữ riêng và không thay bằng thời gian chạy giả."
        )

    elif pd.notna(ratio_value):

        ratio_value = float(
            ratio_value
        )

        if ratio_value < 1:
            st.success(
                f"Với `{selected_query}`, Bao nhanh hơn PostgreSQL: "
                f"Bao/PG = **{ratio_value:.3f}**."
            )
        elif ratio_value > 1:
            st.warning(
                f"Với `{selected_query}`, PostgreSQL nhanh hơn Bao: "
                f"Bao/PG = **{ratio_value:.3f}**."
            )
        else:
            st.info(
                "Bao và PostgreSQL có tỷ lệ thời gian chạy xấp xỉ 1."
            )


# ============================================================
# STEP 6 — INTERPRETATION FOR THIS QUERY
# ============================================================

st.markdown(
    "## ⑥ Truy vấn này cho thấy điều gì?"
)

if (
    performance_row is not None
    and pd.notna(
        performance_row.get(
            "bao_vs_pg_ratio"
        )
    )
    and str(
        performance_row.get(
            "bao_status"
        )
    ) == "OK"
    and str(
        performance_row.get(
            "pg_status"
        )
    ) == "OK"
):

    ratio_value = float(
        performance_row[
            "bao_vs_pg_ratio"
        ]
    )

    winner = str(
        performance_row[
            "winner"
        ]
    )

    st.markdown(
        f"""
### Quan sát ở mức truy vấn

`{selected_query}` thuộc nhóm Novelty **{novelty_level_label(selected_level)}** trong `{selected_split_type}-{selected_split_id}`.

- Novelty cấu trúc = **{result['novelty']:.6f}**
- Tỷ lệ Bao / PostgreSQL = **{ratio_value:.3f}**
- Bộ tối ưu tốt hơn cho truy vấn này = **{winner_label(winner)}**
"""
    )

    if selected_level == "High" and ratio_value > 1:
        st.info(
            "Trường hợp này cho thấy **Novelty cao đi kèm với sự suy giảm "
            "tương đối của Bao**. Đây là mối liên hệ quan sát được ở truy vấn này, "
            "không phải bằng chứng rằng Novelty gây ra hiệu năng kém."
        )

    elif selected_level == "High" and ratio_value < 1:
        st.info(
            "Đây là một phản ví dụ quan trọng: truy vấn có **Novelty cao** "
            "nhưng Bao vẫn tốt hơn PostgreSQL. Vì vậy Novelty cao "
            "không đồng nghĩa Bao chắc chắn sẽ thua."
        )

    elif selected_level == "Low" and ratio_value > 1:
        st.info(
            "Đây cũng là một phản ví dụ hữu ích: **Novelty thấp** "
            "không đảm bảo Bao sẽ hoạt động tốt."
        )

    else:
        st.info(
            "Hiệu năng của truy vấn này cần được đọc cùng với xu hướng "
            "của toàn bộ cách chia dữ liệu; Novelty không phải yếu tố duy nhất."
        )

else:
    st.info(
        "Truy vấn này chưa có tỷ lệ thời gian chạy hợp lệ để tạo "
        "nhận xét hiệu năng ở mức truy vấn."
    )


# ============================================================
# STEP 7 — SPLIT-LEVEL CONTEXT
# ============================================================

st.markdown(
    "## ⑦ Trường hợp này có phù hợp với xu hướng của toàn bộ cách chia dữ liệu không?"
)

st.markdown(
    "Sau khi phân tích một truy vấn cụ thể, nhóm mở rộng phạm vi để kiểm tra "
    "truy vấn đó có phù hợp với xu hướng của toàn bộ cách chia dữ liệu hay không."
)

if spearman_row is not None:

    rho = float(
        spearman_row["rho"]
    )

    p_value = float(
        spearman_row["p_value"]
    )

    stat_c1, stat_c2, stat_c3 = st.columns(3)

    stat_c1.metric(
        "Spearman ρ",
        f"{rho:.3f}",
    )

    stat_c2.metric(
        "p-value",
        f"{p_value:.3f}",
    )

    stat_c3.metric(
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
            "Ở mức toàn bộ cách chia dữ liệu có tín hiệu thống kê: Novelty cao hơn đi cùng "
            "Bao/PG ratio cao hơn, tức Bao có xu hướng mất lợi thế "
            "tương đối so với PostgreSQL."
        )
    elif p_value < 0.05 and rho < 0:
        st.success(
            "Ở mức toàn bộ cách chia dữ liệu có tín hiệu thống kê cho quan hệ âm giữa "
            "Novelty và Bao/PG ratio."
        )
    else:
        st.info(
            "Ở mức toàn bộ cách chia dữ liệu chưa có đủ bằng chứng thống kê tại ngưỡng "
            "p < 0.05. Không nên biến chiều của rho thành kết luận mạnh."
        )

else:
    st.warning(
        "Không tìm thấy kết quả Spearman cho cách chia dữ liệu hiện tại."
    )


if group_summary_row is not None:

    st.markdown(
        f"### Nhóm Novelty {novelty_level_label(selected_level)} trong cách chia {selected_split_type}"
    )

    group_c1, group_c2, group_c3 = (
        st.columns(3)
    )

    group_c1.metric(
        "Số truy vấn",
        int(
            group_summary_row["n"]
        ),
    )

    group_c2.metric(
        "Trung vị tỷ lệ Bao/PG",
        f"{float(group_summary_row['median_bao_pg_ratio']):.3f}",
    )

    group_c3.metric(
        "Tỷ lệ Bao thắng",
        f"{100 * float(group_summary_row['bao_win_rate']):.1f}%",
    )


# ------------------------------------------------------------
# Scatter plot: selected query inside its split
# ------------------------------------------------------------

comparable_split = get_comparable_split_rows(
    bao_performance_df,
    selected_split_type,
)

if not comparable_split.empty:

    st.markdown(
        "### Vị trí truy vấn đang chọn trong toàn bộ cách chia dữ liệu"
    )

    scatter_df = comparable_split[
        [
            "query_id",
            "novelty",
            "bao_vs_pg_ratio",
        ]
    ].copy()

    scatter_df["Point"] = scatter_df[
        "query_id"
    ].apply(
        lambda q:
            "Truy vấn đang chọn"
            if str(q).lower()
            == selected_query.lower()
            else "Các truy vấn kiểm thử khác"
    )

    # Vega-Lite lets us keep the important ratio=1 reference line
    # without introducing an extra plotting dependency.
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
                        "field": "bao_vs_pg_ratio",
                        "type": "quantitative",
                        "title": "Tỷ lệ thời gian chạy Bao / PostgreSQL",
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
                            "field": "bao_vs_pg_ratio",
                            "type": "quantitative",
                            "format": ".3f",
                            "title": "Bao/PG",
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
                        "field": "bao_vs_pg_ratio",
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
        "Đường ngang y = 1: phía dưới là Bao nhanh hơn; "
        "phía trên là PostgreSQL nhanh hơn."
    )


# ============================================================
# PRESENTATION TAKEAWAY
# ============================================================

st.divider()

st.markdown(
    "## 🎯 Kết luận chính"
)

st.markdown(
    """
1. **Novelty cấu trúc** cho biết truy vấn kiểm thử khác tập huấn luyện đến mức nào về bảng, điều kiện nối và điều kiện lọc.  
2. Ta dùng chính truy vấn đó để kiểm tra **Bao hay PostgreSQL cho hiệu năng tốt hơn** thông qua tỷ lệ thời gian chạy.  
3. Sau đó đặt truy vấn vào **bối cảnh của toàn bộ cách chia dữ liệu** bằng Spearman và các nhóm Novelty.  
4. **Novelty là một tín hiệu về khả năng khái quát hóa, không phải nguyên nhân duy nhất quyết định hiệu năng của bộ tối ưu truy vấn.**
"""
)

st.caption(
    "Bước tiếp theo của đồ án: áp dụng cùng quy trình cho Balsa "
    "trên LOO-1, Random-1 và BaseQuery-1, rồi so sánh Bao vs Balsa."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Team14 — Novelty cấu trúc / Bao / Balsa / PostgreSQL"
)
