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
    page_title="LQO Novelty Score Explorer",
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
            "Feature":
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
        "Count",
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
        "Xem các feature khác nhau giữa hai query"
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

feature_map = build_feature_map(
    feature_df
)


# ============================================================
# HEADER
# ============================================================


st.title(
    "🔎 LQO Novelty Score Explorer"
)

st.caption(
    "JOB / Bao-Balsa Novelty Analysis"
)

st.markdown(
    """
Ứng dụng minh họa từng bước cách tính **Structural Novelty Score**
của một Test Query so với tập Train tương ứng.

Novelty hiện tại dựa trên ba nhóm đặc trưng:

- Tables
- Join predicates
- Filter predicates

Literal values được chuẩn hóa thành `<LITERAL>` hoặc
`<LITERAL_SET>`, vì vậy Novelty tập trung vào
**cấu trúc của truy vấn SQL**.
"""
)


# ============================================================
# SIDEBAR
# ============================================================


st.sidebar.header(
    "⚙️ Demo Controls"
)


preferred_order = [
    "LOO",
    "Random",
    "BaseQuery",
]


available_split_types = (
    split_df[
        "split_type"
    ]
    .drop_duplicates()
    .tolist()
)


split_types = [
    split_type
    for split_type
    in preferred_order
    if split_type
    in available_split_types
]


split_types += [
    split_type
    for split_type
    in available_split_types
    if split_type
    not in split_types
]


selected_split_type = (
    st.sidebar.selectbox(
        "1. Chọn Split",
        split_types,
    )
)


available_split_ids = sorted(
    split_df[
        split_df[
            "split_type"
        ]
        == selected_split_type
    ][
        "split_id"
    ]
    .unique()
    .tolist()
)


selected_split_id = (
    st.sidebar.selectbox(
        "2. Chọn Split ID",
        available_split_ids,
    )
)


(
    train_queries,
    test_queries,
) = get_split_queries(
    split_df,
    selected_split_type,
    selected_split_id,
)


# Default q7c when available because
# this is a useful same-query case study.

default_query_index = 0

if "q7c" in test_queries:

    default_query_index = (
        test_queries.index(
            "q7c"
        )
    )


selected_query = (
    st.sidebar.selectbox(
        "3. Chọn Test Query",
        test_queries,
        index=default_query_index,
    )
)


st.sidebar.divider()

st.sidebar.markdown(
    "### Split Information"
)

st.sidebar.metric(
    "Train Queries",
    len(
        train_queries
    ),
)

st.sidebar.metric(
    "Test Queries",
    len(
        test_queries
    ),
)

st.sidebar.metric(
    "JOB Universe",
    len(
        feature_map
    ),
)


# ============================================================
# CALCULATE SELECTED QUERY
# ============================================================


result = (
    calculate_novelty_for_split(
        query_id=
            selected_query,

        split_type=
            selected_split_type,

        split_id=
            selected_split_id,

        split_df=
            split_df,

        feature_map=
            feature_map,
    )
)


test_feature = (
    feature_map[
        selected_query
    ]
)


query_sql, query_sql_path = (
    load_query_sql(
        selected_query
    )
)


# ============================================================
# TOP SUMMARY
# ============================================================


st.subheader(
    f"Query {selected_query} — "
    f"{selected_split_type}-"
    f"{selected_split_id}"
)


m1, m2, m3, m4 = (
    st.columns(4)
)


m1.metric(
    "Novelty",
    f"{result['novelty']:.6f}",
)


m2.metric(
    "Nearest Train Query",
    result[
        "nearest"
    ],
)


m3.metric(
    "Nearest Family",
    result[
        "nearest_family"
    ],
)


m4.metric(
    "Train Candidates",
    result[
        "train_size"
    ],
)


# ============================================================
# TABS
# ============================================================


(
    tab_overview,
    tab_features,
    tab_candidates,
    tab_jaccard,
    tab_compare,
) = st.tabs(
    [
        "① Overview",
        "② Query Features",
        "③ Train Comparison",
        "④ Jaccard Derivation",
        "⑤ Compare Splits",
    ]
)


# ============================================================
# TAB 1 - OVERVIEW
# ============================================================


with tab_overview:

    st.header(
        "① Novelty Pipeline"
    )

    st.markdown(
        f"""
**Test Query:** `{selected_query}`

**Split:** `{selected_split_type}-{selected_split_id}`

**Train size:** `{result["train_size"]}`
"""
    )

    # ========================================================
    # ORIGINAL SQL
    # ========================================================

    st.markdown(
        "### SQL Query gốc"
    )

    st.caption(
        "Truy vấn được đọc trực tiếp từ "
        "canonical JOB workload."
    )

    with st.expander(
        f"📄 Xem SQL gốc của {selected_query}",
        expanded=True,
    ):

        st.code(
            query_sql,
            language="sql",
        )

        st.caption(
            f"Source: {query_sql_path}"
        )


    st.markdown(
        """
SQL trên chính là đầu vào được dùng để trích ba nhóm
đặc trưng **Tables**, **Joins** và **Filters**.
"""
    )


    # ========================================================
    # STEP 1
    # ========================================================

    st.markdown(
        "### Bước 1 — Trích đặc trưng cấu trúc"
    )

    f1, f2, f3 = (
        st.columns(3)
    )

    f1.metric(
        "Tables",
        test_feature[
            "n_tables"
        ],
    )

    f2.metric(
        "Join Predicates",
        test_feature[
            "n_joins"
        ],
    )

    f3.metric(
        "Filter Predicates",
        test_feature[
            "n_filters"
        ],
    )


    # ========================================================
    # STEP 2
    # ========================================================

    st.markdown(
        "### Bước 2 — So sánh với Train Set"
    )

    st.write(
        f"`{selected_query}` được so sánh với "
        f"**{result['train_size']} Train Queries** "
        f"thuộc `{selected_split_type}-{selected_split_id}`."
    )

    st.markdown(
        "Khoảng cách Jaccard cho từng tập đặc trưng:"
    )

    st.latex(
        r"""
        d_J(A,B)
        =
        1-
        \frac{|A\cap B|}
             {|A\cup B|}
        """
    )

    st.markdown(
        "Khoảng cách tổng giữa Test Query "
        "và một Train Query:"
    )

    st.latex(
        r"""
        D(q,h)
        =
        \frac{
        d_{table}(q,h)
        +
        d_{join}(q,h)
        +
        d_{filter}(q,h)
        }{3}
        """
    )


    # ========================================================
    # STEP 3
    # ========================================================

    st.markdown(
        "### Bước 3 — Tìm Train Query gần nhất"
    )

    st.latex(
        rf"""
        Novelty_{{{selected_split_type}}}
        ({selected_query})
        =
        \min_{{h\in Train_{{{selected_split_type}}}}}
        D({selected_query},h)
        """
    )

    st.write(
        f"Nearest Train Query tìm được là "
        f"**`{result['nearest']}`**."
    )


    # ========================================================
    # STEP 4
    # ========================================================

    st.markdown(
        "### Bước 4 — Novelty Score"
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "d_table",
        f"{result['d_table']:.6f}",
    )

    c2.metric(
        "d_join",
        f"{result['d_join']:.6f}",
    )

    c3.metric(
        "d_filter",
        f"{result['d_filter']:.6f}",
    )


    st.latex(
        rf"""
        Novelty
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


    st.success(
        f"Novelty_{selected_split_type}"
        f"({selected_query}) = "
        f"{result['novelty']:.6f}"
    )


    # ========================================================
    # CHECK WITH BATCH PIPELINE
    # ========================================================

    if novelty_df is not None:

        batch_match = novelty_df[
            (
                novelty_df[
                    "query_id"
                ]
                == selected_query
            )
            &
            (
                novelty_df[
                    "split_type"
                ]
                == selected_split_type
            )
            &
            (
                novelty_df[
                    "split_id"
                ]
                == selected_split_id
            )
        ]


        if not batch_match.empty:

            batch_value = float(
                batch_match.iloc[0][
                    "novelty"
                ]
            )

            difference = abs(
                batch_value
                - result[
                    "novelty"
                ]
            )


            with st.expander(
                "✅ Đối chiếu với novelty.csv"
            ):

                st.write(
                    f"Batch Novelty: "
                    f"`{batch_value:.6f}`"
                )

                st.write(
                    f"Live Calculation: "
                    f"`{result['novelty']:.6f}`"
                )

                st.write(
                    f"Absolute Difference: "
                    f"`{difference:.12f}`"
                )

                if difference <= 1e-6:

                    st.success(
                        "Kết quả live khớp "
                        "với batch pipeline."
                    )

                else:

                    st.warning(
                        "Có sai khác giữa "
                        "live calculation và "
                        "novelty.csv."
                    )


# ============================================================
# TAB 2 - QUERY FEATURES
# ============================================================


with tab_features:

    st.header(
        "② Query Features"
    )

    st.markdown(
        f"### SQL gốc — `{selected_query}`"
    )

    with st.expander(
        "📄 Xem SQL",
        expanded=False,
    ):

        st.code(
            query_sql,
            language="sql",
        )


    st.markdown(
        "### Structural Features"
    )

    st.write(
        "Ba tập đặc trưng dưới đây được "
        "trích trực tiếp từ SQL trên."
    )


    tables_col, joins_col, filters_col = (
        st.columns(3)
    )


    with tables_col:

        display_feature_column(
            "Tables",
            test_feature[
                "tables"
            ],
        )


    with joins_col:

        display_feature_column(
            "Joins",
            test_feature[
                "joins"
            ],
        )


    with filters_col:

        display_feature_column(
            "Filters",
            test_feature[
                "filters"
            ],
        )


    st.info(
        "Literal values như `1980`, `2010`, "
        "`'mini biography'` được chuẩn hóa thành "
        "`<LITERAL>` hoặc `<LITERAL_SET>`. "
        "Do đó Novelty hiện tại phản ánh "
        "structural query novelty."
    )


# ============================================================
# TAB 3 - TRAIN COMPARISON
# ============================================================


with tab_candidates:

    st.header(
        "③ Train Comparison"
    )

    st.write(
        f"`{selected_query}` được so sánh với "
        f"toàn bộ **{result['train_size']}** "
        f"Train Queries trong "
        f"`{selected_split_type}-{selected_split_id}`."
    )


    top_n = st.slider(
        "Số Train Candidates muốn hiển thị",
        min_value=5,
        max_value=result[
            "train_size"
        ],
        value=min(
            20,
            result[
                "train_size"
            ],
        ),
        step=1,
    )


    candidate_display = (
        result[
            "candidate_df"
        ]
        .head(
            top_n
        )
        .copy()
    )


    st.dataframe(
        candidate_display,
        hide_index=True,
        width="stretch",
        column_config={
            "d_table":
                st.column_config.NumberColumn(
                    format="%.6f"
                ),

            "d_join":
                st.column_config.NumberColumn(
                    format="%.6f"
                ),

            "d_filter":
                st.column_config.NumberColumn(
                    format="%.6f"
                ),

            "D(q,h)":
                st.column_config.NumberColumn(
                    format="%.6f"
                ),
        },
    )


    st.success(
        f"MIN D(q,h) = "
        f"{result['novelty']:.6f} "
        f"tại Train Query "
        f"`{result['nearest']}`."
    )


    if (
        result[
            "tie_count"
        ]
        > 1
    ):

        st.warning(
            f"Có {result['tie_count']} "
            f"Train Queries cùng đạt "
            f"khoảng cách nhỏ nhất."
        )


    st.markdown(
        "### Top 10 Nearest Train Queries"
    )


    chart_df = (
        result[
            "candidate_df"
        ]
        .head(10)
        .set_index(
            "Train Query"
        )[
            [
                "D(q,h)"
            ]
        ]
    )


    st.bar_chart(
        chart_df
    )


# ============================================================
# TAB 4 - JACCARD DERIVATION
# ============================================================


with tab_jaccard:

    st.header(
        "④ Jaccard Derivation"
    )

    st.write(
        "Chọn một Train Query để xem đầy đủ "
        "intersection, union và phép tính "
        "Jaccard Distance."
    )


    candidate_order = (
        result[
            "candidate_df"
        ][
            "Train Query"
        ]
        .tolist()
    )


    selected_train_query = (
        st.selectbox(
            "Train Query",
            candidate_order,
            index=0,
            help=(
                "Query đầu tiên mặc định là "
                "Nearest Train Query."
            ),
        )
    )


    pair_details = (
        result[
            "detail_map"
        ][
            selected_train_query
        ]
    )


    train_feature = (
        feature_map[
            selected_train_query
        ]
    )


    st.markdown(
        f"""
**Test Query:** `{selected_query}`

**Train Query:** `{selected_train_query}`

**Train Family:** `{train_feature["family"]}`
"""
    )


    pair_m1, pair_m2, pair_m3, pair_m4 = (
        st.columns(4)
    )


    pair_m1.metric(
        "d_table",
        f"{pair_details['table']['distance']:.6f}",
    )


    pair_m2.metric(
        "d_join",
        f"{pair_details['join']['distance']:.6f}",
    )


    pair_m3.metric(
        "d_filter",
        f"{pair_details['filter']['distance']:.6f}",
    )


    pair_m4.metric(
        "D(q,h)",
        f"{pair_details['distance']:.6f}",
    )


    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    display_jaccard_derivation(
        "4.1 Table Distance",
        pair_details[
            "table"
        ],
    )


    st.divider()


    # --------------------------------------------------------
    # JOIN
    # --------------------------------------------------------

    display_jaccard_derivation(
        "4.2 Join Distance",
        pair_details[
            "join"
        ],
    )


    st.divider()


    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    display_jaccard_derivation(
        "4.3 Filter Distance",
        pair_details[
            "filter"
        ],
    )


    st.divider()


    # --------------------------------------------------------
    # COMBINED DISTANCE
    # --------------------------------------------------------

    st.markdown(
        "### 4.4 Combined Query Distance"
    )


    d_table = (
        pair_details[
            "table"
        ][
            "distance"
        ]
    )


    d_join = (
        pair_details[
            "join"
        ][
            "distance"
        ]
    )


    d_filter = (
        pair_details[
            "filter"
        ][
            "distance"
        ]
    )


    total_distance = (
        pair_details[
            "distance"
        ]
    )


    st.latex(
        rf"""
        D({selected_query},
          {selected_train_query})
        =
        \frac{{
        d_{{table}}
        +
        d_{{join}}
        +
        d_{{filter}}
        }}{{3}}
        """
    )


    st.latex(
        rf"""
        =
        \frac{{
        {d_table:.6f}
        +
        {d_join:.6f}
        +
        {d_filter:.6f}
        }}{{3}}
        =
        {total_distance:.6f}
        """
    )


    if (
        selected_train_query
        == result[
            "nearest"
        ]
    ):

        st.success(
            "Đây là Nearest Train Query. "
            "Do đó D(q,h) này chính là "
            "Novelty Score."
        )


        st.latex(
            rf"""
            Novelty_{{{selected_split_type}}}
            ({selected_query})
            =
            D(
            {selected_query},
            {selected_train_query}
            )
            =
            {result['novelty']:.6f}
            """
        )

    else:

        st.info(
            "Train Query này không phải "
            "candidate nhỏ nhất. "
            f"Novelty vẫn là "
            f"{result['novelty']:.6f} "
            f"tại `{result['nearest']}`."
        )


# ============================================================
# TAB 5 - COMPARE SPLITS
# ============================================================


with tab_compare:

    st.header(
        "⑤ Same Query — Different Train Sets"
    )


    test_splits = (
        get_test_splits_for_query(
            split_df,
            selected_query,
        )
    )


    st.write(
        f"`{selected_query}` xuất hiện trong "
        f"Test của **{len(test_splits)} split(s)**."
    )


    comparison_records = []


    for (
        split_type,
        split_id,
    ) in test_splits:

        split_result = (
            calculate_novelty_for_split(
                query_id=
                    selected_query,

                split_type=
                    split_type,

                split_id=
                    split_id,

                split_df=
                    split_df,

                feature_map=
                    feature_map,
            )
        )


        comparison_records.append(
            {
                "Split":
                    (
                        f"{split_type}-"
                        f"{split_id}"
                    ),

                "Train Size":
                    split_result[
                        "train_size"
                    ],

                "Nearest":
                    split_result[
                        "nearest"
                    ],

                "Nearest Family":
                    split_result[
                        "nearest_family"
                    ],

                "d_table":
                    split_result[
                        "d_table"
                    ],

                "d_join":
                    split_result[
                        "d_join"
                    ],

                "d_filter":
                    split_result[
                        "d_filter"
                    ],

                "Novelty":
                    split_result[
                        "novelty"
                    ],
            }
        )


    comparison_df = pd.DataFrame(
        comparison_records
    )


    st.dataframe(
        comparison_df,
        hide_index=True,
        width="stretch",
        column_config={
            "d_table":
                st.column_config.NumberColumn(
                    format="%.6f"
                ),

            "d_join":
                st.column_config.NumberColumn(
                    format="%.6f"
                ),

            "d_filter":
                st.column_config.NumberColumn(
                    format="%.6f"
                ),

            "Novelty":
                st.column_config.NumberColumn(
                    format="%.6f"
                ),
        },
    )


    if len(
        comparison_df
    ) >= 2:

        st.markdown(
            "### Novelty by Split"
        )


        chart = (
            comparison_df[
                [
                    "Split",
                    "Novelty",
                ]
            ]
            .set_index(
                "Split"
            )
        )


        st.bar_chart(
            chart
        )


        st.markdown(
            "### Pairwise Novelty Difference"
        )


        delta_records = []


        for (
            left,
            right,
        ) in combinations(
            comparison_records,
            2,
        ):

            delta = (
                right[
                    "Novelty"
                ]
                -
                left[
                    "Novelty"
                ]
            )


            delta_records.append(
                {
                    "Query":
                        selected_query,

                    "Split A":
                        left[
                            "Split"
                        ],

                    "Novelty A":
                        left[
                            "Novelty"
                        ],

                    "Split B":
                        right[
                            "Split"
                        ],

                    "Novelty B":
                        right[
                            "Novelty"
                        ],

                    "Δ (B - A)":
                        delta,
                }
            )


        delta_df = pd.DataFrame(
            delta_records
        )


        st.dataframe(
            delta_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Novelty A":
                    st.column_config.NumberColumn(
                        format="%.6f"
                    ),

                "Novelty B":
                    st.column_config.NumberColumn(
                        format="%.6f"
                    ),

                "Δ (B - A)":
                    st.column_config.NumberColumn(
                        format="%+.6f"
                    ),
            },
        )


        st.info(
            "Phần này minh họa trực tiếp: "
            "cùng một SQL Query có thể có "
            "Novelty khác nhau khi Train Set "
            "thay đổi."
        )

    else:

        st.warning(
            "Query này chỉ xuất hiện trong "
            "Test của một split nên chưa thể "
            "thực hiện same-query comparison."
        )


# ============================================================
# PIPELINE VALIDATION
# ============================================================


st.divider()


with st.expander(
    "🔧 Pipeline & Data Validation"
):

    st.write(
        f"Repository root: "
        f"`{REPO_ROOT}`"
    )

    st.write(
        f"Canonical JOB SQL: "
        f"`{JOB_SQL_DIR}`"
    )

    st.write(
        f"split_master.csv: "
        f"`{SPLIT_MASTER_PATH}`"
    )

    st.write(
        f"query_features.csv: "
        f"`{QUERY_FEATURES_PATH}`"
    )

    st.write(
        f"novelty.csv: "
        f"`{NOVELTY_PATH}`"
    )


    st.markdown(
        "#### Dataset Counts"
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    c1.metric(
        "JOB SQL Files",
        len(
            list(
                JOB_SQL_DIR.glob(
                    "*.sql"
                )
            )
        ),
    )


    c2.metric(
        "Query Features",
        len(
            feature_df
        ),
    )


    c3.metric(
        "Split Master Rows",
        len(
            split_df
        ),
    )


    if novelty_df is not None:

        c4.metric(
            "Novelty Rows",
            len(
                novelty_df
            ),
        )

    else:

        c4.metric(
            "Novelty Rows",
            "N/A",
        )


    st.markdown(
        "#### Split Validation"
    )


    validation_records = []


    for (
        split_type,
        split_id,
    ), group in (
        split_df.groupby(
            [
                "split_type",
                "split_id",
            ]
        )
    ):

        train = set(
            group[
                group[
                    "role"
                ]
                == "train"
            ][
                "query_id"
            ]
        )


        test = set(
            group[
                group[
                    "role"
                ]
                == "test"
            ][
                "query_id"
            ]
        )


        validation_records.append(
            {
                "Split":
                    (
                        f"{split_type}-"
                        f"{split_id}"
                    ),

                "Train":
                    len(
                        train
                    ),

                "Test":
                    len(
                        test
                    ),

                "Total":
                    len(
                        train
                        | test
                    ),

                "Train/Test Overlap":
                    len(
                        train
                        & test
                    ),
            }
        )


    validation_df = pd.DataFrame(
        validation_records
    )


    st.dataframe(
        validation_df,
        hide_index=True,
        width="stretch",
    )


# ============================================================
# FOOTER
# ============================================================


st.divider()

st.caption(
    "Team14 — Bao / Balsa / PostgreSQL Novelty Analysis"
)