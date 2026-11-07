from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Structural Novelty và hiệu năng LQO",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "analysis" / "data"

SPLIT_MASTER_PATH = DATA_DIR / "split_master.csv"
QUERY_FEATURES_PATH = DATA_DIR / "query_features.csv"
NOVELTY_PATH = DATA_DIR / "novelty.csv"

BAO_PERFORMANCE_PATH = DATA_DIR / "bao_novelty_performance.csv"
BALSA_PERFORMANCE_PATH = DATA_DIR / "balsa_novelty_performance.csv"

BAO_SPEARMAN_PATH = DATA_DIR / "bao_spearman_summary.csv"
BALSA_SPEARMAN_PATH = DATA_DIR / "balsa_spearman_summary.csv"

JOB_SQL_CANDIDATES = [
    REPO_ROOT / "balsa" / "source" / "queries" / "join-order-benchmark",
    REPO_ROOT / "bao" / "queries" / "join-order-benchmark",
    Path(r"D:\lqo_ml_perspective\bao\queries\join-order-benchmark"),
]

JOB_SQL_DIR = next(
    (path for path in JOB_SQL_CANDIDATES if path.exists()),
    JOB_SQL_CANDIDATES[0],
)

SPLIT_ORDER = ["LOO", "Random", "BaseQuery"]
SPLIT_LABEL = {
    "LOO": "LOO-1",
    "Random": "Random-1",
    "BaseQuery": "BaseQuery-1",
}
METHOD_ORDER = ["PostgreSQL", "Bao", "Balsa"]


# ============================================================
# BASIC HELPERS
# ============================================================

def query_sort_key(query_id: str) -> Tuple[int, str]:
    match = re.fullmatch(r"q(\d+)([a-z]+)", str(query_id).lower())
    if not match:
        return 999999, str(query_id)
    return int(match.group(1)), match.group(2)


def parse_json_feature(value) -> Set[str]:
    if pd.isna(value):
        return set()
    parsed = json.loads(str(value))
    return set(parsed)


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_common_columns(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None

    df = df.copy()

    if "query_id" in df.columns:
        df["query_id"] = df["query_id"].astype(str).str.lower()

    if "split_id" in df.columns:
        df["split_id"] = pd.to_numeric(df["split_id"], errors="coerce").astype("Int64")

    return df


def format_ms(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.3f} ms"


def format_ratio(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}×"


# ============================================================
# LOAD DATA
# ============================================================

if not SPLIT_MASTER_PATH.exists() or not QUERY_FEATURES_PATH.exists():
    st.error("Thiếu split_master.csv hoặc query_features.csv trong analysis/data.")
    st.stop()

split_df = pd.read_csv(SPLIT_MASTER_PATH, encoding="utf-8-sig")
feature_df = pd.read_csv(QUERY_FEATURES_PATH, encoding="utf-8-sig")
novelty_df = load_optional_csv(NOVELTY_PATH)
bao_df = load_optional_csv(BAO_PERFORMANCE_PATH)
balsa_df = load_optional_csv(BALSA_PERFORMANCE_PATH)
bao_spearman_df = load_optional_csv(BAO_SPEARMAN_PATH)
balsa_spearman_df = load_optional_csv(BALSA_SPEARMAN_PATH)

split_df["query_id"] = split_df["query_id"].astype(str).str.lower()
split_df["role"] = split_df["role"].astype(str).str.lower()
split_df["split_id"] = pd.to_numeric(split_df["split_id"], errors="coerce").astype(int)

feature_df["query_id"] = feature_df["query_id"].astype(str).str.lower()

novelty_df = normalize_common_columns(novelty_df)
bao_df = normalize_common_columns(bao_df)
balsa_df = normalize_common_columns(balsa_df)


# ============================================================
# FEATURE MAP + NOVELTY
# ============================================================

def build_feature_map(df: pd.DataFrame) -> Dict[str, dict]:
    result: Dict[str, dict] = {}

    for _, row in df.iterrows():
        query_id = row["query_id"]
        tables = parse_json_feature(row["tables"])
        joins = parse_json_feature(row["joins"])
        filters = parse_json_feature(row["filters"])

        result[query_id] = {
            "family": row.get("family", ""),
            "tables": tables,
            "joins": joins,
            "filters": filters,
            "n_tables": int(row.get("n_tables", len(tables))),
            "n_joins": int(row.get("n_joins", len(joins))),
            "n_filters": int(row.get("n_filters", len(filters))),
        }

    return result


feature_map = build_feature_map(feature_df)


def jaccard_distance(a: Set[str], b: Set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return 1.0 - (len(a & b) / len(union))


def query_distance(test_feature: dict, train_feature: dict) -> dict:
    d_table = jaccard_distance(test_feature["tables"], train_feature["tables"])
    d_join = jaccard_distance(test_feature["joins"], train_feature["joins"])
    d_filter = jaccard_distance(test_feature["filters"], train_feature["filters"])

    return {
        "d_table": d_table,
        "d_join": d_join,
        "d_filter": d_filter,
        "distance": (d_table + d_join + d_filter) / 3.0,
    }


def get_split_queries(
    split_type: str,
    split_id: int,
) -> Tuple[List[str], List[str]]:
    selected = split_df[
        (split_df["split_type"] == split_type)
        & (split_df["split_id"] == split_id)
    ]

    train = sorted(
        selected[selected["role"] == "train"]["query_id"].tolist(),
        key=query_sort_key,
    )
    test = sorted(
        selected[selected["role"] == "test"]["query_id"].tolist(),
        key=query_sort_key,
    )

    return train, test


def calculate_novelty(
    query_id: str,
    split_type: str,
    split_id: int,
) -> dict:
    train_queries, test_queries = get_split_queries(split_type, split_id)

    if query_id not in test_queries:
        raise ValueError(f"{query_id} không thuộc Test của {split_type}-{split_id}")

    test_feature = feature_map[query_id]
    records = []

    for train_query in train_queries:
        details = query_distance(test_feature, feature_map[train_query])
        records.append(
            {
                "Train Query": train_query,
                "Family": feature_map[train_query]["family"],
                "d_table": details["d_table"],
                "d_join": details["d_join"],
                "d_filter": details["d_filter"],
                "D(q,h)": details["distance"],
            }
        )

    candidate_df = pd.DataFrame(records).sort_values(
        by=["D(q,h)", "Train Query"],
        key=lambda col: col if col.name != "Train Query" else col.map(query_sort_key),
    )
    candidate_df = candidate_df.reset_index(drop=True)
    candidate_df.insert(0, "Rank", range(1, len(candidate_df) + 1))

    nearest = candidate_df.iloc[0]

    return {
        "train_size": len(train_queries),
        "test_size": len(test_queries),
        "nearest": str(nearest["Train Query"]),
        "novelty": float(nearest["D(q,h)"]),
        "d_table": float(nearest["d_table"]),
        "d_join": float(nearest["d_join"]),
        "d_filter": float(nearest["d_filter"]),
        "candidate_df": candidate_df,
    }


def novelty_thresholds(split_type: str, split_id: int) -> Tuple[float | None, float | None]:
    if novelty_df is None or "novelty" not in novelty_df.columns:
        return None, None

    rows = novelty_df[
        (novelty_df["split_type"] == split_type)
        & (novelty_df["split_id"] == split_id)
    ].copy()

    rows["novelty"] = pd.to_numeric(rows["novelty"], errors="coerce")
    rows = rows[rows["novelty"].notna()]

    if rows.empty:
        return None, None

    return (
        float(rows["novelty"].quantile(1 / 3)),
        float(rows["novelty"].quantile(2 / 3)),
    )


def novelty_level(value: float, q33: float | None, q67: float | None) -> str:
    if q33 is None or q67 is None:
        return "N/A"
    if value <= q33:
        return "THẤP"
    if value <= q67:
        return "TRUNG BÌNH"
    return "CAO"


# ============================================================
# SQL
# ============================================================

@st.cache_data
def load_query_sql(query_id: str) -> Tuple[str, str]:
    filename = query_id.lower().removeprefix("q") + ".sql"
    path = JOB_SQL_DIR / filename

    if not path.exists():
        return "-- Không tìm thấy SQL trong repo hiện tại.", str(path)

    return path.read_text(encoding="utf-8"), str(path)


# ============================================================
# PERFORMANCE HELPERS
# ============================================================

def selected_performance_row(
    df: pd.DataFrame | None,
    split_type: str,
    split_id: int,
    query_id: str,
) -> pd.Series | None:
    if df is None:
        return None

    required = {"split_type", "split_id", "query_id"}
    if not required.issubset(df.columns):
        return None

    rows = df[
        (df["split_type"] == split_type)
        & (df["split_id"] == split_id)
        & (df["query_id"].astype(str).str.lower() == query_id.lower())
    ]

    if rows.empty:
        return None
    return rows.iloc[0]


def comparable_rows(df: pd.DataFrame | None, model_key: str) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()

    status_col = f"{model_key}_status"
    required = {"split_type", "split_id", "query_id", status_col, "pg_status"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    return df[
        (df[status_col].astype(str).str.upper() == "OK")
        & (df["pg_status"].astype(str).str.upper() == "OK")
    ].copy()


def safe_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    return pd.to_numeric(df[column], errors="coerce")


def median_ratio(numerator: pd.Series, denominator: pd.Series) -> float | None:
    temp = pd.DataFrame(
        {
            "num": pd.to_numeric(numerator, errors="coerce"),
            "den": pd.to_numeric(denominator, errors="coerce"),
        }
    )
    temp = temp[temp["num"].notna() & temp["den"].notna() & (temp["den"] > 0)]
    if temp.empty:
        return None
    return float((temp["num"] / temp["den"]).median())


def build_method_overview(df: pd.DataFrame | None, model_key: str, method_label: str) -> pd.DataFrame:
    rows = comparable_rows(df, model_key)
    if rows.empty:
        return pd.DataFrame()

    output = []

    for split_type in SPLIT_ORDER:
        split_rows = rows[rows["split_type"] == split_type].copy()
        if split_rows.empty:
            continue

        pg_planning = safe_numeric(split_rows, "pg_planning_ms")
        pg_execution = safe_numeric(split_rows, "pg_execution_ms")

        model_planning = safe_numeric(split_rows, f"{model_key}_planning_ms")
        model_execution = safe_numeric(split_rows, f"{model_key}_execution_ms")

        if model_key == "balsa":
            inference = safe_numeric(split_rows, "balsa_inference_ms").fillna(0.0)
        else:
            # Bao inference is already included in Bao planning.
            inference = pd.Series([0.0] * len(split_rows), index=split_rows.index)

        planning_ratio = median_ratio(model_planning, pg_planning)
        inference_ratio = median_ratio(inference, pg_planning)
        execution_ratio = median_ratio(model_execution, pg_execution)

        output.append(
            {
                "split_type": split_type,
                "split": SPLIT_LABEL[split_type],
                "method": method_label,
                "planning_ratio": planning_ratio,
                "inference_ratio": inference_ratio,
                "execution_ratio": execution_ratio,
                "n": len(split_rows),
            }
        )

    return pd.DataFrame(output)


def build_overview_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    bao_summary = build_method_overview(bao_df, "bao", "Bao")
    balsa_summary = build_method_overview(balsa_df, "balsa", "Balsa")

    summaries = [df for df in [bao_summary, balsa_summary] if not df.empty]
    if not summaries:
        return pd.DataFrame(), pd.DataFrame()

    model_summary = pd.concat(summaries, ignore_index=True)

    planning_records = []
    execution_records = []

    for split_type in SPLIT_ORDER:
        split_label = SPLIT_LABEL[split_type]

        # PostgreSQL is the normalized baseline = 1.
        planning_records.append(
            {
                "split_method": f"{split_label} · PostgreSQL",
                "split": split_label,
                "method": "PostgreSQL",
                "component": "Planning",
                "ratio": 1.0,
            }
        )
        execution_records.append(
            {
                "split_method": f"{split_label} · PostgreSQL",
                "split": split_label,
                "method": "PostgreSQL",
                "ratio": 1.0,
            }
        )

        for method in ["Bao", "Balsa"]:
            row = model_summary[
                (model_summary["split_type"] == split_type)
                & (model_summary["method"] == method)
            ]

            if row.empty:
                continue

            row = row.iloc[0]
            planning_ratio = row["planning_ratio"]
            inference_ratio = row["inference_ratio"]
            execution_ratio = row["execution_ratio"]

            if pd.notna(planning_ratio):
                planning_records.append(
                    {
                        "split_method": f"{split_label} · {method}",
                        "split": split_label,
                        "method": method,
                        "component": "Planning",
                        "ratio": float(planning_ratio),
                    }
                )

            if method == "Balsa" and pd.notna(inference_ratio) and float(inference_ratio) > 0:
                planning_records.append(
                    {
                        "split_method": f"{split_label} · {method}",
                        "split": split_label,
                        "method": method,
                        "component": "Inference",
                        "ratio": float(inference_ratio),
                    }
                )

            if pd.notna(execution_ratio):
                execution_records.append(
                    {
                        "split_method": f"{split_label} · {method}",
                        "split": split_label,
                        "method": method,
                        "ratio": float(execution_ratio),
                    }
                )

    return pd.DataFrame(planning_records), pd.DataFrame(execution_records)


def build_e2e_summary() -> pd.DataFrame:
    records = []

    for df, model_key, method_label in [
        (bao_df, "bao", "Bao"),
        (balsa_df, "balsa", "Balsa"),
    ]:
        rows = comparable_rows(df, model_key)
        if rows.empty:
            continue

        ratio_col = f"{model_key}_vs_pg_ratio"

        for split_type in SPLIT_ORDER:
            split_rows = rows[rows["split_type"] == split_type].copy()
            if split_rows.empty:
                continue

            if ratio_col in split_rows.columns:
                ratios = pd.to_numeric(split_rows[ratio_col], errors="coerce").dropna()
            else:
                ratios = (
                    pd.to_numeric(split_rows[f"{model_key}_end_to_end_ms"], errors="coerce")
                    / pd.to_numeric(split_rows["pg_end_to_end_ms"], errors="coerce")
                ).dropna()

            ratios = ratios[ratios > 0]
            if ratios.empty:
                continue

            records.append(
                {
                    "Split": SPLIT_LABEL[split_type],
                    "Method": method_label,
                    "Median E2E / PG": float(ratios.median()),
                    "Comparable": len(ratios),
                }
            )

    # Add PostgreSQL baseline rows.
    for split_type in SPLIT_ORDER:
        records.append(
            {
                "Split": SPLIT_LABEL[split_type],
                "Method": "PostgreSQL",
                "Median E2E / PG": 1.0,
                "Comparable": None,
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    result["Split"] = pd.Categorical(
        result["Split"],
        [SPLIT_LABEL[x] for x in SPLIT_ORDER],
        ordered=True,
    )
    result["Method"] = pd.Categorical(result["Method"], METHOD_ORDER, ordered=True)
    return result.sort_values(["Split", "Method"]).reset_index(drop=True)


# ============================================================
# VISUAL 1 — PAPER-STYLE OVERVIEW
# ============================================================

def render_paper_style_overview() -> None:
    st.markdown("## ⑤ Tổng quan hiệu năng trên toàn test set")
    st.caption(
        "Visual theo tinh thần Figure 4: bên trái là Planning/Inference, "
        "bên phải là Execution. PostgreSQL được chuẩn hóa về 1× để Bao, "
        "Balsa và PostgreSQL có thể đặt trên cùng một hình dù hai pipeline "
        "có PostgreSQL baseline thô khác nhau."
    )

    planning_df, execution_df = build_overview_data()

    if planning_df.empty or execution_df.empty:
        st.warning("Chưa đủ dữ liệu Bao/Balsa để dựng overview.")
        return

    left, right = st.columns(2)

    with left:
        st.markdown("### Planning + Inference")

        planning_order = []
        for split_type in SPLIT_ORDER:
            for method in METHOD_ORDER:
                planning_order.append(f"{SPLIT_LABEL[split_type]} · {method}")

        planning_spec = {
            "mark": {"type": "bar"},
            "encoding": {
                "y": {
                    "field": "split_method",
                    "type": "nominal",
                    "sort": planning_order,
                    "title": None,
                },
                "x": {
                    "aggregate": "sum",
                    "field": "ratio",
                    "type": "quantitative",
                    "title": "Median ratio vs PostgreSQL planning (PG = 1)",
                },
                "color": {
                    "field": "component",
                    "type": "nominal",
                    "title": None,
                },
                "tooltip": [
                    {"field": "split", "type": "nominal", "title": "Split"},
                    {"field": "method", "type": "nominal", "title": "Method"},
                    {"field": "component", "type": "nominal", "title": "Component"},
                    {"field": "ratio", "type": "quantitative", "format": ".3f", "title": "Ratio"},
                ],
            },
        }

        st.vega_lite_chart(planning_df, planning_spec, use_container_width=True)
        st.caption("Bao inference đã nằm trong Bao planning; Balsa inference được tách riêng.")

    with right:
        st.markdown("### Execution")

        execution_order = []
        for split_type in SPLIT_ORDER:
            for method in METHOD_ORDER:
                execution_order.append(f"{SPLIT_LABEL[split_type]} · {method}")

        execution_spec = {
            "layer": [
                {
                    "mark": {"type": "rule", "strokeDash": [6, 4]},
                    "encoding": {"x": {"datum": 1}},
                },
                {
                    "mark": {"type": "bar"},
                    "encoding": {
                        "y": {
                            "field": "split_method",
                            "type": "nominal",
                            "sort": execution_order,
                            "title": None,
                        },
                        "x": {
                            "field": "ratio",
                            "type": "quantitative",
                            "title": "Median execution ratio vs PostgreSQL (PG = 1)",
                        },
                        "color": {
                            "field": "method",
                            "type": "nominal",
                            "title": None,
                        },
                        "tooltip": [
                            {"field": "split", "type": "nominal", "title": "Split"},
                            {"field": "method", "type": "nominal", "title": "Method"},
                            {"field": "ratio", "type": "quantitative", "format": ".3f", "title": "Ratio"},
                        ],
                    },
                },
            ]
        }

        st.vega_lite_chart(execution_df, execution_spec, use_container_width=True)
        st.caption("Đường dọc 1× là PostgreSQL baseline.")

    e2e = build_e2e_summary()
    if not e2e.empty:
        st.markdown("### End-to-End · median trên test set")
        st.dataframe(
            e2e,
            hide_index=True,
            width="stretch",
            column_config={
                "Median E2E / PG": st.column_config.NumberColumn(format="%.3f×"),
            },
        )


# ============================================================
# VISUAL 2 — NOVELTY ↔ PERFORMANCE
# ============================================================

def scatter_data_for_split(
    df: pd.DataFrame | None,
    model_key: str,
    split_type: str,
    split_id: int,
    selected_query: str,
) -> pd.DataFrame:
    rows = comparable_rows(df, model_key)
    if rows.empty:
        return pd.DataFrame()

    rows = rows[
        (rows["split_type"] == split_type)
        & (rows["split_id"] == split_id)
    ].copy()

    ratio_col = f"{model_key}_vs_pg_ratio"
    if ratio_col not in rows.columns:
        return pd.DataFrame()

    rows["novelty"] = pd.to_numeric(rows["novelty"], errors="coerce")
    rows["ratio"] = pd.to_numeric(rows[ratio_col], errors="coerce")
    rows = rows[rows["novelty"].notna() & rows["ratio"].notna() & (rows["ratio"] > 0)]

    if rows.empty:
        return rows

    rows["Point"] = rows["query_id"].apply(
        lambda q: "Query đang chọn" if str(q).lower() == selected_query.lower() else "Query khác"
    )

    return rows[["query_id", "novelty", "ratio", "Point"]].copy()


def spearman_text(df: pd.DataFrame | None, split_type: str) -> str | None:
    if df is None:
        return None

    key = "split_type" if "split_type" in df.columns else "group" if "group" in df.columns else None
    if key is None:
        return None

    rows = df[df[key] == split_type]
    if rows.empty:
        return None

    row = rows.iloc[0]
    rho = row.get("rho")
    p_value = row.get("p_value")

    if pd.isna(rho) or pd.isna(p_value):
        return None

    return f"ρ = {float(rho):.3f} · p = {float(p_value):.3f}"


def render_scatter(
    df: pd.DataFrame | None,
    model_key: str,
    model_label: str,
    split_type: str,
    split_id: int,
    selected_query: str,
    spearman_df: pd.DataFrame | None,
) -> None:
    scatter_df = scatter_data_for_split(
        df,
        model_key,
        split_type,
        split_id,
        selected_query,
    )

    if scatter_df.empty:
        st.warning(f"Không đủ dữ liệu để vẽ {model_label}.")
        return

    selected_filter = "datum.Point == 'Query đang chọn'"

    spec = {
        "layer": [
            {
                "mark": {"type": "rule", "strokeDash": [6, 4]},
                "encoding": {"y": {"datum": 1}},
            },
            {
                "mark": {"type": "point", "filled": True},
                "encoding": {
                    "x": {
                        "field": "novelty",
                        "type": "quantitative",
                        "title": "Structural Novelty",
                    },
                    "y": {
                        "field": "ratio",
                        "type": "quantitative",
                        "scale": {"type": "log"},
                        "title": f"{model_label} / PostgreSQL · E2E",
                    },
                    "color": {
                        "field": "Point",
                        "type": "nominal",
                        "title": None,
                    },
                    "size": {
                        "condition": {"test": selected_filter, "value": 180},
                        "value": 70,
                    },
                    "tooltip": [
                        {"field": "query_id", "type": "nominal", "title": "Query"},
                        {"field": "novelty", "type": "quantitative", "format": ".6f", "title": "Novelty"},
                        {"field": "ratio", "type": "quantitative", "format": ".3f", "title": f"{model_label}/PG"},
                    ],
                },
            },
            {
                "transform": [{"filter": selected_filter}],
                "mark": {"type": "text", "dy": -14, "fontWeight": "bold"},
                "encoding": {
                    "x": {"field": "novelty", "type": "quantitative"},
                    "y": {"field": "ratio", "type": "quantitative", "scale": {"type": "log"}},
                    "text": {"field": "query_id"},
                },
            },
        ]
    }

    st.vega_lite_chart(scatter_df, spec, use_container_width=True)

    stats = spearman_text(spearman_df, split_type)
    if stats:
        st.caption(stats)


def render_novelty_performance_visual(
    split_type: str,
    split_id: int,
    selected_query: str,
) -> None:
    st.markdown("## ⑥ Structural Novelty ↔ hiệu năng")
    st.caption(
        "Đây là visual chính của phần mở rộng: mỗi điểm là một test query. "
        "Trục X là Novelty; trục Y là E2E model/PostgreSQL. Đường 1× là PostgreSQL."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Bao")
        render_scatter(
            bao_df,
            "bao",
            "Bao",
            split_type,
            split_id,
            selected_query,
            bao_spearman_df,
        )

    with right:
        st.markdown("### Balsa")
        render_scatter(
            balsa_df,
            "balsa",
            "Balsa",
            split_type,
            split_id,
            selected_query,
            balsa_spearman_df,
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Truy vấn minh họa")

available_split_types = split_df["split_type"].drop_duplicates().tolist()
split_types = [x for x in SPLIT_ORDER if x in available_split_types]
split_types += [x for x in available_split_types if x not in split_types]

selected_split_type = st.sidebar.selectbox(
    "1. Chọn split",
    split_types,
)

split_ids = sorted(
    split_df[split_df["split_type"] == selected_split_type]["split_id"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)
selected_split_id = int(split_ids[0])

train_queries, test_queries = get_split_queries(selected_split_type, selected_split_id)

DEFAULT_QUERY = {
    "LOO": "q11b",
    "Random": "q6c",
    "BaseQuery": "q15b",
}

preferred = DEFAULT_QUERY.get(selected_split_type)
default_index = test_queries.index(preferred) if preferred in test_queries else 0

selected_query = st.sidebar.selectbox(
    "2. Chọn test query",
    test_queries,
    index=default_index,
)

st.sidebar.caption(f"{SPLIT_LABEL.get(selected_split_type, selected_split_type)}")
st.sidebar.metric("Train", len(train_queries))
st.sidebar.metric("Test", len(test_queries))


# ============================================================
# SELECTED QUERY CALCULATION
# ============================================================

result = calculate_novelty(
    selected_query,
    selected_split_type,
    selected_split_id,
)

q33, q67 = novelty_thresholds(selected_split_type, selected_split_id)
level = novelty_level(result["novelty"], q33, q67)
query_feature = feature_map[selected_query]
query_sql, query_sql_path = load_query_sql(selected_query)

bao_row = selected_performance_row(
    bao_df,
    selected_split_type,
    selected_split_id,
    selected_query,
)
balsa_row = selected_performance_row(
    balsa_df,
    selected_split_type,
    selected_split_id,
    selected_query,
)


# ============================================================
# HEADER
# ============================================================

st.title("Structural Novelty và hiệu năng Bao · Balsa · PostgreSQL")
st.caption("JOB · LOO-1 · Random-1 · BaseQuery-1")


# ============================================================
# STEP 1 — SELECTED QUERY + NOVELTY
# ============================================================

st.markdown("## ① Test query được chọn")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Query", selected_query)
c2.metric("Split", SPLIT_LABEL.get(selected_split_type, selected_split_type))
c3.metric("Train queries", result["train_size"])
c4.metric("Test queries", result["test_size"])

with st.expander("Xem SQL gốc", expanded=False):
    st.code(query_sql, language="sql")
    st.caption(query_sql_path)


# ============================================================
# STEP 2 — STRUCTURAL FEATURES
# ============================================================

st.markdown("## ② Đặc trưng cấu trúc")

f1, f2, f3 = st.columns(3)
f1.metric("Tables", query_feature["n_tables"])
f2.metric("Joins", query_feature["n_joins"])
f3.metric("Filters", query_feature["n_filters"])

with st.expander("Xem chi tiết Tables / Joins / Filters", expanded=False):
    a, b, c = st.columns(3)

    with a:
        st.markdown("**Tables**")
        st.dataframe(pd.DataFrame({"value": sorted(query_feature["tables"])}), hide_index=True, width="stretch")

    with b:
        st.markdown("**Joins**")
        st.dataframe(pd.DataFrame({"value": sorted(query_feature["joins"])}), hide_index=True, width="stretch")

    with c:
        st.markdown("**Filters**")
        st.dataframe(pd.DataFrame({"value": sorted(query_feature["filters"])}), hide_index=True, width="stretch")


# ============================================================
# STEP 3 — NEAREST TRAIN QUERY
# ============================================================

st.markdown("## ③ So sánh với tập huấn luyện")

st.latex(r"D(q,h)=\frac{d_{table}(q,h)+d_{join}(q,h)+d_{filter}(q,h)}{3}")

preview = result["candidate_df"].head(5).copy()
st.dataframe(
    preview,
    hide_index=True,
    width="stretch",
    column_config={
        "d_table": st.column_config.NumberColumn(format="%.6f"),
        "d_join": st.column_config.NumberColumn(format="%.6f"),
        "d_filter": st.column_config.NumberColumn(format="%.6f"),
        "D(q,h)": st.column_config.NumberColumn(format="%.6f"),
    },
)


# ============================================================
# STEP 4 — NOVELTY SCORE
# ============================================================

st.markdown("## ④ Structural Novelty Score")

n1, n2, n3, n4 = st.columns(4)
n1.metric("d_table", f"{result['d_table']:.6f}")
n2.metric("d_join", f"{result['d_join']:.6f}")
n3.metric("d_filter", f"{result['d_filter']:.6f}")
n4.metric("Novelty", f"{result['novelty']:.6f}")

st.latex(
    rf"Novelty({selected_query}) = "
    rf"\min_{{h\in Train}}D({selected_query},h) = {result['novelty']:.6f}"
)

level_col, nearest_col = st.columns(2)
level_col.metric("Mức Novelty", level)
nearest_col.metric("Nearest train query", result["nearest"])

if q33 is not None and q67 is not None:
    st.caption(f"q33 = {q33:.6f} · q67 = {q67:.6f}")


# ============================================================
# STEP 5 — PAPER-STYLE OVERVIEW
# ============================================================

render_paper_style_overview()


# ============================================================
# STEP 6 — NOVELTY × PERFORMANCE
# ============================================================

render_novelty_performance_visual(
    selected_split_type,
    selected_split_id,
    selected_query,
)


# ============================================================
# SELECTED QUERY PERFORMANCE
# ============================================================

st.markdown("## ⑦ Query đang chọn nằm ở đâu trong kết quả?")

perf1, perf2 = st.columns(2)

with perf1:
    st.markdown("### Bao")
    if bao_row is None:
        st.warning("Không có dữ liệu Bao cho query này.")
    else:
        p1, p2 = st.columns(2)
        p1.metric("Bao E2E", format_ms(bao_row.get("bao_end_to_end_ms")))
        p2.metric("Bao / PG", format_ratio(bao_row.get("bao_vs_pg_ratio")))
        st.caption(f"Status: {bao_row.get('bao_status', 'N/A')}")

with perf2:
    st.markdown("### Balsa")
    if balsa_row is None:
        st.warning("Không có dữ liệu Balsa cho query này.")
    else:
        p1, p2 = st.columns(2)
        p1.metric("Balsa E2E", format_ms(balsa_row.get("balsa_end_to_end_ms")))
        p2.metric("Balsa / PG", format_ratio(balsa_row.get("balsa_vs_pg_ratio")))
        st.caption(f"Status: {balsa_row.get('balsa_status', 'N/A')} · Balsa hiện dùng 3 iterations")


# ============================================================
# FINAL TAKEAWAY
# ============================================================

st.divider()
st.markdown("## 🎯 Cách đọc demo")
st.markdown(
    """
- **④ Structural Novelty**: phần mở rộng của nhóm — đo test query khác tập train đến mức nào.
- **⑤ Overview**: tổng hợp toàn test set theo LOO / Random / BaseQuery, theo tinh thần Figure 4.
- **⑥ Novelty ↔ performance**: visual trực tiếp của phần mở rộng — kiểm tra Novelty có đi cùng thay đổi hiệu năng hay không.
- **⑦ Query đang chọn**: dùng một query cụ thể để minh họa vị trí của nó trong xu hướng chung.
"""
)

st.caption("Team14 · JOB · Structural Novelty · Bao · Balsa · PostgreSQL")
