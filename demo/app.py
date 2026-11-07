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
    page_title="Paper → Team Evaluation → Structural Novelty",
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
BAO_GROUP_PATH = DATA_DIR / "bao_novelty_group_summary.csv"
BALSA_GROUP_PATH = DATA_DIR / "balsa_novelty_group_summary.csv"
BAO_CASES_PATH = DATA_DIR / "bao_case_studies.csv"
BALSA_CASES_PATH = DATA_DIR / "balsa_case_studies.csv"

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
bao_group_df = load_optional_csv(BAO_GROUP_PATH)
balsa_group_df = load_optional_csv(BALSA_GROUP_PATH)
bao_cases_df = load_optional_csv(BAO_CASES_PATH)
balsa_cases_df = load_optional_csv(BALSA_CASES_PATH)

split_df["query_id"] = split_df["query_id"].astype(str).str.lower()
split_df["role"] = split_df["role"].astype(str).str.lower()
split_df["split_id"] = pd.to_numeric(split_df["split_id"], errors="coerce").astype(int)

feature_df["query_id"] = feature_df["query_id"].astype(str).str.lower()

novelty_df = normalize_common_columns(novelty_df)
bao_df = normalize_common_columns(bao_df)
balsa_df = normalize_common_columns(balsa_df)
bao_cases_df = normalize_common_columns(bao_cases_df)
balsa_cases_df = normalize_common_columns(balsa_cases_df)


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
# SPLIT-LEVEL PAPER/TEAM EVALUATION
# ============================================================

BALSA_RUNS_ROOT = REPO_ROOT / "balsa" / "runs"

BAO_RAW_EVAL_PATH = REPO_ROOT / "bao" / "results" / "ALL_evaluation_results.csv"
BAO_RAW_PG_PATH = REPO_ROOT / "bao" / "results" / "ALL_postgres_results.csv"
BALSA_RUN_DIRS = {
    "LOO": BALSA_RUNS_ROOT / "LOO-1" / "balsa-v2-run-51121",
    "Random": BALSA_RUNS_ROOT / "Random-1" / "balsa-v2-run-698900",
    "BaseQuery": BALSA_RUNS_ROOT / "BaseQuery-1" / "balsa-v2-run-1308903",
}

BAO_RUN_DIRS = {
    "LOO": REPO_ROOT / "bao" / "results" / "LOO-1" / "bao_loo1_run01",
    "Random": REPO_ROOT / "bao" / "results" / "Random-1" / "bao_rnd1_run01",
    "BaseQuery": REPO_ROOT / "bao" / "results" / "BaseQuery-1" / "bao_bq1_run01",
}

BAO_ARM_HINTS = {
    "arm_0": "Không vô hiệu hóa thêm join/scan operator (default arm).",
    "arm_1": "Disable Nested Loop.",
    "arm_2": "Disable Merge Join + Index Scan.",
    "arm_3": "Disable Nested Loop + Merge Join + Index Scan.",
    "arm_4": "Disable Merge Join.",
    "MULTIPLE": "Bao chọn nhiều arm khác nhau qua các repeat; xem danh sách chosen_plan_ids.",
}


def format_seconds_from_ms(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) / 1000.0:,.3f} s"


def split_comparable_rows(df: pd.DataFrame | None, model_key: str, split_type: str) -> pd.DataFrame:
    rows = comparable_rows(df, model_key)
    if rows.empty:
        return rows
    return rows[(rows["split_type"] == split_type) & (rows["split_id"] == 1)].copy()


def build_split_overview_table() -> pd.DataFrame:
    records = []
    for split_type in SPLIT_ORDER:
        record = {"Split": SPLIT_LABEL[split_type]}

        bao_rows = split_comparable_rows(bao_df, "bao", split_type)
        if not bao_rows.empty:
            ratios = pd.to_numeric(bao_rows["bao_vs_pg_ratio"], errors="coerce").dropna()
            record["Bao comparable"] = len(bao_rows)
            record["Bao median E2E/PG"] = float(ratios.median()) if not ratios.empty else None
            record["Bao wins"] = int((ratios < 1).sum()) if not ratios.empty else 0
            record["Bao total E2E (s)"] = pd.to_numeric(bao_rows["bao_end_to_end_ms"], errors="coerce").sum() / 1000.0
            record["PostgreSQL (Bao eval) total E2E (s)"] = pd.to_numeric(bao_rows["pg_end_to_end_ms"], errors="coerce").sum() / 1000.0
        else:
            record.update({
                "Bao comparable": 0,
                "Bao median E2E/PG": None,
                "Bao wins": 0,
                "Bao total E2E (s)": None,
                "PostgreSQL (Bao eval) total E2E (s)": None,
            })

        balsa_rows = split_comparable_rows(balsa_df, "balsa", split_type)
        if not balsa_rows.empty:
            ratios = pd.to_numeric(balsa_rows["balsa_vs_pg_ratio"], errors="coerce").dropna()
            record["Balsa comparable"] = len(balsa_rows)
            record["Balsa median E2E/PG"] = float(ratios.median()) if not ratios.empty else None
            record["Balsa wins"] = int((ratios < 1).sum()) if not ratios.empty else 0
            record["Balsa total E2E (s)"] = pd.to_numeric(balsa_rows["balsa_end_to_end_ms"], errors="coerce").sum() / 1000.0
            record["PostgreSQL (Balsa eval) total E2E (s)"] = pd.to_numeric(balsa_rows["pg_end_to_end_ms"], errors="coerce").sum() / 1000.0
        else:
            record.update({
                "Balsa comparable": 0,
                "Balsa median E2E/PG": None,
                "Balsa wins": 0,
                "Balsa total E2E (s)": None,
                "PostgreSQL (Balsa eval) total E2E (s)": None,
            })

        records.append(record)

    return pd.DataFrame(records)


def component_totals(df: pd.DataFrame, model_key: str, model_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    records = []

    pg_planning = pd.to_numeric(df["pg_planning_ms"], errors="coerce").sum() / 1000.0
    pg_execution = pd.to_numeric(df["pg_execution_ms"], errors="coerce").sum() / 1000.0
    records.extend([
        {"Method": "PostgreSQL baseline", "Component": "Planning", "Seconds": pg_planning},
        {"Method": "PostgreSQL baseline", "Component": "Execution", "Seconds": pg_execution},
    ])

    if model_key == "bao":
        planning = pd.to_numeric(df["bao_planning_ms"], errors="coerce").sum() / 1000.0
        execution = pd.to_numeric(df["bao_execution_ms"], errors="coerce").sum() / 1000.0
        records.extend([
            {"Method": model_label, "Component": "Planning + Bao inference", "Seconds": planning},
            {"Method": model_label, "Component": "Execution", "Seconds": execution},
        ])
    else:
        inference = pd.to_numeric(df["balsa_inference_ms"], errors="coerce").sum() / 1000.0
        planning = pd.to_numeric(df["balsa_planning_ms"], errors="coerce").sum() / 1000.0
        execution = pd.to_numeric(df["balsa_execution_ms"], errors="coerce").sum() / 1000.0
        records.extend([
            {"Method": model_label, "Component": "Inference / plan search", "Seconds": inference},
            {"Method": model_label, "Component": "PostgreSQL planning", "Seconds": planning},
            {"Method": model_label, "Component": "Execution", "Seconds": execution},
        ])

    return pd.DataFrame(records)


def render_component_chart(chart_df: pd.DataFrame) -> None:
    if chart_df.empty:
        st.warning("Không đủ dữ liệu để dựng biểu đồ.")
        return

    spec = {
        "mark": {"type": "bar"},
        "encoding": {
            "y": {"field": "Method", "type": "nominal", "title": None},
            "x": {
                "aggregate": "sum",
                "field": "Seconds",
                "type": "quantitative",
                "title": "Tổng thời gian trên các query comparable (giây)",
            },
            "color": {"field": "Component", "type": "nominal", "title": None},
            "tooltip": [
                {"field": "Method", "type": "nominal"},
                {"field": "Component", "type": "nominal"},
                {"field": "Seconds", "type": "quantitative", "format": ".3f"},
            ],
        },
    }
    st.vega_lite_chart(chart_df, spec, use_container_width=True)


def render_model_split_panel(df: pd.DataFrame | None, model_key: str, model_label: str, split_type: str) -> None:
    rows = split_comparable_rows(df, model_key, split_type)
    if rows.empty:
        st.warning(f"Không có dữ liệu comparable cho {model_label} · {SPLIT_LABEL[split_type]}.")
        return

    ratio_col = f"{model_key}_vs_pg_ratio"
    model_e2e_col = f"{model_key}_end_to_end_ms"
    ratios = pd.to_numeric(rows[ratio_col], errors="coerce").dropna()
    model_total = pd.to_numeric(rows[model_e2e_col], errors="coerce").sum()
    pg_total = pd.to_numeric(rows["pg_end_to_end_ms"], errors="coerce").sum()
    total_ratio = model_total / pg_total if pg_total > 0 else None
    wins = int((ratios < 1).sum()) if not ratios.empty else 0

    if model_key == "bao":
        st.markdown("**Pipeline được tái hiện:** author checkpoint → Bao chọn hint set → PostgreSQL planning → PostgreSQL execution")
    else:
        st.markdown("**Pipeline được tái hiện:** team checkpoint → Balsa plan search → chosen plan/hints → PostgreSQL planning → PostgreSQL execution")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Comparable queries", len(rows))
    m2.metric(f"{model_label} total E2E", format_seconds_from_ms(model_total))
    m3.metric("PG baseline total E2E", format_seconds_from_ms(pg_total))
    m4.metric("Total E2E / PG", "N/A" if total_ratio is None else f"{total_ratio:.3f}×")

    st.caption(
        f"Median per-query {model_label}/PG = {float(ratios.median()):.3f}× · "
        f"{model_label} thắng {wins}/{len(rows)} query comparable."
    )
    render_component_chart(component_totals(rows, model_key, model_label))

    if model_key == "bao":
        checkpoints = sorted(rows.get("bao_checkpoint_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        if checkpoints:
            st.caption(f"Checkpoint: {', '.join(checkpoints)} · Bao inference đã nằm trong Bao planning.")
    else:
        checkpoints = sorted(rows.get("balsa_checkpoint_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        if checkpoints:
            st.caption(f"Checkpoint: {', '.join(checkpoints)} · Main analysis: resource-constrained 3 iterations.")
        if "pg_baseline_session" in rows.columns:
            sessions = sorted(rows["pg_baseline_session"].dropna().astype(str).unique().tolist())
            if sessions:
                st.caption(f"PG baseline provenance: {', '.join(sessions)}.")


def build_figure4_like_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    planning_records = []
    execution_records = []
    row_order: List[str] = []

    for split_type in SPLIT_ORDER:
        split_label = SPLIT_LABEL[split_type]

        bao_rows = split_comparable_rows(bao_df, "bao", split_type)
        balsa_rows = split_comparable_rows(balsa_df, "balsa", split_type)

        if not bao_rows.empty:
            rows = [
                (f"{split_label} · PostgreSQL (Bao eval)", "Planning", pd.to_numeric(bao_rows["pg_planning_ms"], errors="coerce").sum() / 1000.0),
                (f"{split_label} · Bao", "Planning + Bao inference", pd.to_numeric(bao_rows["bao_planning_ms"], errors="coerce").sum() / 1000.0),
            ]
            for row_label, component, seconds in rows:
                planning_records.append({"Row": row_label, "Component": component, "Seconds": float(seconds)})
                if row_label not in row_order:
                    row_order.append(row_label)

            execution_records.extend([
                {
                    "Row": f"{split_label} · PostgreSQL (Bao eval)",
                    "Seconds": float(pd.to_numeric(bao_rows["pg_execution_ms"], errors="coerce").sum() / 1000.0),
                },
                {
                    "Row": f"{split_label} · Bao",
                    "Seconds": float(pd.to_numeric(bao_rows["bao_execution_ms"], errors="coerce").sum() / 1000.0),
                },
            ])

        if not balsa_rows.empty:
            pg_row = f"{split_label} · PostgreSQL (Balsa eval)"
            balsa_row = f"{split_label} · Balsa"
            planning_records.extend([
                {
                    "Row": pg_row,
                    "Component": "Planning",
                    "Seconds": float(pd.to_numeric(balsa_rows["pg_planning_ms"], errors="coerce").sum() / 1000.0),
                },
                {
                    "Row": balsa_row,
                    "Component": "PostgreSQL planning",
                    "Seconds": float(pd.to_numeric(balsa_rows["balsa_planning_ms"], errors="coerce").sum() / 1000.0),
                },
                {
                    "Row": balsa_row,
                    "Component": "Balsa inference / plan search",
                    "Seconds": float(pd.to_numeric(balsa_rows["balsa_inference_ms"], errors="coerce").sum() / 1000.0),
                },
            ])
            for row_label in [pg_row, balsa_row]:
                if row_label not in row_order:
                    row_order.append(row_label)

            execution_records.extend([
                {
                    "Row": pg_row,
                    "Seconds": float(pd.to_numeric(balsa_rows["pg_execution_ms"], errors="coerce").sum() / 1000.0),
                },
                {
                    "Row": balsa_row,
                    "Seconds": float(pd.to_numeric(balsa_rows["balsa_execution_ms"], errors="coerce").sum() / 1000.0),
                },
            ])

    return pd.DataFrame(planning_records), pd.DataFrame(execution_records), row_order


def render_figure4_like_overview() -> None:
    planning_df, execution_df, row_order = build_figure4_like_data()
    if planning_df.empty or execution_df.empty:
        st.warning("Không đủ dữ liệu để dựng Figure-4-like overview.")
        return

    shared_y = {
        "field": "Row",
        "type": "nominal",
        "sort": row_order,
        "title": None,
        "axis": {"labelLimit": 210},
    }

    spec = {
        "hconcat": [
            {
                "title": "Planning + Inference",
                "width": 500,
                "height": {"step": 25},
                "data": {"values": planning_df.to_dict(orient="records")},
                "mark": {"type": "bar"},
                "encoding": {
                    "y": shared_y,
                    "x": {
                        "aggregate": "sum",
                        "field": "Seconds",
                        "type": "quantitative",
                        "title": "Tổng thời gian trên query comparable (s)",
                    },
                    "color": {"field": "Component", "type": "nominal", "title": None},
                    "tooltip": [
                        {"field": "Row", "type": "nominal", "title": "Split / method"},
                        {"field": "Component", "type": "nominal"},
                        {"field": "Seconds", "type": "quantitative", "format": ".3f", "title": "Seconds"},
                    ],
                },
            },
            {
                "title": "Execution",
                "width": 500,
                "height": {"step": 25},
                "data": {"values": execution_df.to_dict(orient="records")},
                "mark": {"type": "bar"},
                "encoding": {
                    "y": shared_y,
                    "x": {
                        "field": "Seconds",
                        "type": "quantitative",
                        "title": "Tổng thời gian trên query comparable (s)",
                    },
                    "tooltip": [
                        {"field": "Row", "type": "nominal", "title": "Split / method"},
                        {"field": "Seconds", "type": "quantitative", "format": ".3f", "title": "Seconds"},
                    ],
                },
            },
        ],
        "resolve": {"scale": {"x": "independent"}},
    }
    st.vega_lite_chart(spec, use_container_width=True)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _status_summary(df: pd.DataFrame, status_col: str = "status") -> str:
    if df.empty or status_col not in df.columns:
        return "N/A"
    counts = df[status_col].fillna("UNKNOWN").astype(str).value_counts()
    return " · ".join(f"{key}: {int(value)}" for key, value in counts.items())


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_execution_evidence_summary() -> pd.DataFrame:
    raw_bao = _read_csv_if_exists(BAO_RAW_EVAL_PATH)
    raw_bao_pg = _read_csv_if_exists(BAO_RAW_PG_PATH)
    rows = []

    for split_type in SPLIT_ORDER:
        split_label = SPLIT_LABEL[split_type]
        train_queries, test_queries = get_split_queries(split_type, 1)

        bao_rows = raw_bao[
            (raw_bao.get("split_type") == split_type)
            & (pd.to_numeric(raw_bao.get("split_id"), errors="coerce") == 1)
        ].copy() if not raw_bao.empty else pd.DataFrame()
        bao_pg_rows = raw_bao_pg[
            (raw_bao_pg.get("split_type") == split_type)
            & (pd.to_numeric(raw_bao_pg.get("split_id"), errors="coerce") == 1)
        ].copy() if not raw_bao_pg.empty else pd.DataFrame()

        bao_manifest = _load_json_if_exists(BAO_RUN_DIRS[split_type] / "run_manifest.json")
        bao_checkpoint = (
            ", ".join(sorted(bao_rows["checkpoint_id"].dropna().astype(str).unique()))
            if not bao_rows.empty and "checkpoint_id" in bao_rows.columns
            else "N/A"
        )

        balsa_run_dir = BALSA_RUN_DIRS[split_type]
        balsa_rows = _read_csv_if_exists(balsa_run_dir / "evaluation_results.csv")
        balsa_pg_rows = _read_csv_if_exists(balsa_run_dir / "postgres_results.csv")
        balsa_manifest = _load_json_if_exists(balsa_run_dir / "run_manifest.json")
        balsa_checkpoint = (
            ", ".join(sorted(balsa_rows["checkpoint_id"].dropna().astype(str).unique()))
            if not balsa_rows.empty and "checkpoint_id" in balsa_rows.columns
            else "N/A"
        )

        eval_isolation = []
        if balsa_manifest.get("train_called_during_eval") is False:
            eval_isolation.append("no-train")
        if balsa_manifest.get("weights_unchanged") is True:
            eval_isolation.append("weights unchanged")

        rows.append({
            "Split": split_label,
            "Train / Test": f"{len(train_queries)} / {len(test_queries)}",
            "Bao checkpoint": bao_checkpoint,
            "Bao eval": _status_summary(bao_rows),
            "PostgreSQL (Bao eval)": _status_summary(bao_pg_rows),
            "Balsa checkpoint": balsa_checkpoint,
            "Balsa eval": _status_summary(balsa_rows),
            "PostgreSQL (Balsa eval)": _status_summary(balsa_pg_rows),
            "Balsa eval isolation": " · ".join(eval_isolation) if eval_isolation else "N/A",
        })

    return pd.DataFrame(rows)


def render_execution_evidence() -> None:
    st.markdown("### Nguồn thực nghiệm")
    evidence = build_execution_evidence_summary()
    st.dataframe(evidence, hide_index=True, width="stretch")
    st.caption(
        "Bao: author checkpoint + team evaluation · "
        "Balsa: team-trained checkpoint + independent evaluation · "
        "Nguồn timing: evaluation CSV và PostgreSQL baseline CSV trong repo."
    )


def render_experiment_csv_evidence() -> None:
    with st.expander("Dữ liệu chi tiết: checkpoint · planning · inference · execution · E2E", expanded=False):
        selected_split = st.selectbox(
            "Split",
            SPLIT_ORDER,
            format_func=lambda x: SPLIT_LABEL[x],
            key="csv_evidence_split",
        )

        bao_tab, balsa_tab = st.tabs(["Bao output", "Balsa output"])

        with bao_tab:
            raw_bao = _read_csv_if_exists(BAO_RAW_EVAL_PATH)
            raw_pg = _read_csv_if_exists(BAO_RAW_PG_PATH)
            st.caption(f"Bao: {BAO_RAW_EVAL_PATH.relative_to(REPO_ROOT)}")
            st.caption(f"PostgreSQL baseline: {BAO_RAW_PG_PATH.relative_to(REPO_ROOT)}")

            if not raw_bao.empty:
                rows = raw_bao[(raw_bao["split_type"] == selected_split) & (pd.to_numeric(raw_bao["split_id"], errors="coerce") == 1)].copy()
                cols = [c for c in ["query_id", "checkpoint_id", "planning_ms", "execution_ms", "end_to_end_ms", "status", "chosen_plan_id", "chosen_plan_ids"] if c in rows.columns]
                st.dataframe(rows[cols], hide_index=True, width="stretch")
            else:
                st.warning("Không tìm thấy Bao evaluation CSV.")

            if not raw_pg.empty:
                with st.expander("PostgreSQL baseline tương ứng của Bao", expanded=False):
                    rows = raw_pg[(raw_pg["split_type"] == selected_split) & (pd.to_numeric(raw_pg["split_id"], errors="coerce") == 1)].copy()
                    cols = [c for c in ["query_id", "planning_ms", "execution_ms", "end_to_end_ms", "status", "timeout_ms"] if c in rows.columns]
                    st.dataframe(rows[cols], hide_index=True, width="stretch")

        with balsa_tab:
            run_dir = BALSA_RUN_DIRS[selected_split]
            raw_eval_path = run_dir / "evaluation_results.csv"
            raw_pg_path = run_dir / "postgres_results.csv"
            raw_eval = _read_csv_if_exists(raw_eval_path)
            raw_pg = _read_csv_if_exists(raw_pg_path)
            st.caption(f"Balsa: {raw_eval_path.relative_to(REPO_ROOT)}")
            st.caption(f"PostgreSQL baseline: {raw_pg_path.relative_to(REPO_ROOT)}")

            if not raw_eval.empty:
                cols = [c for c in ["query_id", "checkpoint_id", "inference_ms", "planning_ms", "execution_ms", "end_to_end_ms", "status", "chosen_plan_id"] if c in raw_eval.columns]
                st.dataframe(raw_eval[cols], hide_index=True, width="stretch")
            else:
                st.warning("Không tìm thấy Balsa evaluation CSV.")

            if not raw_pg.empty:
                with st.expander("PostgreSQL baseline tương ứng của Balsa", expanded=False):
                    cols = [c for c in ["query_id", "planning_ms", "execution_ms", "end_to_end_ms", "status", "timeout_ms"] if c in raw_pg.columns]
                    st.dataframe(raw_pg[cols], hide_index=True, width="stretch")


def render_paper_protocol_replay() -> None:
    st.markdown("## ① Tái hiện thực nghiệm theo protocol đánh giá của paper")
    st.caption("JOB · LOO-1 · Random-1 · BaseQuery-1 · Split-level evaluation")

    # 1) Provenance / artifacts first
    render_execution_evidence()

    # 2) Numeric results before visualization
    st.markdown("### Kết quả chạy theo split")
    overview = build_split_overview_table()
    st.dataframe(
        overview,
        hide_index=True,
        width="stretch",
        column_config={
            "Bao median E2E/PG": st.column_config.NumberColumn(format="%.3f×"),
            "Balsa median E2E/PG": st.column_config.NumberColumn(format="%.3f×"),
            "Bao total E2E (s)": st.column_config.NumberColumn(format="%.3f"),
            "PostgreSQL (Bao eval) total E2E (s)": st.column_config.NumberColumn(format="%.3f"),
            "Balsa total E2E (s)": st.column_config.NumberColumn(format="%.3f"),
            "PostgreSQL (Balsa eval) total E2E (s)": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    render_experiment_csv_evidence()

    # 3) One combined Figure-4-like visualization
    st.markdown("### Figure tổng hợp — Planning / Inference và Execution")
    st.caption(
        "Bao: inference được tính trong planning · "
        "Balsa: inference/plan-search tách riêng trước PostgreSQL planning · "
        "PostgreSQL (Bao eval) và PostgreSQL (Balsa eval): baseline tương ứng của từng evaluation pipeline."
    )
    render_figure4_like_overview()


# ============================================================
# STRUCTURAL NOVELTY — SPLIT LEVEL
# ============================================================

def build_novelty_split_summary() -> pd.DataFrame:
    if novelty_df is None or novelty_df.empty:
        return pd.DataFrame()

    records = []
    for split_type in SPLIT_ORDER:
        rows = novelty_df[(novelty_df["split_type"] == split_type) & (novelty_df["split_id"] == 1)].copy()
        if rows.empty:
            continue
        vals = pd.to_numeric(rows["novelty"], errors="coerce").dropna()
        same_family = (
            rows["family"].astype(str).str.lower()
            == rows["nearest_train_family"].astype(str).str.lower()
        )
        records.append({
            "Split": SPLIT_LABEL[split_type],
            "Test queries": len(rows),
            "Median Novelty": float(vals.median()),
            "Mean Novelty": float(vals.mean()),
            "Max Novelty": float(vals.max()),
            "Nearest Train cùng family": float(same_family.mean() * 100.0),
        })
    return pd.DataFrame(records)


def render_novelty_distribution() -> None:
    summary = build_novelty_split_summary()
    if summary.empty:
        st.warning("Không có novelty.csv để tổng hợp theo split.")
        return

    left, right = st.columns([1.1, 1])
    with left:
        st.dataframe(
            summary,
            hide_index=True,
            width="stretch",
            column_config={
                "Median Novelty": st.column_config.NumberColumn(format="%.6f"),
                "Mean Novelty": st.column_config.NumberColumn(format="%.6f"),
                "Max Novelty": st.column_config.NumberColumn(format="%.6f"),
                "Nearest Train cùng family": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
    with right:
        chart_df = summary[["Split", "Median Novelty"]].copy()
        spec = {
            "mark": {"type": "bar"},
            "encoding": {
                "x": {"field": "Split", "type": "nominal", "sort": [SPLIT_LABEL[x] for x in SPLIT_ORDER], "title": None},
                "y": {"field": "Median Novelty", "type": "quantitative", "title": "Median Structural Novelty"},
                "tooltip": [
                    {"field": "Split", "type": "nominal"},
                    {"field": "Median Novelty", "type": "quantitative", "format": ".6f"},
                ],
            },
        }
        st.vega_lite_chart(chart_df, spec, use_container_width=True)


def split_scatter_data(df: pd.DataFrame | None, model_key: str, split_type: str) -> pd.DataFrame:
    rows = split_comparable_rows(df, model_key, split_type)
    if rows.empty:
        return pd.DataFrame()
    ratio_col = f"{model_key}_vs_pg_ratio"
    rows["novelty"] = pd.to_numeric(rows["novelty"], errors="coerce")
    rows["ratio"] = pd.to_numeric(rows[ratio_col], errors="coerce")
    rows = rows[rows["novelty"].notna() & rows["ratio"].notna() & (rows["ratio"] > 0)]
    return rows[["query_id", "novelty", "ratio"]].copy()


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


def render_split_scatter(df: pd.DataFrame | None, model_key: str, model_label: str, split_type: str, spearman_df: pd.DataFrame | None) -> None:
    rows = split_scatter_data(df, model_key, split_type)
    if rows.empty:
        st.warning(f"Không đủ dữ liệu {model_label} cho {SPLIT_LABEL[split_type]}.")
        return

    spec = {
        "layer": [
            {"mark": {"type": "rule", "strokeDash": [6, 4]}, "encoding": {"y": {"datum": 1}}},
            {
                "mark": {"type": "point", "filled": True, "size": 80},
                "encoding": {
                    "x": {"field": "novelty", "type": "quantitative", "title": "Structural Novelty"},
                    "y": {
                        "field": "ratio",
                        "type": "quantitative",
                        "scale": {"type": "log"},
                        "title": f"{model_label} / PostgreSQL · E2E",
                    },
                    "tooltip": [
                        {"field": "query_id", "type": "nominal", "title": "Query"},
                        {"field": "novelty", "type": "quantitative", "format": ".6f", "title": "Novelty"},
                        {"field": "ratio", "type": "quantitative", "format": ".3f", "title": f"{model_label}/PG"},
                    ],
                },
            },
        ]
    }
    st.vega_lite_chart(rows, spec, use_container_width=True)
    text = spearman_text(spearman_df, split_type)
    if text:
        st.caption(f"Spearman: {text} · Đường 1× là PostgreSQL baseline.")


def _group_row(df: pd.DataFrame | None, split_type: str, group: str) -> pd.Series | None:
    if df is None or df.empty:
        return None
    rows = df[(df["split_type"] == split_type) & (df["novelty_group"] == group)]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_group_comparison_table() -> pd.DataFrame:
    records = []
    for split_type in SPLIT_ORDER:
        bao_low = _group_row(bao_group_df, split_type, "Low")
        bao_high = _group_row(bao_group_df, split_type, "High")
        balsa_low = _group_row(balsa_group_df, split_type, "Low")
        balsa_high = _group_row(balsa_group_df, split_type, "High")

        def _value(row, col):
            if row is None or col not in row or pd.isna(row[col]):
                return None
            return float(row[col])

        bao_low_ratio = _value(bao_low, "median_bao_pg_ratio")
        bao_high_ratio = _value(bao_high, "median_bao_pg_ratio")
        balsa_low_ratio = _value(balsa_low, "median_balsa_pg_ratio")
        balsa_high_ratio = _value(balsa_high, "median_balsa_pg_ratio")

        records.append({
            "Split": SPLIT_LABEL[split_type],
            "Bao Low": bao_low_ratio,
            "Bao High": bao_high_ratio,
            "Bao High/Low": (bao_high_ratio / bao_low_ratio) if bao_low_ratio and bao_high_ratio else None,
            "Balsa Low": balsa_low_ratio,
            "Balsa High": balsa_high_ratio,
            "Balsa High/Low": (balsa_high_ratio / balsa_low_ratio) if balsa_low_ratio and balsa_high_ratio else None,
        })
    return pd.DataFrame(records)


def build_group_chart_data(df: pd.DataFrame | None, model_key: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    ratio_col = f"median_{model_key}_pg_ratio"
    if ratio_col not in df.columns:
        return pd.DataFrame()
    rows = df[df["split_type"].isin(SPLIT_ORDER)].copy()
    rows["Split"] = rows["split_type"].map(SPLIT_LABEL)
    rows["Median ratio"] = pd.to_numeric(rows[ratio_col], errors="coerce")
    return rows[["Split", "novelty_group", "Median ratio"]].dropna()


def render_group_ratio_chart(df: pd.DataFrame | None, model_key: str, label: str) -> None:
    rows = build_group_chart_data(df, model_key)
    if rows.empty:
        st.warning(f"Không có group summary cho {label}.")
        return
    spec = {
        "mark": {"type": "bar"},
        "encoding": {
            "x": {
                "field": "Split",
                "type": "nominal",
                "sort": [SPLIT_LABEL[x] for x in SPLIT_ORDER],
                "title": None,
            },
            "xOffset": {
                "field": "novelty_group",
                "sort": ["Low", "Medium", "High"],
            },
            "y": {
                "field": "Median ratio",
                "type": "quantitative",
                "title": f"Median {label}/PG · E2E",
            },
            "color": {
                "field": "novelty_group",
                "type": "nominal",
                "sort": ["Low", "Medium", "High"],
                "title": "Novelty group",
            },
            "tooltip": [
                {"field": "Split", "type": "nominal"},
                {"field": "novelty_group", "type": "nominal", "title": "Group"},
                {"field": "Median ratio", "type": "quantitative", "format": ".3f"},
            ],
        },
    }
    st.vega_lite_chart(rows, spec, use_container_width=True)


def build_spearman_comparison_table() -> pd.DataFrame:
    records = []
    for split_type in SPLIT_ORDER:
        record = {"Split": SPLIT_LABEL[split_type]}
        for label, df in [("Bao", bao_spearman_df), ("Balsa", balsa_spearman_df)]:
            if df is None or df.empty:
                record[f"{label} ρ"] = None
                record[f"{label} p"] = None
                continue
            key = "split_type" if "split_type" in df.columns else "group"
            rows = df[df[key] == split_type]
            if rows.empty:
                record[f"{label} ρ"] = None
                record[f"{label} p"] = None
            else:
                row = rows.iloc[0]
                record[f"{label} ρ"] = float(row["rho"])
                record[f"{label} p"] = float(row["p_value"])
        records.append(record)
    return pd.DataFrame(records)


def build_extension_evidence_table() -> pd.DataFrame:
    evidence = []
    summary = build_novelty_split_summary()
    if not summary.empty:
        bq = summary[summary["Split"] == "BaseQuery-1"]
        if not bq.empty:
            row = bq.iloc[0]
            evidence.append({
                "Evidence": "Structural gap",
                "Observed result": (
                    f"BaseQuery-1 median Novelty = {row['Median Novelty']:.3f}; "
                    f"nearest Train cùng family = {row['Nearest Train cùng family']:.1f}%"
                ),
            })

    group_table = build_group_comparison_table()
    rnd = group_table[group_table["Split"] == "Random-1"] if not group_table.empty else pd.DataFrame()
    if not rnd.empty:
        row = rnd.iloc[0]
        evidence.append({
            "Evidence": "High-vs-Low on Random-1",
            "Observed result": (
                f"Bao High/Low = {row['Bao High/Low']:.2f}×; "
                f"Balsa High/Low = {row['Balsa High/Low']:.2f}×"
            ),
        })

    if bao_cases_df is not None and not bao_cases_df.empty:
        same = bao_cases_df[(bao_cases_df["split_type"] == "LOO") & (bao_cases_df["query_id"].isin(["q19a", "q11b"]))]
        if len(same) == 2:
            parts = []
            for _, row in same.sort_values("query_id").iterrows():
                parts.append(f"{row['query_id']}: Novelty {float(row['novelty']):.3f}, Bao/PG {float(row['bao_vs_pg_ratio']):.2f}×")
            evidence.append({"Evidence": "Same Novelty, different outcome", "Observed result": " · ".join(parts)})

    if balsa_cases_df is not None and not balsa_cases_df.empty:
        rows = balsa_cases_df[balsa_cases_df["case_type"].isin(["LOW_NOVELTY_WORST_SLOWDOWN", "BALSA_TIMEOUT"])]
        if not rows.empty:
            row = rows.iloc[0]
            ratio = row.get("balsa_vs_pg_ratio")
            ratio_text = f"{float(ratio):.2f}×" if pd.notna(ratio) else str(row.get("balsa_status", "N/A"))
            evidence.append({
                "Evidence": "Low/Medium Novelty counterexample",
                "Observed result": f"{row['query_id']}: Novelty {float(row['novelty']):.3f}, Balsa result = {ratio_text}",
            })

    return pd.DataFrame(evidence)


def render_structural_novelty_extension() -> None:
    st.markdown("## ② Structural Novelty — phân tích generalization theo split")

    st.markdown("### 2.1 Structural Train–Test gap")
    render_novelty_distribution()

    st.markdown("### 2.2 Performance theo Low / Medium / High Novelty")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Bao")
        render_group_ratio_chart(bao_group_df, "bao", "Bao")
    with right:
        st.markdown("#### Balsa")
        render_group_ratio_chart(balsa_group_df, "balsa", "Balsa")

    group_table = build_group_comparison_table()
    if not group_table.empty:
        st.dataframe(
            group_table,
            hide_index=True,
            width="stretch",
            column_config={
                "Bao Low": st.column_config.NumberColumn(format="%.3f×"),
                "Bao High": st.column_config.NumberColumn(format="%.3f×"),
                "Bao High/Low": st.column_config.NumberColumn(format="%.2f×"),
                "Balsa Low": st.column_config.NumberColumn(format="%.3f×"),
                "Balsa High": st.column_config.NumberColumn(format="%.3f×"),
                "Balsa High/Low": st.column_config.NumberColumn(format="%.2f×"),
            },
        )

    st.markdown("### 2.3 Spearman: Novelty ↔ E2E ratio")
    spearman_table = build_spearman_comparison_table()
    st.dataframe(
        spearman_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Bao ρ": st.column_config.NumberColumn(format="%.3f"),
            "Bao p": st.column_config.NumberColumn(format="%.3f"),
            "Balsa ρ": st.column_config.NumberColumn(format="%.3f"),
            "Balsa p": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    with st.expander("Xem scatter theo từng split", expanded=False):
        tabs = st.tabs([SPLIT_LABEL[x] for x in SPLIT_ORDER])
        for tab, split_type in zip(tabs, SPLIT_ORDER):
            with tab:
                left, right = st.columns(2)
                with left:
                    st.markdown("#### Bao")
                    render_split_scatter(bao_df, "bao", "Bao", split_type, bao_spearman_df)
                with right:
                    st.markdown("#### Balsa")
                    render_split_scatter(balsa_df, "balsa", "Balsa", split_type, balsa_spearman_df)

    st.markdown("### 2.4 Tóm tắt bằng chứng")
    evidence = build_extension_evidence_table()
    if not evidence.empty:
        st.dataframe(evidence, hide_index=True, width="stretch")


# ============================================================
# QUERY-LEVEL CASE STUDY
# ============================================================

@st.cache_data
def load_balsa_hinted_sql(split_type: str, run_id: str, query_id: str) -> Tuple[str | None, str | None]:
    split_dir = BALSA_RUNS_ROOT / SPLIT_LABEL.get(split_type, split_type)
    stem = query_id.lower().removeprefix("q")
    preferred = split_dir / str(run_id) / "evaluation" / "plans" / f"{stem}_repeat_0_hinted.sql"
    if preferred.exists():
        return preferred.read_text(encoding="utf-8"), str(preferred)

    matches = list(split_dir.glob(f"*/evaluation/plans/{stem}_repeat_0_hinted.sql")) if split_dir.exists() else []
    if matches:
        path = matches[0]
        return path.read_text(encoding="utf-8"), str(path)
    return None, None


def extract_hint_block(sql_text: str | None) -> str | None:
    if not sql_text:
        return None
    match = re.search(r"/\*\+(.*?)\*/", sql_text, flags=re.DOTALL)
    if not match:
        return None
    return "\n".join(line.strip() for line in match.group(1).strip().splitlines() if line.strip())


def render_query_case_study() -> None:
    st.markdown("## ③ Query Case Study — drill-down sau khi đã hiểu kết quả theo split")
    st.caption(
        "Selector chỉ xuất hiện ở đây. Phần này không dùng để tổng hợp cả split; "
        "nó minh họa một query cụ thể để giải thích quyết định Bao/Balsa và Structural Novelty."
    )

    control1, control2 = st.columns(2)
    with control1:
        selected_split_type = st.selectbox(
            "Chọn split cho case study",
            SPLIT_ORDER,
            format_func=lambda x: SPLIT_LABEL.get(x, x),
            key="case_split",
        )

    selected_split_id = 1
    train_queries, test_queries = get_split_queries(selected_split_type, selected_split_id)
    defaults = {"LOO": "q11b", "Random": "q6c", "BaseQuery": "q15b"}
    preferred = defaults.get(selected_split_type)
    default_idx = test_queries.index(preferred) if preferred in test_queries else 0

    with control2:
        selected_query = st.selectbox(
            "Chọn Test Query",
            test_queries,
            index=default_idx,
            key="case_query",
        )

    result = calculate_novelty(selected_query, selected_split_type, selected_split_id)
    q33, q67 = novelty_thresholds(selected_split_type, selected_split_id)
    level = novelty_level(result["novelty"], q33, q67)
    query_feature = feature_map[selected_query]
    query_sql, query_sql_path = load_query_sql(selected_query)
    bao_row = selected_performance_row(bao_df, selected_split_type, selected_split_id, selected_query)
    balsa_row = selected_performance_row(balsa_df, selected_split_type, selected_split_id, selected_query)

    st.markdown("### 3.1 Query và vị trí của nó so với Training Set")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Query", selected_query)
    m2.metric("Tables", query_feature["n_tables"])
    m3.metric("Joins", query_feature["n_joins"])
    m4.metric("Filters", query_feature["n_filters"])
    m5.metric("Novelty", f"{result['novelty']:.6f}")

    st.caption(f"Nearest Train Query: {result['nearest']} · Novelty level: {level}")
    with st.expander("Xem SQL gốc", expanded=False):
        st.code(query_sql, language="sql")
        st.caption(query_sql_path)

    st.dataframe(
        result["candidate_df"].head(5),
        hide_index=True,
        width="stretch",
        column_config={
            "d_table": st.column_config.NumberColumn(format="%.6f"),
            "d_join": st.column_config.NumberColumn(format="%.6f"),
            "d_filter": st.column_config.NumberColumn(format="%.6f"),
            "D(q,h)": st.column_config.NumberColumn(format="%.6f"),
        },
    )

    st.markdown("### 3.2 Bao và Balsa đã quyết định gì với query này?")
    left, right = st.columns(2)

    with left:
        st.markdown("#### Bao")
        if bao_row is None:
            st.warning("Không có Bao artifact cho query này.")
        else:
            chosen = str(bao_row.get("bao_chosen_plan_id", "N/A"))
            chosen_ids = str(bao_row.get("bao_chosen_plan_ids", ""))
            st.metric("Chosen hint arm", chosen)
            st.markdown(f"**Ý nghĩa hint:** {BAO_ARM_HINTS.get(chosen, 'Không có mapping mô tả trong app.')} ")
            if chosen_ids and chosen_ids.lower() != "nan":
                st.caption(f"Arms qua các repeat: {chosen_ids}")
            p1, p2, p3 = st.columns(3)
            p1.metric("Planning + inference", format_ms(bao_row.get("bao_planning_ms")))
            p2.metric("Execution", format_ms(bao_row.get("bao_execution_ms")))
            p3.metric("E2E / PG", format_ratio(bao_row.get("bao_vs_pg_ratio")))

    with right:
        st.markdown("#### Balsa")
        if balsa_row is None:
            st.warning("Không có Balsa artifact cho query này.")
        else:
            plan_id = str(balsa_row.get("balsa_chosen_plan_id", "N/A"))
            run_id = str(balsa_row.get("balsa_run_id", ""))
            st.metric("Chosen plan id", plan_id)
            p1, p2, p3 = st.columns(3)
            p1.metric("Inference / plan search", format_ms(balsa_row.get("balsa_inference_ms")))
            p2.metric("PG planning", format_ms(balsa_row.get("balsa_planning_ms")))
            p3.metric("E2E / PG", format_ratio(balsa_row.get("balsa_vs_pg_ratio")))

            hinted_sql, hinted_path = load_balsa_hinted_sql(selected_split_type, run_id, selected_query)
            hint_block = extract_hint_block(hinted_sql)
            if hint_block:
                with st.expander("Xem physical-plan hints Balsa chọn", expanded=False):
                    st.code(hint_block, language="text")
                    if hinted_path:
                        st.caption(hinted_path)
            elif hinted_sql:
                with st.expander("Xem hinted SQL của Balsa", expanded=False):
                    st.code(hinted_sql, language="sql")
                    if hinted_path:
                        st.caption(hinted_path)
            else:
                st.caption("Không tìm thấy hinted SQL artifact trong repo hiện tại.")



# ============================================================
# MAIN
# ============================================================

st.title("Bao · Balsa · PostgreSQL — Tái hiện benchmark và mở rộng Structural Novelty")
st.caption("JOB · Paper benchmark → kết quả tái hiện theo split → Structural Novelty → query case study")

st.markdown(
    """
**Luồng trình bày:** nhóm bắt đầu từ cách paper benchmark LQO ở cấp split, tái hiện bằng output thực nghiệm của Bao/Balsa/PostgreSQL;
sau đó mới đặt câu hỏi mở rộng về structural generalization gap và dùng Structural Novelty để phân tích.
"""
)

render_paper_protocol_replay()

st.divider()
render_structural_novelty_extension()

st.divider()
render_query_case_study()

st.caption("Team14 · JOB · Paper protocol → Team evaluation → Structural Novelty → Case study")
