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
    page_title="Đánh giá tối ưu hóa truy vấn bằng học máy",
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


RAW_CSV_LABELS = {'query_id': 'Mã truy vấn', 'checkpoint_id': 'Mốc mô hình', 'planning_ms': 'Lập kế hoạch (ms)', 'execution_ms': 'Thực thi (ms)', 'inference_ms': 'Suy luận (ms)', 'end_to_end_ms': 'Xử lý đầu cuối (ms)', 'status': 'Trạng thái', 'chosen_plan_id': 'Mã kế hoạch được chọn', 'chosen_plan_ids': 'Các mã kế hoạch được chọn', 'timeout_ms': 'Ngưỡng thời gian (ms)'}

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
                "Truy vấn huấn luyện": train_query,
                "Nhóm truy vấn gốc": feature_map[train_query]["family"],
                "d_table": details["d_table"],
                "d_join": details["d_join"],
                "d_filter": details["d_filter"],
                "D(q,h)": details["distance"],
            }
        )

    candidate_df = pd.DataFrame(records).sort_values(
        by=["D(q,h)", "Truy vấn huấn luyện"],
        key=lambda col: col if col.name != "Truy vấn huấn luyện" else col.map(query_sort_key),
    )
    candidate_df = candidate_df.reset_index(drop=True)
    candidate_df.insert(0, "Thứ hạng", range(1, len(candidate_df) + 1))

    nearest = candidate_df.iloc[0]

    return {
        "train_size": len(train_queries),
        "test_size": len(test_queries),
        "nearest": str(nearest["Truy vấn huấn luyện"]),
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
# EXPERIMENT SCENARIO + DATA PREPARATION
# ============================================================

def _feature_list_markdown(values: Set[str], empty_text: str = "Không có") -> str:
    if not values:
        return f"- {empty_text}"
    return "\n".join(f"- `{value}`" for value in sorted(values))


def render_experiment_scenario_and_data() -> None:
    st.markdown("## Dữ liệu đầu vào và quy trình thực nghiệm")
    st.caption(
        "Phần này mô tả dữ liệu đi qua các bước nào trước khi xuất hiện ở kết quả thực nghiệm phía dưới."
    )

    # ---- Experiment workflow ----
    st.markdown("### Quy trình đánh giá")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("#### 1 · Chuẩn bị JOB")
        st.markdown(
            "**113 truy vấn · 33 nhóm truy vấn gốc**\n\n"
            "Tạo ba cách chia tập huấn luyện và tập kiểm tra: **LOO-1, Random-1, BaseQuery-1**."
        )

    with c2:
        st.markdown("#### 2 · Huấn luyện và mô hình")
        st.markdown(
            "**Bao:** dùng mô hình đã huấn luyện do tác giả cung cấp theo từng cách chia dữ liệu.\n\n"
            "**Balsa:** nhóm huấn luyện mô hình riêng cho từng cách chia, với ba vòng lặp huấn luyện."
        )

    with c3:
        st.markdown("#### 3 · Kiểm tra truy vấn")
        st.markdown(
            "**Bao:** chọn bộ gợi ý thực thi → PostgreSQL lập kế hoạch và thực thi.\n\n"
            "**Balsa:** chọn kế hoạch truy vấn → PostgreSQL lập kế hoạch và thực thi."
        )

    with c4:
        st.markdown("#### 4 · Ghi nhận kết quả")
        st.markdown(
            "Ghi nhận **thời gian lập kế hoạch, suy luận, thực thi và xử lý đầu cuối (E2E)** vào các tệp CSV. "
            "Những kết quả này được trình bày ở phần tái lập thực nghiệm."
        )

    st.markdown("### Dữ liệu huấn luyện")
    split_cols = st.columns(3)
    split_notes = {
        "LOO": "Mỗi nhóm truy vấn gốc có một biến thể truy vấn trong tập kiểm tra.",
        "Random": "Các truy vấn được phân ngẫu nhiên vào tập huấn luyện và tập kiểm tra.",
        "BaseQuery": "Các biến thể truy vấn cùng nhóm gốc chỉ nằm trong tập huấn luyện hoặc tập kiểm tra.",
    }

    for col, split_type in zip(split_cols, SPLIT_ORDER):
        train_queries, test_queries = get_split_queries(split_type, 1)
        with col:
            st.markdown(f"#### {SPLIT_LABEL[split_type]}")
            m1, m2 = st.columns(2)
            m1.metric("Huấn luyện", len(train_queries))
            m2.metric("Kiểm tra", len(test_queries))
            st.caption(split_notes[split_type])




def render_paper_encoding_and_pipeline() -> None:
    """Trình bày ngắn gọn đường đi của một truy vấn khi đánh giá Bao/Balsa.

    Chỉ minh họa quy trình theo bài báo; các phép đo nằm ở phần Kết quả bên dưới.
    """
    st.markdown("### Xử lý truy vấn trên tập kiểm tra")
    st.caption("Hai LQO sử dụng mô hình đã chuẩn bị ở bước huấn luyện để lựa chọn phương án cho truy vấn kiểm tra.")

    bao_col, balsa_col = st.columns(2)
    with bao_col:
        with st.container(border=True):
            st.markdown("#### Bao · Chọn bộ gợi ý")
            st.markdown(
                "**Truy vấn SQL** → PostgreSQL tạo kế hoạch ứng viên  \n"
                "↓  \n"
                "**Bao suy luận:** đánh giá biểu diễn kế hoạch, chọn bộ gợi ý  \n"
                "↓  \n"
                "**PostgreSQL:** lập kế hoạch theo gợi ý → thực thi truy vấn"
            )
            st.caption("Đo thời gian: lập kế hoạch (gồm suy luận Bao) + thực thi.")

    with balsa_col:
        with st.container(border=True):
            st.markdown("#### Balsa · Tìm kiếm kế hoạch")
            st.markdown(
                "**Truy vấn SQL** → biểu diễn truy vấn và kế hoạch  \n"
                "↓  \n"
                "**Balsa suy luận:** tìm kiếm, chọn kế hoạch thực thi  \n"
                "↓  \n"
                "**PostgreSQL:** lập kế hoạch tương ứng → thực thi truy vấn"
            )
            st.caption("Đo thời gian: suy luận/tìm kiếm + lập kế hoạch + thực thi.")


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
                "component": "Lập kế hoạch",
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
                        "component": "Lập kế hoạch",
                        "ratio": float(planning_ratio),
                    }
                )

            if method == "Balsa" and pd.notna(inference_ratio) and float(inference_ratio) > 0:
                planning_records.append(
                    {
                        "split_method": f"{split_label} · {method}",
                        "split": split_label,
                        "method": method,
                        "component": "Suy luận",
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
                    "Cách chia": SPLIT_LABEL[split_type],
                    "Phương pháp": method_label,
                    "Trung vị tỷ số E2E/PG": float(ratios.median()),
                    "Số truy vấn so sánh được": len(ratios),
                }
            )

    # Add PostgreSQL đối chứng rows.
    for split_type in SPLIT_ORDER:
        records.append(
            {
                "Cách chia": SPLIT_LABEL[split_type],
                "Phương pháp": "PostgreSQL",
                "Trung vị tỷ số E2E/PG": 1.0,
                "Số truy vấn so sánh được": None,
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    result["Cách chia"] = pd.Categorical(
        result["Cách chia"],
        [SPLIT_LABEL[x] for x in SPLIT_ORDER],
        ordered=True,
    )
    result["Phương pháp"] = pd.Categorical(result["Phương pháp"], METHOD_ORDER, ordered=True)
    return result.sort_values(["Cách chia", "Phương pháp"]).reset_index(drop=True)


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
    "arm_0": "Giữ nguyên các toán tử kết hợp bảng và quét dữ liệu (phương án mặc định).",
    "arm_1": "Vô hiệu hóa phép kết hợp vòng lặp lồng nhau (Nested Loop).",
    "arm_2": "Vô hiệu hóa Merge Join và Index Scan.",
    "arm_3": "Vô hiệu hóa Nested Loop, Merge Join và Index Scan.",
    "arm_4": "Vô hiệu hóa Merge Join.",
    "MULTIPLE": "Bao chọn nhiều phương án qua các lần chạy lặp; xem danh sách chosen_plan_ids.",
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
        record = {"Cách chia": SPLIT_LABEL[split_type]}

        bao_rows = split_comparable_rows(bao_df, "bao", split_type)
        if not bao_rows.empty:
            ratios = pd.to_numeric(bao_rows["bao_vs_pg_ratio"], errors="coerce").dropna()
            record["Số truy vấn so sánh được (Bao)"] = len(bao_rows)
            record["Trung vị E2E/PG (Bao)"] = float(ratios.median()) if not ratios.empty else None
            record["Số truy vấn Bao nhanh hơn"] = int((ratios < 1).sum()) if not ratios.empty else 0
            record["Tổng E2E của Bao (giây)"] = pd.to_numeric(bao_rows["bao_end_to_end_ms"], errors="coerce").sum() / 1000.0
            record["Tổng E2E PostgreSQL khi so sánh Bao (giây)"] = pd.to_numeric(bao_rows["pg_end_to_end_ms"], errors="coerce").sum() / 1000.0
        else:
            record.update({
                "Số truy vấn so sánh được (Bao)": 0,
                "Trung vị E2E/PG (Bao)": None,
                "Số truy vấn Bao nhanh hơn": 0,
                "Tổng E2E của Bao (giây)": None,
                "Tổng E2E PostgreSQL khi so sánh Bao (giây)": None,
            })

        balsa_rows = split_comparable_rows(balsa_df, "balsa", split_type)
        if not balsa_rows.empty:
            ratios = pd.to_numeric(balsa_rows["balsa_vs_pg_ratio"], errors="coerce").dropna()
            record["Số truy vấn so sánh được (Balsa)"] = len(balsa_rows)
            record["Trung vị E2E/PG (Balsa)"] = float(ratios.median()) if not ratios.empty else None
            record["Số truy vấn Balsa nhanh hơn"] = int((ratios < 1).sum()) if not ratios.empty else 0
            record["Tổng E2E của Balsa (giây)"] = pd.to_numeric(balsa_rows["balsa_end_to_end_ms"], errors="coerce").sum() / 1000.0
            record["Tổng E2E PostgreSQL khi so sánh Balsa (giây)"] = pd.to_numeric(balsa_rows["pg_end_to_end_ms"], errors="coerce").sum() / 1000.0
        else:
            record.update({
                "Số truy vấn so sánh được (Balsa)": 0,
                "Trung vị E2E/PG (Balsa)": None,
                "Số truy vấn Balsa nhanh hơn": 0,
                "Tổng E2E của Balsa (giây)": None,
                "Tổng E2E PostgreSQL khi so sánh Balsa (giây)": None,
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
        {"Phương pháp": "PostgreSQL đối chứng", "Thành phần": "Lập kế hoạch", "Thời gian (giây)": pg_planning},
        {"Phương pháp": "PostgreSQL đối chứng", "Thành phần": "Thực thi", "Thời gian (giây)": pg_execution},
    ])

    if model_key == "bao":
        planning = pd.to_numeric(df["bao_planning_ms"], errors="coerce").sum() / 1000.0
        execution = pd.to_numeric(df["bao_execution_ms"], errors="coerce").sum() / 1000.0
        records.extend([
            {"Phương pháp": model_label, "Thành phần": "Lập kế hoạch và suy luận của Bao", "Thời gian (giây)": planning},
            {"Phương pháp": model_label, "Thành phần": "Thực thi", "Thời gian (giây)": execution},
        ])
    else:
        inference = pd.to_numeric(df["balsa_inference_ms"], errors="coerce").sum() / 1000.0
        planning = pd.to_numeric(df["balsa_planning_ms"], errors="coerce").sum() / 1000.0
        execution = pd.to_numeric(df["balsa_execution_ms"], errors="coerce").sum() / 1000.0
        records.extend([
            {"Phương pháp": model_label, "Thành phần": "Suy luận và tìm kế hoạch", "Thời gian (giây)": inference},
            {"Phương pháp": model_label, "Thành phần": "Lập kế hoạch bằng PostgreSQL", "Thời gian (giây)": planning},
            {"Phương pháp": model_label, "Thành phần": "Thực thi", "Thời gian (giây)": execution},
        ])

    return pd.DataFrame(records)


def render_component_chart(chart_df: pd.DataFrame) -> None:
    if chart_df.empty:
        st.warning("Không đủ dữ liệu để dựng biểu đồ.")
        return

    spec = {
        "mark": {"type": "bar"},
        "encoding": {
            "y": {"field": "Phương pháp", "type": "nominal", "title": None},
            "x": {
                "aggregate": "sum",
                "field": "Thời gian (giây)",
                "type": "quantitative",
                "title": "Tổng thời gian trên các truy vấn có thể so sánh (giây)",
            },
            "color": {"field": "Thành phần", "type": "nominal", "title": None},
            "tooltip": [
                {"field": "Phương pháp", "type": "nominal"},
                {"field": "Thành phần", "type": "nominal"},
                {"field": "Thời gian (giây)", "type": "quantitative", "format": ".3f"},
            ],
        },
    }
    st.vega_lite_chart(chart_df, spec, use_container_width=True)


def render_model_split_panel(df: pd.DataFrame | None, model_key: str, model_label: str, split_type: str) -> None:
    rows = split_comparable_rows(df, model_key, split_type)
    if rows.empty:
        st.warning(f"Không có dữ liệu có thể so sánh cho {model_label} · {SPLIT_LABEL[split_type]}.")
        return

    ratio_col = f"{model_key}_vs_pg_ratio"
    model_e2e_col = f"{model_key}_end_to_end_ms"
    ratios = pd.to_numeric(rows[ratio_col], errors="coerce").dropna()
    model_total = pd.to_numeric(rows[model_e2e_col], errors="coerce").sum()
    pg_total = pd.to_numeric(rows["pg_end_to_end_ms"], errors="coerce").sum()
    total_ratio = model_total / pg_total if pg_total > 0 else None
    wins = int((ratios < 1).sum()) if not ratios.empty else 0

    if model_key == "bao":
        st.markdown("**Quy trình được tái hiện:** mô hình do tác giả cung cấp → Bao chọn bộ gợi ý → PostgreSQL lập kế hoạch → PostgreSQL thực thi")
    else:
        st.markdown("**Quy trình được tái hiện:** mô hình do nhóm huấn luyện → Balsa tìm kế hoạch → chọn kế hoạch và gợi ý → PostgreSQL lập kế hoạch → PostgreSQL thực thi")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số truy vấn so sánh được queries", len(rows))
    m2.metric(f"{model_label}: tổng E2E", format_seconds_from_ms(model_total))
    m3.metric("PostgreSQL: tổng E2E đối chứng", format_seconds_from_ms(pg_total))
    m4.metric("Tỷ số tổng E2E / PostgreSQL", "N/A" if total_ratio is None else f"{total_ratio:.3f}×")

    st.caption(
        f"Trung vị tỷ số theo truy vấn {model_label}/PG = {float(ratios.median()):.3f}× · "
        f"{model_label} thắng {wins}/{len(rows)} truy vấn có thể so sánh."
    )
    render_component_chart(component_totals(rows, model_key, model_label))

    if model_key == "bao":
        checkpoints = sorted(rows.get("bao_checkpoint_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        if checkpoints:
            st.caption(f"Mốc mô hình: {', '.join(checkpoints)} · Thời gian suy luận của Bao đã nằm trong thời gian lập kế hoạch của Bao.")
    else:
        checkpoints = sorted(rows.get("balsa_checkpoint_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        if checkpoints:
            st.caption(f"Mốc mô hình: {', '.join(checkpoints)} · Phân tích chính: huấn luyện ba vòng lặp do giới hạn tài nguyên.")
        if "pg_baseline_session" in rows.columns:
            sessions = sorted(rows["pg_baseline_session"].dropna().astype(str).unique().tolist())
            if sessions:
                st.caption(f"Nguồn dữ liệu PostgreSQL đối chứng: {', '.join(sessions)}.")


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
                (f"{split_label} · PostgreSQL (đối chứng Bao)", "Lập kế hoạch", pd.to_numeric(bao_rows["pg_planning_ms"], errors="coerce").sum() / 1000.0),
                (f"{split_label} · Bao", "Lập kế hoạch và suy luận của Bao", pd.to_numeric(bao_rows["bao_planning_ms"], errors="coerce").sum() / 1000.0),
            ]
            for row_label, component, seconds in rows:
                planning_records.append({"Phương pháp và cách chia": row_label, "Thành phần": component, "Thời gian (giây)": float(seconds)})
                if row_label not in row_order:
                    row_order.append(row_label)

            execution_records.extend([
                {
                    "Phương pháp và cách chia": f"{split_label} · PostgreSQL (đối chứng Bao)",
                    "Thời gian (giây)": float(pd.to_numeric(bao_rows["pg_execution_ms"], errors="coerce").sum() / 1000.0),
                },
                {
                    "Phương pháp và cách chia": f"{split_label} · Bao",
                    "Thời gian (giây)": float(pd.to_numeric(bao_rows["bao_execution_ms"], errors="coerce").sum() / 1000.0),
                },
            ])

        if not balsa_rows.empty:
            pg_row = f"{split_label} · PostgreSQL (đối chứng Balsa)"
            balsa_row = f"{split_label} · Balsa"
            planning_records.extend([
                {
                    "Phương pháp và cách chia": pg_row,
                    "Thành phần": "Lập kế hoạch",
                    "Thời gian (giây)": float(pd.to_numeric(balsa_rows["pg_planning_ms"], errors="coerce").sum() / 1000.0),
                },
                {
                    "Phương pháp và cách chia": balsa_row,
                    "Thành phần": "Lập kế hoạch bằng PostgreSQL",
                    "Thời gian (giây)": float(pd.to_numeric(balsa_rows["balsa_planning_ms"], errors="coerce").sum() / 1000.0),
                },
                {
                    "Phương pháp và cách chia": balsa_row,
                    "Thành phần": "Suy luận và tìm kế hoạch của Balsa",
                    "Thời gian (giây)": float(pd.to_numeric(balsa_rows["balsa_inference_ms"], errors="coerce").sum() / 1000.0),
                },
            ])
            for row_label in [pg_row, balsa_row]:
                if row_label not in row_order:
                    row_order.append(row_label)

            execution_records.extend([
                {
                    "Phương pháp và cách chia": pg_row,
                    "Thời gian (giây)": float(pd.to_numeric(balsa_rows["pg_execution_ms"], errors="coerce").sum() / 1000.0),
                },
                {
                    "Phương pháp và cách chia": balsa_row,
                    "Thời gian (giây)": float(pd.to_numeric(balsa_rows["balsa_execution_ms"], errors="coerce").sum() / 1000.0),
                },
            ])

    return pd.DataFrame(planning_records), pd.DataFrame(execution_records), row_order


def render_figure4_like_overview() -> None:
    planning_df, execution_df, row_order = build_figure4_like_data()
    if planning_df.empty or execution_df.empty:
        st.warning("Không đủ dữ liệu để dựng biểu đồ tổng hợp.")
        return

    shared_y = {
        "field": "Phương pháp và cách chia",
        "type": "nominal",
        "sort": row_order,
        "title": None,
        "axis": {"labelLimit": 210},
    }

    spec = {
        "hconcat": [
            {
                "title": "Lập kế hoạch và suy luận",
                "width": 500,
                "height": {"step": 25},
                "data": {"values": planning_df.to_dict(orient="records")},
                "mark": {"type": "bar"},
                "encoding": {
                    "y": shared_y,
                    "x": {
                        "aggregate": "sum",
                        "field": "Thời gian (giây)",
                        "type": "quantitative",
                        "title": "Tổng thời gian trên truy vấn có thể so sánh (s)",
                    },
                    "color": {"field": "Thành phần", "type": "nominal", "title": None},
                    "tooltip": [
                        {"field": "Phương pháp và cách chia", "type": "nominal", "title": "Cách chia / phương pháp"},
                        {"field": "Thành phần", "type": "nominal"},
                        {"field": "Thời gian (giây)", "type": "quantitative", "format": ".3f", "title": "Thời gian (giây)"},
                    ],
                },
            },
            {
                "title": "Thực thi",
                "width": 500,
                "height": {"step": 25},
                "data": {"values": execution_df.to_dict(orient="records")},
                "mark": {"type": "bar"},
                "encoding": {
                    "y": shared_y,
                    "x": {
                        "field": "Thời gian (giây)",
                        "type": "quantitative",
                        "title": "Tổng thời gian trên truy vấn có thể so sánh (s)",
                    },
                    "tooltip": [
                        {"field": "Phương pháp và cách chia", "type": "nominal", "title": "Cách chia / phương pháp"},
                        {"field": "Thời gian (giây)", "type": "quantitative", "format": ".3f", "title": "Thời gian (giây)"},
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
            eval_isolation.append("không huấn luyện lại")
        if balsa_manifest.get("weights_unchanged") is True:
            eval_isolation.append("trọng số không thay đổi")

        rows.append({
            "Cách chia": split_label,
            "Huấn luyện / Kiểm tra": f"{len(train_queries)} / {len(test_queries)}",
            "Mốc mô hình Bao": bao_checkpoint,
            "Đánh giá Bao": _status_summary(bao_rows),
            "PostgreSQL (đối chứng Bao)": _status_summary(bao_pg_rows),
            "Mốc mô hình Balsa": balsa_checkpoint,
            "Đánh giá Balsa": _status_summary(balsa_rows),
            "PostgreSQL (đối chứng Balsa)": _status_summary(balsa_pg_rows),
            "Tách biệt huấn luyện và đánh giá Balsa": " · ".join(eval_isolation) if eval_isolation else "N/A",
        })

    return pd.DataFrame(rows)


def render_execution_evidence() -> None:
    st.markdown("### Nguồn thực nghiệm")
    evidence = build_execution_evidence_summary()
    st.dataframe(evidence, hide_index=True, width="stretch")
    st.caption(
        "Bao: mô hình do tác giả cung cấp, nhóm thực hiện đánh giá · "
        "Balsa: mô hình do nhóm huấn luyện, đánh giá độc lập · "
        "Nguồn số liệu thời gian: các tệp CSV đánh giá và PostgreSQL đối chứng trong kho mã nguồn."
    )


def render_experiment_csv_evidence() -> None:
    """Hiển thị cả ba cách chia trong từng bảng Bao và Balsa."""
    with st.expander("Dữ liệu chi tiết: mốc mô hình · lập kế hoạch · suy luận · thực thi · E2E", expanded=False):
        st.caption("Các bảng hiển thị đồng thời LOO-1, Random-1 và BaseQuery-1 (lần chia số 1).")
        bao_tab, balsa_tab = st.tabs(["Kết quả Bao", "Kết quả Balsa"])

        def show_rows(data: pd.DataFrame, columns: list[str]) -> None:
            if data.empty:
                st.warning("Không tìm thấy dữ liệu thực nghiệm tương ứng.")
                return
            visible = [col for col in columns if col in data.columns]
            result = data[visible].rename(columns={"_split_label": "Cách chia", **RAW_CSV_LABELS})
            st.dataframe(result, hide_index=True, width="stretch")

        def all_bao_splits(path: Path) -> pd.DataFrame:
            data = _read_csv_if_exists(path)
            if data.empty or not {"split_type", "split_id"}.issubset(data.columns):
                return pd.DataFrame()
            data = data[
                data["split_type"].isin(SPLIT_ORDER)
                & (pd.to_numeric(data["split_id"], errors="coerce") == 1)
            ].copy()
            data["_split_label"] = data["split_type"].map(SPLIT_LABEL)
            data["_split_order"] = data["split_type"].map({name: i for i, name in enumerate(SPLIT_ORDER)})
            if "query_id" in data.columns:
                data["_query_order"] = data["query_id"].astype(str).map(query_sort_key)
                data = data.sort_values(["_split_order", "_query_order"], kind="stable")
            return data

        def all_balsa_splits(filename: str) -> pd.DataFrame:
            frames = []
            for split_type in SPLIT_ORDER:
                path = BALSA_RUN_DIRS[split_type] / filename
                data = _read_csv_if_exists(path)
                if data.empty:
                    continue
                # Các tệp của Balsa nằm trong thư mục riêng của từng cách chia.
                if "split_type" in data.columns:
                    data = data[data["split_type"] == split_type].copy()
                if "split_id" in data.columns:
                    data = data[pd.to_numeric(data["split_id"], errors="coerce") == 1].copy()
                if data.empty:
                    continue
                data["_split_label"] = SPLIT_LABEL[split_type]
                frames.append(data)
            if not frames:
                return pd.DataFrame()
            result = pd.concat(frames, ignore_index=True)
            result["_split_order"] = result["_split_label"].map(
                {SPLIT_LABEL[name]: i for i, name in enumerate(SPLIT_ORDER)}
            )
            if "query_id" in result.columns:
                result["_query_order"] = result["query_id"].astype(str).map(query_sort_key)
                result = result.sort_values(["_split_order", "_query_order"], kind="stable")
            return result

        with bao_tab:
            st.caption(f"Nguồn Bao: {BAO_RAW_EVAL_PATH.relative_to(REPO_ROOT)}")
            show_rows(
                all_bao_splits(BAO_RAW_EVAL_PATH),
                ["_split_label", "query_id", "checkpoint_id", "planning_ms", "execution_ms",
                 "end_to_end_ms", "status", "chosen_plan_id", "chosen_plan_ids"],
            )
            with st.expander("PostgreSQL đối chứng của Bao", expanded=False):
                st.caption(str(BAO_RAW_PG_PATH.relative_to(REPO_ROOT)))
                show_rows(
                    all_bao_splits(BAO_RAW_PG_PATH),
                    ["_split_label", "query_id", "planning_ms", "execution_ms",
                     "end_to_end_ms", "status", "timeout_ms"],
                )

        with balsa_tab:
            st.caption("Nguồn Balsa: tệp evaluation_results.csv trong thư mục chạy của từng cách chia")
            show_rows(
                all_balsa_splits("evaluation_results.csv"),
                ["_split_label", "query_id", "checkpoint_id", "inference_ms",
                 "planning_ms", "execution_ms", "end_to_end_ms", "status", "chosen_plan_id"],
            )
            with st.expander("PostgreSQL đối chứng của Balsa", expanded=False):
                st.caption("Nguồn: postgres_results.csv trong thư mục chạy tương ứng")
                show_rows(
                    all_balsa_splits("postgres_results.csv"),
                    ["_split_label", "query_id", "planning_ms", "execution_ms",
                     "end_to_end_ms", "status", "timeout_ms"],
                )


def render_paper_protocol_replay() -> None:
    st.markdown("## Tái lập thực nghiệm")
    st.caption("JOB · LOO-1 · Random-1 · BaseQuery-1 · Đánh giá theo cách chia dữ liệu")

    # 1) Provenance / artifacts first
    render_execution_evidence()

    # 2) Numeric results before visualization
    st.markdown("### Kết quả")
    overview = build_split_overview_table()
    st.dataframe(
        overview,
        hide_index=True,
        width="stretch",
        column_config={
            "Trung vị E2E/PG (Bao)": st.column_config.NumberColumn(format="%.3f×"),
            "Trung vị E2E/PG (Balsa)": st.column_config.NumberColumn(format="%.3f×"),
            "Tổng E2E của Bao (giây)": st.column_config.NumberColumn(format="%.3f"),
            "Tổng E2E PostgreSQL khi so sánh Bao (giây)": st.column_config.NumberColumn(format="%.3f"),
            "Tổng E2E của Balsa (giây)": st.column_config.NumberColumn(format="%.3f"),
            "Tổng E2E PostgreSQL khi so sánh Balsa (giây)": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    render_experiment_csv_evidence()

    # 3) One combined Figure-4-like visualization
    st.markdown("### Trực quan hóa kết quả")
    st.caption("Biểu đồ phân rã thời gian lập kế hoạch, suy luận và thực thi theo từng cách chia dữ liệu.")
    st.caption(
        "Bao: thời gian suy luận được tính trong thời gian lập kế hoạch · "
        "Balsa: thời gian suy luận và tìm kế hoạch được tính riêng trước thời gian lập kế hoạch bằng PostgreSQL · "
        "PostgreSQL đối chứng được ghi nhận riêng cho từng quy trình đánh giá Bao và Balsa."
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
            "Cách chia": SPLIT_LABEL[split_type],
            "Số truy vấn kiểm tra": len(rows),
            "Trung vị điểm mức độ mới": float(vals.median()),
            "Trung bình điểm mức độ mới": float(vals.mean()),
            "Điểm mức độ mới lớn nhất": float(vals.max()),
            "Truy vấn gần nhất cùng nhóm (%)": float(same_family.mean() * 100.0),
        })
    return pd.DataFrame(records)


def render_novelty_distribution() -> None:
    summary = build_novelty_split_summary()
    if summary.empty:
        st.warning("Không tìm thấy novelty.csv để tổng hợp theo cách chia dữ liệu.")
        return

    left, right = st.columns([1.1, 1])
    with left:
        st.dataframe(
            summary,
            hide_index=True,
            width="stretch",
            column_config={
                "Trung vị điểm mức độ mới": st.column_config.NumberColumn(format="%.6f"),
                "Trung bình điểm mức độ mới": st.column_config.NumberColumn(format="%.6f"),
                "Điểm mức độ mới lớn nhất": st.column_config.NumberColumn(format="%.6f"),
                "Truy vấn gần nhất cùng nhóm (%)": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
    with right:
        chart_df = summary[["Cách chia", "Trung vị điểm mức độ mới"]].copy()
        spec = {
            "mark": {"type": "bar"},
            "encoding": {
                "x": {"field": "Cách chia", "type": "nominal", "sort": [SPLIT_LABEL[x] for x in SPLIT_ORDER], "title": None},
                "y": {"field": "Trung vị điểm mức độ mới", "type": "quantitative", "title": "Trung vị điểm mức độ mới về cấu trúc"},
                "tooltip": [
                    {"field": "Cách chia", "type": "nominal"},
                    {"field": "Trung vị điểm mức độ mới", "type": "quantitative", "format": ".6f"},
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
                    "x": {"field": "novelty", "type": "quantitative", "title": "Điểm mức độ mới về cấu trúc"},
                    "y": {
                        "field": "ratio",
                        "type": "quantitative",
                        "scale": {"type": "log"},
                        "title": f"Tỷ số thời gian E2E: {model_label}/PostgreSQL",
                    },
                    "tooltip": [
                        {"field": "query_id", "type": "nominal", "title": "Truy vấn"},
                        {"field": "novelty", "type": "quantitative", "format": ".6f", "title": "Điểm mức độ mới"},
                        {"field": "ratio", "type": "quantitative", "format": ".3f", "title": f"Tỷ số E2E của {model_label}/PostgreSQL"},
                    ],
                },
            },
        ]
    }
    st.vega_lite_chart(rows, spec, use_container_width=True)
    text = spearman_text(spearman_df, split_type)
    if text:
        st.caption(f"Spearman: {text} · Đường 1× là PostgreSQL đối chứng.")


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
            "Cách chia": SPLIT_LABEL[split_type],
            "Bao: mức thấp": bao_low_ratio,
            "Bao: mức cao": bao_high_ratio,
            "Bao: cao/thấp": (bao_high_ratio / bao_low_ratio) if bao_low_ratio and bao_high_ratio else None,
            "Balsa: mức thấp": balsa_low_ratio,
            "Balsa: mức cao": balsa_high_ratio,
            "Balsa: cao/thấp": (balsa_high_ratio / balsa_low_ratio) if balsa_low_ratio and balsa_high_ratio else None,
        })
    return pd.DataFrame(records)


def build_group_chart_data(df: pd.DataFrame | None, model_key: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    ratio_col = f"median_{model_key}_pg_ratio"
    if ratio_col not in df.columns:
        return pd.DataFrame()
    rows = df[df["split_type"].isin(SPLIT_ORDER)].copy()
    rows["Cách chia"] = rows["split_type"].map(SPLIT_LABEL)
    rows["Trung vị tỷ số E2E/PG"] = pd.to_numeric(rows[ratio_col], errors="coerce")
    rows["novelty_group"] = rows["novelty_group"].map({"Low": "Thấp", "Medium": "Trung bình", "High": "Cao"})
    return rows[["Cách chia", "novelty_group", "Trung vị tỷ số E2E/PG"]].dropna()


def render_group_ratio_chart(df: pd.DataFrame | None, model_key: str, label: str) -> None:
    rows = build_group_chart_data(df, model_key)
    if rows.empty:
        st.warning(f"Không có dữ liệu tổng hợp theo nhóm cho {label}.")
        return
    spec = {
        "mark": {"type": "bar"},
        "encoding": {
            "x": {
                "field": "Cách chia",
                "type": "nominal",
                "sort": [SPLIT_LABEL[x] for x in SPLIT_ORDER],
                "title": None,
            },
            "xOffset": {
                "field": "novelty_group",
                "sort": ["Thấp", "Trung bình", "Cao"],
            },
            "y": {
                "field": "Trung vị tỷ số E2E/PG",
                "type": "quantitative",
                "title": f"Trung vị tỷ số E2E của {label}/PostgreSQL",
            },
            "color": {
                "field": "novelty_group",
                "type": "nominal",
                "sort": ["Thấp", "Trung bình", "Cao"],
                "title": "Nhóm mức độ mới",
            },
            "tooltip": [
                {"field": "Cách chia", "type": "nominal"},
                {"field": "novelty_group", "type": "nominal", "title": "Mức độ mới"},
                {"field": "Trung vị tỷ số E2E/PG", "type": "quantitative", "format": ".3f"},
            ],
        },
    }
    st.vega_lite_chart(rows, spec, use_container_width=True)


def build_spearman_comparison_table() -> pd.DataFrame:
    records = []
    for split_type in SPLIT_ORDER:
        record = {"Cách chia": SPLIT_LABEL[split_type]}
        for label, df in [("Bao", bao_spearman_df), ("Balsa", balsa_spearman_df)]:
            if df is None or df.empty:
                record[f"{label} ρ"] = None
                record[f"{label} (p-value)"] = None
                continue
            key = "split_type" if "split_type" in df.columns else "group"
            rows = df[df[key] == split_type]
            if rows.empty:
                record[f"{label} ρ"] = None
                record[f"{label} (p-value)"] = None
            else:
                row = rows.iloc[0]
                record[f"{label} ρ"] = float(row["rho"])
                record[f"{label} (p-value)"] = float(row["p_value"])
        records.append(record)
    return pd.DataFrame(records)


def build_extension_evidence_table() -> pd.DataFrame:
    evidence = []
    summary = build_novelty_split_summary()
    if not summary.empty:
        bq = summary[summary["Cách chia"] == "BaseQuery-1"]
        if not bq.empty:
            row = bq.iloc[0]
            evidence.append({
                "Bằng chứng": "Khác biệt cấu trúc",
                "Kết quả quan sát": (
                    f"BaseQuery-1 trung vị điểm mức độ mới = {row['Trung vị điểm mức độ mới']:.3f}; "
                    f"truy vấn huấn luyện gần nhất cùng nhóm = {row['Truy vấn gần nhất cùng nhóm (%)']:.1f}%"
                ),
            })

    group_table = build_group_comparison_table()
    rnd = group_table[group_table["Cách chia"] == "Random-1"] if not group_table.empty else pd.DataFrame()
    if not rnd.empty:
        row = rnd.iloc[0]
        evidence.append({
            "Bằng chứng": "So sánh mức cao và thấp ở Random-1",
            "Kết quả quan sát": (
                f"Bao: cao/thấp = {row['Bao: cao/thấp']:.2f}×; "
                f"Balsa: cao/thấp = {row['Balsa: cao/thấp']:.2f}×"
            ),
        })

    if bao_cases_df is not None and not bao_cases_df.empty:
        same = bao_cases_df[(bao_cases_df["split_type"] == "LOO") & (bao_cases_df["query_id"].isin(["q19a", "q11b"]))]
        if len(same) == 2:
            parts = []
            for _, row in same.sort_values("query_id").iterrows():
                parts.append(f"{row['query_id']}: điểm mức độ mới {float(row['novelty']):.3f}, Bao/PG {float(row['bao_vs_pg_ratio']):.2f}×")
            evidence.append({"Bằng chứng": "Cùng mức độ mới, khác hiệu năng", "Kết quả quan sát": " · ".join(parts)})

    if balsa_cases_df is not None and not balsa_cases_df.empty:
        rows = balsa_cases_df[balsa_cases_df["case_type"].isin(["LOW_NOVELTY_WORST_SLOWDOWN", "BALSA_TIMEOUT"])]
        if not rows.empty:
            row = rows.iloc[0]
            ratio = row.get("balsa_vs_pg_ratio")
            ratio_text = f"{float(ratio):.2f}×" if pd.notna(ratio) else str(row.get("balsa_status", "N/A"))
            evidence.append({
                "Bằng chứng": "Trường hợp ngoại lệ ở mức độ mới thấp/trung bình",
                "Kết quả quan sát": f"{row['query_id']}: điểm mức độ mới {float(row['novelty']):.3f}, kết quả Balsa = {ratio_text}",
            })

    return pd.DataFrame(evidence)


def render_structural_novelty_extension() -> None:
    st.markdown("## 5. Kết quả tổng hợp")

    st.markdown("### 5.1. Độ mới theo cách chia dữ liệu (RQ1)")
    render_novelty_distribution()

    st.markdown("### 5.2. Hiệu năng theo mức độ mới")
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
                "Bao: mức thấp": st.column_config.NumberColumn(format="%.3f×"),
                "Bao: mức cao": st.column_config.NumberColumn(format="%.3f×"),
                "Bao: cao/thấp": st.column_config.NumberColumn(format="%.2f×"),
                "Balsa: mức thấp": st.column_config.NumberColumn(format="%.3f×"),
                "Balsa: mức cao": st.column_config.NumberColumn(format="%.3f×"),
                "Balsa: cao/thấp": st.column_config.NumberColumn(format="%.2f×"),
            },
        )

    st.markdown("### 5.3. Tương quan với hiệu năng (RQ2)")
    spearman_table = build_spearman_comparison_table()
    st.dataframe(
        spearman_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Bao ρ": st.column_config.NumberColumn(format="%.3f"),
            "Bao (p-value)": st.column_config.NumberColumn(format="%.3f"),
            "Balsa ρ": st.column_config.NumberColumn(format="%.3f"),
            "Balsa (p-value)": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    with st.expander("Xem biểu đồ phân tán theo từng cách chia dữ liệu", expanded=False):
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

    st.markdown("### 5.4. Tóm tắt")
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

    # Không fallback sang run khác: case study phải truy xuất đúng artifact
    # của run_id đang được hiển thị.
    return None, str(preferred)


def extract_hint_block(sql_text: str | None) -> str | None:
    if not sql_text:
        return None
    match = re.search(r"/\*\+(.*?)\*/", sql_text, flags=re.DOTALL)
    if not match:
        return None
    return "\n".join(line.strip() for line in match.group(1).strip().splitlines() if line.strip())


def render_query_case_study() -> None:
    """Luồng minh họa một truy vấn trong phần nghiên cứu mở rộng.

    Các đặc trưng SQL được đọc từ CSV đã xử lý; chỉ khoảng cách
    đến tập huấn luyện được tính lại khi người dùng chọn truy vấn.
    """
    st.markdown("## Phân tích truy vấn")
    st.caption(
        "Chọn truy vấn kiểm tra để xem đặc trưng cấu trúc, cách tính điểm mức độ mới "
        "và kết quả Bao/Balsa đã ghi nhận."
    )

    st.markdown("### 1. Chọn dữ liệu")
    control1, control2 = st.columns(2)
    with control1:
        selected_split_type = st.selectbox(
            "Cách chia dữ liệu",
            SPLIT_ORDER,
            format_func=lambda x: SPLIT_LABEL.get(x, x),
            key="case_split",
        )

    selected_split_id = 1
    train_queries, test_queries = get_split_queries(selected_split_type, selected_split_id)
    if not test_queries:
        st.warning("Không tìm thấy truy vấn kiểm tra trong cách chia đã chọn.")
        return
    defaults = {"LOO": "q11b", "Random": "q6c", "BaseQuery": "q15b"}
    preferred = defaults.get(selected_split_type)
    default_idx = test_queries.index(preferred) if preferred in test_queries else 0

    with control2:
        # Khóa riêng theo cách chia tránh lưu lựa chọn không còn thuộc tập kiểm tra.
        selected_query = st.selectbox(
            "Truy vấn kiểm tra",
            test_queries,
            index=default_idx,
            key=f"case_query_{selected_split_type}",
        )
    st.caption(
        f"{SPLIT_LABEL[selected_split_type]}: {len(train_queries)} truy vấn huấn luyện, "
        f"{len(test_queries)} truy vấn kiểm tra."
    )

    result = calculate_novelty(selected_query, selected_split_type, selected_split_id)
    q33, q67 = novelty_thresholds(selected_split_type, selected_split_id)
    level = novelty_level(result["novelty"], q33, q67)
    query_feature = feature_map[selected_query]
    query_sql, query_sql_path = load_query_sql(selected_query)
    bao_row = selected_performance_row(bao_df, selected_split_type, selected_split_id, selected_query)
    balsa_row = selected_performance_row(balsa_df, selected_split_type, selected_split_id, selected_query)

    st.markdown("### 2. Đặc trưng truy vấn")
    st.caption(
        "Ba tập đặc trưng được trích xuất trước và lưu trong query_features.csv. "
        "Việc chọn truy vấn chỉ hiển thị kết quả đã trích xuất, không phân tích lại SQL."
    )
    sql_col, feature_col = st.columns([1.15, 1])
    with sql_col:
        st.markdown(f"#### SQL · {selected_query}")
        st.code(query_sql, language="sql")
        st.caption(query_sql_path)
    with feature_col:
        n1, n2, n3 = st.columns(3)
        n1.metric("Bảng dữ liệu", query_feature["n_tables"])
        n2.metric("Điều kiện kết hợp", query_feature["n_joins"])
        n3.metric("Điều kiện lọc", query_feature["n_filters"])
        tables_tab, joins_tab, filters_tab = st.tabs(
            ["Bảng dữ liệu", "Điều kiện kết hợp", "Điều kiện lọc"]
        )
        with tables_tab:
            st.markdown(_feature_list_markdown(query_feature["tables"]))
        with joins_tab:
            st.markdown(_feature_list_markdown(query_feature["joins"]))
        with filters_tab:
            st.markdown(_feature_list_markdown(query_feature["filters"]))
    st.caption(
        "Nhóm sử dụng các đặc trưng này để phân tích sau thực nghiệm; "
        "chúng không phải bộ đặc trưng đầu vào chung của Bao và Balsa."
    )

    st.markdown("### 3. Tính điểm mức độ mới")
    st.caption(
        "So sánh truy vấn đã chọn với từng truy vấn huấn luyện bằng khoảng cách Jaccard "
        "trên ba tập đặc trưng. Khoảng cách trung bình nhỏ nhất là điểm mức độ mới."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Điểm mức độ mới", f"{result['novelty']:.6f}")
    m2.metric("Mức độ mới", level)
    m3.metric("Truy vấn huấn luyện gần nhất", result["nearest"])
    d1, d2, d3 = st.columns(3)
    d1.metric("Khoảng cách bảng", f"{result['d_table']:.6f}")
    d2.metric("Khoảng cách kết hợp", f"{result['d_join']:.6f}")
    d3.metric("Khoảng cách lọc", f"{result['d_filter']:.6f}")
    with st.expander("Xem 5 truy vấn huấn luyện gần nhất", expanded=False):
        st.dataframe(
            result["candidate_df"].head(5),
            hide_index=True,
            width="stretch",
            column_config={
                "d_table": st.column_config.NumberColumn("Khoảng cách bảng", format="%.6f"),
                "d_join": st.column_config.NumberColumn("Khoảng cách kết hợp", format="%.6f"),
                "d_filter": st.column_config.NumberColumn("Khoảng cách lọc", format="%.6f"),
                "D(q,h)": st.column_config.NumberColumn("Khoảng cách trung bình", format="%.6f"),
            },
        )

    st.markdown("### 4. Đối chiếu mức độ mới và hiệu năng")
    st.caption("Điểm mức độ mới được tính từ cấu trúc SQL; tỷ số E2E lấy từ kết quả thực nghiệm đã lưu.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Điểm mức độ mới", f"{result['novelty']:.6f}")
    m2.metric("Bao / PostgreSQL (E2E)", format_ratio(bao_row.get("bao_vs_pg_ratio")) if bao_row is not None else "N/A")
    m3.metric("Balsa / PostgreSQL (E2E)", format_ratio(balsa_row.get("balsa_vs_pg_ratio")) if balsa_row is not None else "N/A")
    st.caption(
        "Mỗi tỷ số dùng kết quả PostgreSQL đối chứng của chính thực nghiệm Bao hoặc Balsa. "
        "Giá trị nhỏ hơn 1× nghĩa là LQO nhanh hơn PostgreSQL; lớn hơn 1× nghĩa là chậm hơn. "
        "Điểm mức độ mới không được dùng để lựa chọn kế hoạch thực thi."
    )



def render_model_provenance(split_type: str) -> None:
    """Nguồn huấn luyện và đánh giá theo cách chia đang chọn; không chạy mô hình."""
    split_label = SPLIT_LABEL[split_type]
    train_queries, test_queries = get_split_queries(split_type, 1)

    bao_manifest_path = BAO_RUN_DIRS[split_type] / "run_manifest.json"
    balsa_manifest_path = BALSA_RUN_DIRS[split_type] / "run_manifest.json"
    bao_manifest = _load_json_if_exists(bao_manifest_path)
    balsa_manifest = _load_json_if_exists(balsa_manifest_path)

    st.caption(
        f"**Nguồn dữ liệu:** `analysis/data/split_master.csv` · "
        f"{split_label}: **{len(train_queries)} truy vấn huấn luyện**, "
        f"**{len(test_queries)} truy vấn kiểm tra**."
    )

    with st.expander("Nguồn mô hình và mã đánh giá", expanded=False):
        bao_col, balsa_col = st.columns(2)
        with bao_col:
            st.markdown("**Bao**")
            st.write("**Huấn luyện:** dùng mô hình đã huấn luyện do tác giả cung cấp; nhóm không huấn luyện lại Bao.")
            checkpoint_path = bao_manifest.get("checkpoint_path")
            if checkpoint_path:
                st.caption(f"Mô hình: `{checkpoint_path}`")
            st.caption(f"Nhật ký thực nghiệm: `{bao_manifest_path.relative_to(REPO_ROOT)}`")
            st.caption(
                f"Kết quả đánh giá: `{(BAO_RUN_DIRS[split_type] / 'evaluation_results.csv').relative_to(REPO_ROOT)}`"
            )
            st.caption("Mã chạy đánh giá: `bao/scripts/test_ckpt.sh`")
            st.caption("Mã tổng hợp kết quả lựa chọn: `bao/scripts/build_bao_outputs.py`")

        with balsa_col:
            st.markdown("**Balsa**")
            st.write("**Huấn luyện:** nhóm huấn luyện theo cách chia đang chọn (cấu hình 3 vòng lặp).")
            st.caption("Mã huấn luyện: `balsa/source/scripts/train_split.py`")
            checkpoint_path = balsa_manifest.get("checkpoint_path")
            if checkpoint_path:
                st.caption(f"Mốc mô hình theo nhật ký: `{checkpoint_path}`")
            st.caption(f"Nhật ký thực nghiệm: `{balsa_manifest_path.relative_to(REPO_ROOT)}`")
            st.caption(
                f"Kết quả đánh giá: `{(BALSA_RUN_DIRS[split_type] / 'evaluation_results.csv').relative_to(REPO_ROOT)}`"
            )
            st.caption("Mã đánh giá và ghi nhận kế hoạch: `balsa/source/scripts/evaluate_checkpoint.py`")

        st.caption(
            "Các tệp đánh giá ghi nhận phương án được mô hình chọn và thời gian xử lý. "
            "Giao diện chỉ đọc kết quả đã lưu; không thể suy ra đầy đủ lý do nội bộ "
            "mô hình chọn phương án đó chỉ từ mã phương án."
        )


def render_paper_query_decisions() -> None:
    """Xem phương án thực tế Bao/Balsa đã chọn, không chạy lại mô hình."""
    st.markdown("### Phương án thực thi được lựa chọn")
    split_col, query_col = st.columns(2)
    with split_col:
        split_type = st.selectbox(
            "Cách chia dữ liệu", SPLIT_ORDER,
            format_func=lambda x: SPLIT_LABEL[x], key="paper_decision_split",
        )
    _, test_queries = get_split_queries(split_type, 1)
    if not test_queries:
        st.warning("Không có truy vấn kiểm tra cho cách chia này.")
        return
    defaults = {"LOO": "q11b", "Random": "q6c", "BaseQuery": "q15b"}
    default_query = defaults.get(split_type)
    idx = test_queries.index(default_query) if default_query in test_queries else 0
    with query_col:
        query_id = st.selectbox(
            "Truy vấn kiểm tra", test_queries, index=idx,
            key=f"paper_decision_query_{split_type}",
        )

    bao_row = selected_performance_row(bao_df, split_type, 1, query_id)
    balsa_row = selected_performance_row(balsa_df, split_type, 1, query_id)
    with st.expander("Xem truy vấn SQL", expanded=False):
        sql, source_path = load_query_sql(query_id)
        st.code(sql, language="sql")
        st.caption(source_path)

    render_model_provenance(split_type)

    bao_col, balsa_col = st.columns(2)
    with bao_col:
        with st.container(border=True):
            st.markdown("#### Bao · Bộ gợi ý thực thi")
            if bao_row is None:
                st.warning("Không có kết quả Bao cho truy vấn này.")
            else:
                chosen = str(bao_row.get("bao_chosen_plan_id", "N/A"))
                chosen_ids = str(bao_row.get("bao_chosen_plan_ids", ""))
                st.metric("Bộ gợi ý được chọn", chosen)
                st.caption(BAO_ARM_HINTS.get(chosen, "Không có mô tả cho bộ gợi ý này."))
                if chosen_ids and chosen_ids.lower() != "nan":
                    st.caption(f"Các lần chạy lặp: {chosen_ids}")
                a, b, c = st.columns(3)
                a.metric("Lập kế hoạch + suy luận", format_ms(bao_row.get("bao_planning_ms")))
                b.metric("Thực thi", format_ms(bao_row.get("bao_execution_ms")))
                c.metric("E2E / PostgreSQL", format_ratio(bao_row.get("bao_vs_pg_ratio")))

    with balsa_col:
        with st.container(border=True):
            st.markdown("#### Balsa · Kế hoạch thực thi")
            if balsa_row is None:
                st.warning("Không có kết quả Balsa cho truy vấn này.")
            else:
                plan_id = str(balsa_row.get("balsa_chosen_plan_id", "N/A"))
                run_id = str(balsa_row.get("balsa_run_id", ""))
                st.metric("Mã kế hoạch được chọn", plan_id)
                a, b, c = st.columns(3)
                a.metric("Suy luận / tìm kế hoạch", format_ms(balsa_row.get("balsa_inference_ms")))
                b.metric("PostgreSQL lập kế hoạch", format_ms(balsa_row.get("balsa_planning_ms")))
                c.metric("E2E / PostgreSQL", format_ratio(balsa_row.get("balsa_vs_pg_ratio")))
                hinted_sql, hinted_path = load_balsa_hinted_sql(split_type, run_id, query_id)
                hint_block = extract_hint_block(hinted_sql)
                if hint_block:
                    with st.expander("Xem gợi ý thực thi Balsa chọn", expanded=False):
                        st.code(hint_block, language="text")
                        st.caption(hinted_path)
                elif hinted_sql:
                    with st.expander("Xem SQL có gợi ý thực thi", expanded=False):
                        st.code(hinted_sql, language="sql")
                        st.caption(hinted_path)
                else:
                    st.caption("Không tìm thấy tệp gợi ý thực thi trong dữ liệu đã lưu.")
    st.caption("Đây là kết quả thực nghiệm đã lưu; thay đổi lựa chọn không chạy lại Bao, Balsa hoặc PostgreSQL.")


# ============================================================
# MAIN
# ============================================================

st.title("Đánh giá tối ưu hóa truy vấn bằng học máy")
st.caption("Bao · Balsa · PostgreSQL trên JOB")

st.info(
    "**Chế độ xem lại kết quả:** ứng dụng đọc dữ liệu và tệp kết quả thực nghiệm đã lưu trong kho mã nguồn; "
    "thao tác chọn truy vấn không huấn luyện mô hình hoặc chạy lại phép đánh giá."
)

st.markdown(
    "**Hai phần của ứng dụng:** tái lập quy trình và đánh giá trong bài báo gốc; "
    "sau đó phân tích phần nghiên cứu mở rộng do nhóm thực hiện."
)

tab_paper, tab_extension = st.tabs([
    "Tab 1 · Tái lập thực nghiệm bài báo gốc",
    "Tab 2 · Nghiên cứu mở rộng",
])

with tab_paper:
    render_experiment_scenario_and_data()
    render_paper_encoding_and_pipeline()
    render_paper_query_decisions()
    st.divider()
    render_paper_protocol_replay()

with tab_extension:
    render_query_case_study()
    st.divider()
    render_structural_novelty_extension()

st.caption(
    "Nhóm 14 · JOB · Tab 1: tái lập thực nghiệm bài báo · "
    "Tab 2: nghiên cứu mở rộng về mức độ mới cấu trúc"
)
