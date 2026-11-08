#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
BALSA_DB="${BALSA_DB:-imdbload}"
WANDB_MODE="${WANDB_MODE:-offline}"
export BALSA_DB WANDB_MODE

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

python scripts/preflight_balsa.py --db "$BALSA_DB"
mkdir -p raw_logs
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"

run_config() {
  local config="$1"
  local log_path="raw_logs/${RUN_STAMP}_${config}.log"
  echo "Đang chạy ${config}. Log thô: ${log_path}"
  python -u run.py --run "$config" --local 2>&1 | tee "$log_path"
}

case "$MODE" in
  baseline)
    run_config Baseline_BaseQuery1
    ;;
  train)
    if [[ ! -f data/initial_policy_data_basequery1.pkl ]]; then
      echo "Thiếu baseline BaseQuery-1. Hãy chạy: $0 baseline" >&2
      exit 2
    fi
    run_config Balsa_BaseQuery1
    ;;
  full)
    if [[ ! -f data/initial_policy_data_basequery1.pkl ]]; then
      run_config Baseline_BaseQuery1
    fi
    run_config Balsa_BaseQuery1
    ;;
  smoke)
    if [[ ! -f data/initial_policy_data_basequery1_smoke.pkl ]]; then
      run_config Baseline_BaseQuery1_Smoke
    fi
    run_config Balsa_BaseQuery1_Smoke
    ;;
  paper)
    if [[ ! -f data/initial_policy_data.pkl ]]; then
      run_config Baseline
    fi
    run_config Balsa_JOBRandSplit
    ;;
  *)
    echo "Cách dùng: $0 {baseline|train|full|smoke|paper}" >&2
    exit 2
    ;;
esac

echo "Hoàn tất. Xem structured_logs/<wandb_run_id>/experiment.log"
