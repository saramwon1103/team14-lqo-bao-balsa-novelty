#!/usr/bin/env bash
set -uo pipefail
if [[ ${1:-} == --help ]]; then
  echo 'Usage: run_all_splits.sh OUTPUT_ROOT (--smoke|--simulator-smoke|--allow-full) MANIFEST...'
  exit 0
fi
if [[ $# -lt 3 ]]; then
  echo 'Missing arguments; use --help' >&2
  exit 2
fi
root=$1
mode=$2
shift 2
mkdir -p "$root"
status=0
for manifest in "$@"; do
  id="$(basename "$manifest" .json)-$(date +%s%N)"
  python "$(dirname "$0")/run_experiment.py" --manifest "$manifest" --output-dir "$root/$id" "$mode" > "$root/$id.raw.log" 2>&1
  code=$?
  printf '%s\t%s\t%s\n' "$manifest" "$id" "$code" >> "$root/exit_codes.tsv"
  if [[ $code -ne 0 ]]; then status=1; fi
done
exit "$status"
