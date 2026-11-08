#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 SUPERVISOR_OUTPUT_ROOT /absolute/path/to/fuse2fs [SPLIT ...]" >&2
  exit 2
fi

root=$1
fuse2fs=$2
shift 2
if [[ $# -eq 0 ]]; then
  set -- LOO-1 Random-1 BaseQuery-1
fi
for split in "$@"; do
  case "$split" in
    LOO-1|Random-1|BaseQuery-1) ;;
    *) echo "Invalid V2 split: $split" >&2; exit 2 ;;
  esac
done
mkdir -p "$root"
timer="balsa-v2-aggregate-$$"
status=0

# Aggregate hard stop and RAM envelope are installed before any ML process.
systemctl --user set-property --runtime balsa-v2.slice \
  MemoryMax=24G MemoryHigh=23G MemorySwapMax=0 || exit 1
# Earlier guarded attempts consumed about 100 minutes of the approved six-hour
# allowance; cap this continuation conservatively at four hours.
systemd-run --user --unit="$timer" --on-active=14400s --timer-property=AccuracySec=1s \
  /usr/bin/systemctl --user stop balsa-v2.slice || exit 1
trap 'systemctl --user stop "$timer.timer" >/dev/null 2>&1 || true' EXIT

for split in "$@"; do
  run_root="$root/$split"
  if ! /home/wings/miniconda3/envs/balsa/bin/python \
      "$(dirname "$0")/pilot_supervisor.py" \
      --output-dir "$run_root" --fuse2fs "$fuse2fs" --split-type "$split" \
      --ml-memory-gib 24 \
      >"$root/$split.console.log" 2>&1; then
    status=1
    echo "Supervisor failed for $split; see $root/$split.console.log" >&2
  fi
  # Never start another split after the aggregate timer has stopped its slice.
  if [[ "$(systemctl --user is-active "$timer.timer" 2>/dev/null || true)" != active ]]; then
    echo 'Aggregate Balsa V2 deadline has expired; stopping sequence.' >&2
    exit 1
  fi
done
exit "$status"
