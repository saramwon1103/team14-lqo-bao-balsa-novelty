#!/bin/bash
# Test Bao bằng checkpoint có sẵn + PostgreSQL, rồi xuất experiment log
set -u
TAG="$1"; SPLIT="$2"
MODEL="bao_server/bao__job__$SPLIT"
cd /app/bao && source env/bin/activate && mkdir -p logs
say() { echo; echo "[$(date '+%H:%M:%S')] ===== $* ====="; }
[ -d "$MODEL" ] || { echo "Khong co checkpoint $MODEL"; exit 1; }
for f in logs/test__postgres__$TAG.txt logs/test__bao__$TAG.txt; do
  [ -e "$f" ] && { echo "DUNG: $f da ton tai"; exit 1; }
done

say "1/4 Bat Bao server voi checkpoint $MODEL"
pkill -f main.py; sleep 3
( cd bao_server && nohup python3 -u main.py "/app/bao/$MODEL" > ../logs/server__$TAG.txt 2>&1 & )
for i in $(seq 60); do (exec 3<>/dev/tcp/bao_server/9381) 2>/dev/null && break; sleep 1; done
sleep 5
grep -q "Accepted new model" logs/server__$TAG.txt || { echo "LOI: khong nap duoc model"; cat logs/server__$TAG.txt; exit 1; }
echo "Model da nap OK"

say "2/4 TEST PostgreSQL"
python3 -u run_test_queries.py --use_postgres --database_name imdbload \
  --query_dir queries/job__$SPLIT/test --output_file logs/test__postgres__$TAG.txt

say "3/4 TEST Bao"
python3 -u run_test_queries.py --use_bao --database_name imdbload \
  --query_dir queries/job__$SPLIT/test --output_file logs/test__bao__$TAG.txt

pkill -f main.py
n=$(ls queries/job__$SPLIT/test/*.sql | wc -l)
echo "Mong doi $((n*3)) dong:"; wc -l logs/test__postgres__$TAG.txt logs/test__bao__$TAG.txt
echo "Timeout:"; grep -c 360000 logs/test__postgres__$TAG.txt logs/test__bao__$TAG.txt

say "4/4 Xuat experiment log"
python3 make_bao_log.py --split "$SPLIT" \
  --train_log logs/train__bao__$SPLIT.txt \
  --pg_test_log logs/test__postgres__$TAG.txt \
  --bao_test_log logs/test__bao__$TAG.txt \
  --model_path "$MODEL" \
  --out logs/experiment__$TAG.log
say "XONG $TAG"
