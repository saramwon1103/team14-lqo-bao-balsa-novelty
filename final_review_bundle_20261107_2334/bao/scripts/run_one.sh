#!/bin/bash
# Chạy trọn 1 split Bao: model sạch -> train -> test PG -> test Bao -> lưu model -> xuất log
set -u
TAG="$1"; SPLIT="$2"
cd /app/bao && source env/bin/activate && mkdir -p logs
say() { echo; echo "[$(date '+%H:%M:%S')] ===== $* ====="; }

# Chặn chạy trên database rỗng
N=$(python3 -c "import psycopg2;c=psycopg2.connect('dbname=imdbload user=postgres password=postgres host=pg_bao').cursor();c.execute('SELECT count(*) FROM cast_info');print(c.fetchone()[0])" 2>/dev/null || echo 0)
[ "$N" -gt 30000000 ] || { echo "DUNG: cast_info chi co $N dong (can ~36 trieu). Nap du lieu truoc."; exit 1; }
echo "Du lieu OK: cast_info = $N dong"

for f in logs/train__bao__$TAG.txt logs/test__postgres__$TAG.txt logs/test__bao__$TAG.txt; do
  [ -e "$f" ] && { echo "DUNG: $f da ton tai (dung TAG khac, vd ${TAG%run*}run2)"; exit 1; }
done
[ -d "queries/job__$SPLIT/train" ] || { echo "Sai SPLIT: $SPLIT"; exit 1; }

say "1/7 Tat Bao server cu + cat model cu"
pkill -f "main.py" ; sleep 3
ARCH="bao_server/archive_before_${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCH"
for f in bao_default_model bao_previous_model bao_tmp_model bao.db; do
  [ -e "bao_server/$f" ] && mv "bao_server/$f" "$ARCH/"
done

say "2/7 Bat Bao server sach"
( cd bao_server && nohup python3 -u main.py > ../logs/server__$TAG.txt 2>&1 & )
for i in $(seq 60); do (exec 3<>/dev/tcp/bao_server/9381) 2>/dev/null && break; sleep 1; done
(exec 3<>/dev/tcp/bao_server/9381) 2>/dev/null || { echo "Server khong len:"; cat logs/server__$TAG.txt; exit 1; }
grep -q "Loading existing model" logs/server__$TAG.txt && { echo "DUNG: server nap model cu"; exit 1; }
echo "Server OK (trang thai sach)"

say "3/7 TRAIN $SPLIT (1-2 gio)"
python3 -u run_queries.py --database_name imdbload \
  --query_dir queries/job__$SPLIT/train --output_file logs/train__bao__$TAG.txt
[ -d bao_server/bao_default_model ] || { echo "LOI: khong co checkpoint sau khi train"; exit 1; }
echo "train.txt: $(wc -l < logs/train__bao__$TAG.txt) dong (mong doi 1500)"

say "4/7 TEST PostgreSQL"
python3 -u run_test_queries.py --use_postgres --database_name imdbload \
  --query_dir queries/job__$SPLIT/test --output_file logs/test__postgres__$TAG.txt

say "5/7 TEST Bao"
python3 -u run_test_queries.py --use_bao --database_name imdbload \
  --query_dir queries/job__$SPLIT/test --output_file logs/test__bao__$TAG.txt

say "6/7 Luu model + kiem tra"
pkill -f "main.py" ; sleep 3
cp -r bao_server/bao_default_model "bao_server/model__$TAG"
cp bao_server/bao.db "bao_server/bao__$TAG.db"
n=$(ls queries/job__$SPLIT/test/*.sql | wc -l)
echo "Mong doi $((n*3)) dong moi file test:"
wc -l logs/test__postgres__$TAG.txt logs/test__bao__$TAG.txt
echo "So dong bi phat 360000 (timeout/loi):"
grep -c "360000" logs/test__postgres__$TAG.txt logs/test__bao__$TAG.txt logs/train__bao__$TAG.txt

say "7/7 Xuat experiment log"
python3 make_bao_log.py --split "$SPLIT" \
  --train_log logs/train__bao__$TAG.txt \
  --pg_test_log logs/test__postgres__$TAG.txt \
  --bao_test_log logs/test__bao__$TAG.txt \
  --model_path "bao_server/model__$TAG" \
  --bao_db "bao_server/bao__$TAG.db" \
  --out logs/experiment__$TAG.log

say "XONG $TAG -> logs/experiment__$TAG.log"
