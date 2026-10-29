
#!/usr/bin/env python3
"""Tạo experiment log (format nhóm) cho 1 lượt Bao đã train/test xong. Chạy trong container bao_server."""
import argparse, csv, glob, os, sqlite3, statistics, sys
from collections import OrderedDict
from datetime import datetime

BAO_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(BAO_DIR, "bao_server")
PENALTY_MS = 2 * 3 * 60 * 1000
TS_FMT = "%Y-%m-%d %H:%M:%S.%f"
SPLIT_LABEL = {"base_query": "BaseQuery", "leave_one_out": "LOO", "random": "Random"}


def split_label(split):
    for key, label in SPLIT_LABEL.items():
        if split.startswith(key + "_split_"):
            return f"{label}-{split.rsplit('_', 1)[-1]}"
    return split


def qid(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem if stem.startswith("q") else f"q{stem}"


def ms(v):
    return "N/A" if v is None else f"{v:.1f} ms"


def num(v, decimals=1):
    if v is None:
        return "N/A"
    return f"{int(round(v))}" if decimals == 0 else f"{v:.{decimals}f}"


def yn(flag):
    return "YES" if flag else "NO"


def read_log(path):
    rows = []
    if not path or not os.path.exists(path):
        print(f"[!] Không thấy file: {path}")
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = [x.strip() for x in line.rstrip("\n").split(", ")]
            if len(p) < 7:
                continue
            try:
                rows.append({"label": p[-1], "execution": float(p[-2]), "planning": float(p[-3]),
                             "file": p[-4], "ts": datetime.strptime(p[-5], TS_FMT),
                             "iter": int(p[-6]), "head": p[:-6]})
            except ValueError:
                continue
    return rows


def summarize(runs):
    if not runs:
        return None
    if any(r["execution"] >= PENALTY_MS or r["planning"] >= PENALTY_MS for r in runs):
        return {"status": "TIMEOUT", "planning": None, "execution": None, "e2e": None, "hint": None}
    pl = statistics.median(r["planning"] for r in runs)
    ex = statistics.median(r["execution"] for r in runs)
    hint = ", ".join(runs[-1]["head"]) if runs[-1]["head"] else None
    return {"status": "COMPLETED", "planning": pl, "execution": ex, "e2e": pl + ex,
            "hint": None if hint == "x" else hint}


def group(rows):
    g = OrderedDict()
    for r in rows:
        g.setdefault(os.path.basename(r["file"]), []).append(r)
    return g


class PlanInfo:
    def __init__(self, enabled, host, db):
        self.cur, self.TF, self.cache = None, None, {}
        if not enabled:
            return
        try:
            import psycopg2
            conn = psycopg2.connect(f"dbname={db} user=postgres password=postgres host={host}")
            conn.autocommit = True
            self.cur = conn.cursor()
            self.cur.execute("SET enable_bao TO off")
            self.cur.execute("SET statement_timeout TO 60000")
        except Exception as e:
            print(f"[!] Không kết nối được PostgreSQL ({e}) -> Cost/Cardinality = N/A")
        try:
            sys.path.insert(0, SERVER_DIR)
            from featurize import TreeFeaturizer
            self.TF = TreeFeaturizer
        except Exception as e:
            print(f"[!] Không import được encoder của Bao ({e})")

    def get(self, sql_path):
        if sql_path in self.cache:
            return self.cache[sql_path]
        info = {"cost": None, "rows": None, "encoded": None}
        if self.cur is not None and os.path.exists(sql_path):
            try:
                with open(sql_path) as f:
                    self.cur.execute(f"EXPLAIN (FORMAT JSON) {f.read()}")
                plan = self.cur.fetchall()[0][0][-1]["Plan"]
                info["cost"], info["rows"] = plan.get("Total Cost"), plan.get("Plan Rows")
                if self.TF is not None:
                    try:
                        tf = self.TF()
                        tf.fit([{"Plan": plan}])
                        tf.transform([{"Plan": plan}])
                        info["encoded"] = True
                    except Exception:
                        info["encoded"] = False
            except Exception as e:
                print(f"[!] EXPLAIN lỗi {sql_path}: {str(e).splitlines()[0]}")
        self.cache[sql_path] = info
        return info


def experience_count(db_path):
    if not db_path or not os.path.exists(db_path):
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            return conn.execute("SELECT count(*) FROM experience").fetchone()[0]
    except sqlite3.Error:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir")
    ap.add_argument("--split")
    ap.add_argument("--train_log")
    ap.add_argument("--pg_test_log")
    ap.add_argument("--bao_test_log")
    ap.add_argument("--model_path")
    ap.add_argument("--bao_db")
    ap.add_argument("--out")
    ap.add_argument("--all_samples", action="store_true")
    ap.add_argument("--no_db", action="store_true")
    ap.add_argument("--pg_host", default="pg_bao")
    ap.add_argument("--database_name", default="imdbload")
    a = ap.parse_args()

    os.chdir(BAO_DIR)
    if a.run_dir:
        rd = a.run_dir.rstrip("/")
        a.split = a.split or open(os.path.join(rd, "split-name.txt")).read().strip()
        a.train_log = a.train_log or os.path.join(rd, "logs/train.txt")
        a.pg_test_log = a.pg_test_log or os.path.join(rd, "logs/test-postgres.txt")
        a.bao_test_log = a.bao_test_log or os.path.join(rd, "logs/test-bao.txt")
        a.model_path = a.model_path or os.path.join(rd, "bao_server/bao_default_model")
        a.bao_db = a.bao_db or os.path.join(rd, "bao_server/bao.db")
        a.out = a.out or os.path.join(rd, "logs/experiment.log")
    if not a.split or not a.out:
        ap.error("cần --run_dir, hoặc --split + --out + các file log")

    label = split_label(a.split)
    qdir = os.path.join("queries", f"job__{a.split}")
    train_files = {os.path.basename(p): p for p in glob.glob(os.path.join(qdir, "train", "*.sql"))}
    test_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(qdir, "test", "*.sql")))
    info = PlanInfo(not a.no_db, a.pg_host, a.database_name)

    train = read_log(a.train_log)
    last_iter = max((r["iter"] for r in train), default=2)
    reward_runs = [r for r in train if r["iter"] == last_iter]
    samples = []
    for r in reward_runs:
        name = os.path.basename(r["file"])
        ok = r["execution"] < PENALTY_MS
        pi = info.get(train_files.get(name, r["file"]))
        samples.append({"q": qid(name), "ok": ok, "cost": pi["cost"], "rows": pi["rows"],
                        "exec": r["execution"] if ok else None,
                        "encoded": ok and (pi["encoded"] if pi["encoded"] is not None else True),
                        "model_input": ok})
    if a.all_samples:
        blocks = samples
    else:
        uniq = OrderedDict()
        for s in samples:
            uniq.pop(s["q"], None)
            uniq[s["q"]] = s
        blocks = list(uniq.values())

    train_time = ((max(r["ts"] for r in train) - min(r["ts"] for r in train)).total_seconds()
                  if train else None)
    model_ok = bool(a.model_path) and os.path.isdir(a.model_path)
    status = "COMPLETED" if (train and model_ok) else ("FAILED" if train else "SKIPPED (no training log)")

    bao, pg = group(read_log(a.bao_test_log)), group(read_log(a.pg_test_log))
    order = [f for f in test_files if f in bao] or list(bao)

    L = ["===== EXPERIMENT START =====", f"Split: {label}", "Model: Bao"]
    for s in blocks:
        L += ["[1] TRAINING DATA GENERATION", f"Train query: {s['q']}", f"Plan collected: {yn(s['ok'])}",
              f"Cost: {num(s['cost'])}", f"Cardinality: {num(s['rows'], 0)}",
              f"Execution time: {ms(s['exec']) if s['ok'] else 'TIMEOUT'}",
              "[2] QUERY & PLAN ENCODING", f"Query: {s['q']}",
              f"Query/Plan encoded: {yn(s['encoded'])}", f"Model input generated: {yn(s['model_input'])}"]
    tt = "N/A" if train_time is None else f"{train_time:.1f} s ({train_time / 60:.1f} min)"
    SRC = " (author checkpoint, paper hardware)" if a.model_path and os.path.basename(a.model_path.rstrip("/")).startswith("bao__job__") else ""
    L += ["[3] LQO TRAINING", "Model: Bao", f"Split: {label}", f"Training status: {status}{SRC}",
          f"Training time: {tt}{SRC}", f"Model/state: {a.model_path if model_ok else 'N/A'}"]
    rows = []
    r3 = lambda v: None if v is None else round(v, 3)
    for f in order:
        b, p = summarize(bao.get(f)), summarize(pg.get(f))
        L += ["[4] LQO EVALUATION", f"Test query: {qid(f)}", f"Bao inference/planning: {ms(b['planning'])}",
              f"Execution time: {ms(b['execution'])}", f"End-to-End: {ms(b['e2e'])}", f"Status: {b['status']}"]
        if p:
            L += ["[POSTGRESQL]", f"Planning: {ms(p['planning'])}", f"Execution: {ms(p['execution'])}",
                  f"End-to-End: {ms(p['e2e'])}"]
        rows.append([label, qid(f), b["hint"], r3(b["planning"]), r3(b["execution"]), r3(b["e2e"]), b["status"],
                     r3((p or {}).get("planning")), r3((p or {}).get("execution")), r3((p or {}).get("e2e")),
                     (p or {}).get("status"),
                     r3(p["e2e"] / b["e2e"]) if (p and p["e2e"] and b["e2e"]) else None])
    L += ["===== EXPERIMENT END ====="]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        out = []
        for x in L:
            if x.startswith("[") or x.startswith("===== EXPERIMENT END"):
                out.append("")
            out.append(x)
            if x.startswith("===== EXPERIMENT START"):
                out.append("")
        fh.write("\n".join(out) + "\n")
    csv_path = os.path.splitext(a.out)[0] + "_per_query.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "query_id", "bao_hint", "bao_planning_ms", "bao_execution_ms", "bao_e2e_ms",
                    "bao_status", "pg_planning_ms", "pg_execution_ms", "pg_e2e_ms", "pg_status", "speedup_e2e"])
        w.writerows(rows)

    n_ok = sum(s["model_input"] for s in samples)
    exp = experience_count(a.bao_db)
    print(f"[{label}] train: {len(reward_runs)} lần chạy có reward ({n_ok} thành công), "
          f"{len(blocks)} khối trong log; test: {len(order)}/{len(test_files)} query")
    if exp is not None:
        print(f"[{label}] bao.db experience = {exp} vs mẫu thành công = {n_ok} "
              f"[{'OK' if exp == n_ok else 'KHÁC -> kiểm tra lại'}]")
    miss = [f for f in order if f not in pg]
    if miss:
        print(f"[!] Thiếu kết quả PostgreSQL cho: {miss}")
    print(f"Log : {a.out}\nCSV : {csv_path}")


if __name__ == "__main__":
    main()
