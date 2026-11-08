#!/usr/bin/env python3
"""
build_bao_outputs.py — Chuyển kết quả Bao đã chạy sang schema chung của nhóm (TODO_TUAN_BAO.md).

Không chạy lại thí nghiệm. Chỉ đọc các file đã có:
  bao/queries/job__<split>/{train,test}/*.sql
  bao/logs/test__{bao,postgres}__<tag>_ckpt.txt      (đo thật trên máy nhóm)
  bao/logs/train__bao__<split>.txt                   (log train của tác giả, đi kèm checkpoint)
  bao/logs/server__<tag>_ckpt.txt                    (log Bao server lúc test)
  ~/bao_ckpt.out                                     (console lúc test)
  bao/bao_server/bao__job__<split>/                  (checkpoint của tác giả)

Sinh ra cho mỗi split:  bao/runs/<Split>/<run_id>/
  run_manifest.json, split_snapshot.csv, training_trace.csv, checkpoint_manifest.csv,
  evaluation_results.csv, postgres_results.csv, raw_training.log
và bao/runs/ALL_evaluation_results.csv, bao/runs/ALL_postgres_results.csv, bao/runs/VALIDATION.txt

Chạy từ gốc repo NHÓM (không cần Docker / repo upstream):
  python3 bao/scripts/build_bao_outputs.py
Đầu vào: bao/logs/test__{bao,postgres}__*_ckpt.txt, bao/logs/author/train__bao__<split>.txt,
         bao/logs/console_evaluation.log, bao/checkpoints/, analysis/data/split_master.csv, novelty.csv
(Cách cũ, từ repo upstream:)
  python3 bao/build_bao_outputs.py --console ~/bao_ckpt.out \
      --git-commit "$(git rev-parse HEAD)" \
      --pg-version "$(docker exec -u postgres pg_bao psql -Atc 'SELECT version();')"
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import statistics
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
# Script nằm ở <team-repo>/bao/scripts/ -> BAO = <team-repo>/bao ; ROOT = <team-repo>
BAO = os.path.dirname(_HERE) if os.path.basename(_HERE) == "scripts" else _HERE
ROOT = os.path.dirname(BAO)
TIMEOUT_MS = 3 * 60 * 1000          # statement_timeout thực tế trong run_test_queries.py
PENALTY_MS = 2 * TIMEOUT_MS         # giá trị repo ghi khi timeout/lỗi (360000) -> KHÔNG phải số đo
TS_FMT = "%Y-%m-%d %H:%M:%S.%f"
MODEL = "Bao"
MODEL_SOURCE = "author_checkpoint"

# (split_type, split_id, thư mục split, tag lúc chạy, nhãn thư mục, run_id, số train, số test)
SPLITS = [
    ("LOO", 1, "leave_one_out_split_1", "loo1", "LOO-1", "bao_loo1_run01", 80, 33),
    ("Random", 1, "random_split_1", "rnd1", "Random-1", "bao_rnd1_run01", 91, 22),
    ("BaseQuery", 1, "base_query_split_1", "bq1", "BaseQuery-1", "bao_bq1_run01", 94, 19),
]

# 5 arm Bao dùng (bao_num_arms = 5), suy ra từ pg_extension/bao_planner.h: set_arm_options + arm_to_hint
ARM_HINTS = {
    "arm_0": frozenset(),
    "arm_1": frozenset({"SET enable_nestloop TO off"}),
    "arm_2": frozenset({"SET enable_mergejoin TO off", "SET enable_indexscan TO off"}),
    "arm_3": frozenset({"SET enable_nestloop TO off", "SET enable_mergejoin TO off",
                        "SET enable_indexscan TO off"}),
    "arm_4": frozenset({"SET enable_mergejoin TO off"}),
}


# ----------------------------------------------------------------------------- helpers
def qid(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem if stem.startswith("q") else f"q{stem}"


def family(q):
    m = re.match(r"^(q\d+)", q)
    return m.group(1) if m else q


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S") if dt else None


def r3(v):
    return "" if v is None else f"{v:.3f}"


def hint_to_arm(hint):
    if hint is None:
        return ""
    h = hint.strip()
    if h in ("", "x"):
        return ""
    if h == "(no hint)":
        return "arm_0"
    stmts = frozenset(s.strip() for s in h.split(";") if s.strip())
    for arm, s in ARM_HINTS.items():
        if s == stmts:
            return arm
    return "hint:" + h  # không khớp arm nào -> giữ nguyên chuỗi, không đoán


def read_runs(path):
    """Dòng log gốc: <head...>, iter, ts, file, planning, execution, label (đọc từ phải sang)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            p = line.split(", ")
            if len(p) < 7:
                raise ValueError(f"{path}:{n}: sai định dạng")
            rows.append({"head": ", ".join(p[:-6]), "iter": int(p[-6]),
                         "ts": datetime.strptime(p[-5].strip(), TS_FMT), "file": p[-4].strip(),
                         "planning": float(p[-3]), "execution": float(p[-2]), "label": p[-1].strip()})
    return rows


def summarize(runs):
    """Gộp 3 lần đo -> 1 dòng. Bất kỳ lần nào ghi giá trị phạt -> TIMEOUT, để trống thời gian."""
    base = {"plan": "", "plans": "", "warning": ""}
    if not runs:
        return {**base, "status": "ERROR", "pl": None, "ex": None, "e2e": None, "error": "missing"}
    if any(r["planning"] >= PENALTY_MS or r["execution"] >= PENALTY_MS for r in runs):
        return {**base, "status": "TIMEOUT", "pl": None, "ex": None, "e2e": None, "error": "timeout"}
    pl = statistics.median(r["planning"] for r in runs)
    ex = statistics.median(r["execution"] for r in runs)
    arms = [hint_to_arm(r["head"]) for r in runs]
    same = len(set(arms)) == 1
    return {"status": "OK", "pl": pl, "ex": ex, "e2e": pl + ex, "error": "",
            "plan": arms[0] if same else "MULTIPLE", "plans": "|".join(arms),
            "warning": "" if same else "hint_changed_between_runs"}


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def console_section(console, split_dir, tag):
    """Đoạn console (nguyên văn) của 1 split trong ~/bao_ckpt.out."""
    if not console or not os.path.exists(console):
        return None
    lines = open(console, encoding="utf-8", errors="replace").read().splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if f"bao__job__{split_dir}" in l and "1/4" in l), None)
    if start is None:
        return None
    end = next((i for i in range(start, len(lines)) if f"XONG {tag}_ckpt" in lines[i]), len(lines) - 1)
    return "".join(lines[start:end + 1])


# ----------------------------------------------------------------------------- per split
def build(split, args, report):
    stype, sid, sdir, tag, label, run_id, n_train_exp, n_test_exp = split
    out = os.path.join(args.out, label, run_id)
    os.makedirs(out, exist_ok=True)
    errs = []

    # ---- split_snapshot.csv
    # split_master.csv là nguồn duy nhất
    sm = [r for r in args.split_master_rows if r["split_type"] == stype and str(r["split_id"]) == str(sid)]
    tr = {r["sql_file"] for r in sm if r["role"] == "train"}
    te = {r["sql_file"] for r in sm if r["role"] == "test"}
    if tr & te:
        errs.append(f"Train ∩ Test != ∅: {sorted(tr & te)}")
    if len(sm) != 113 or len(tr | te) != 113:
        errs.append(f"split_master.csv: {len(sm)} dòng / {len(tr | te)} query (mong đợi 113)")
    if (len(tr), len(te)) != (n_train_exp, n_test_exp):
        errs.append(f"Số query train/test = {len(tr)}/{len(te)}, mong đợi {n_train_exp}/{n_test_exp}")
    # (tuỳ chọn) đối chiếu với thư mục queries/ của repo upstream
    if args.upstream_queries:
        up_tr = {os.path.basename(p) for p in glob.glob(os.path.join(args.upstream_queries, f"job__{sdir}", "train", "*.sql"))}
        up_te = {os.path.basename(p) for p in glob.glob(os.path.join(args.upstream_queries, f"job__{sdir}", "test", "*.sql"))}
        if (up_tr, up_te) != (tr, te):
            errs.append(f"split_master.csv KHÔNG khớp upstream job__{sdir}: train lệch {sorted(up_tr ^ tr)}, "
                        f"test lệch {sorted(up_te ^ te)}")
    snap = [[MODEL, stype, sid, r["query_id"], r["family"], r["role"], r["sql_file"]] for r in sm]
    snap.sort(key=lambda r: (int(re.match(r"q(\d+)", r[3]).group(1)), r[3]))
    write_csv(os.path.join(out, "split_snapshot.csv"),
              ["model", "split_type", "split_id", "query_id", "family", "role", "sql_file"], snap)

    # ---- evaluation_results.csv + postgres_results.csv (đo thật trên máy nhóm)
    bao_f = os.path.join(BAO, "logs", f"test__bao__{tag}_ckpt.txt")
    pg_f = os.path.join(BAO, "logs", f"test__postgres__{tag}_ckpt.txt")
    bao_runs, pg_runs = read_runs(bao_f), read_runs(pg_f)
    by = lambda rows: {os.path.basename(r["file"]): [x for x in rows if os.path.basename(x["file"]) == os.path.basename(r["file"])] for r in rows}
    bao_g, pg_g = by(bao_runs), by(pg_runs)
    ckpt_id = f"bao_{tag}_final"
    ev, pgr = [], []
    for f in sorted(te, key=lambda f: (int(re.match(r"(\d+)", f).group(1)), f)):
        q = qid(f)
        b, p = summarize(bao_g.get(f)), summarize(pg_g.get(f))
        if bao_g.get(f) and len(bao_g[f]) != 3:
            errs.append(f"Bao {q}: {len(bao_g[f])} lần đo (mong đợi 3)")
        if pg_g.get(f) and len(pg_g[f]) != 3:
            errs.append(f"PG {q}: {len(pg_g[f])} lần đo (mong đợi 3)")
        ev.append([run_id, MODEL, stype, sid, q, family(q), ckpt_id, r3(b["pl"]), "", r3(b["ex"]), r3(b["e2e"]),
                   b["status"], TIMEOUT_MS, b["plan"], b["error"], b["warning"], b["plans"]])
        pgr.append([stype, sid, q, family(q), r3(p["pl"]), r3(p["ex"]), r3(p["e2e"]), p["status"], TIMEOUT_MS])
    extra = (set(bao_g) | set(pg_g)) - te
    if extra:
        errs.append(f"Log test có query ngoài tập test: {sorted(extra)}")
    write_csv(os.path.join(out, "evaluation_results.csv"),
              ["run_id", "model", "split_type", "split_id", "query_id", "family", "checkpoint_id", "planning_ms",
               "inference_ms", "execution_ms", "end_to_end_ms", "status", "timeout_ms", "chosen_plan_id", "error", "warning",
               "chosen_plan_ids"], ev)
    write_csv(os.path.join(out, "postgres_results.csv"),
              ["split_type", "split_id", "query_id", "family", "planning_ms", "execution_ms", "end_to_end_ms",
               "status", "timeout_ms"], pgr)
    if len(ev) != n_test_exp:
        errs.append(f"evaluation rows = {len(ev)}, mong đợi {n_test_exp}")

    # ---- training_trace.csv (log train của tác giả, đi kèm checkpoint)
    tr_f = next((p for p in (os.path.join(BAO, "logs", "author", f"train__bao__{sdir}.txt"),
                             os.path.join(BAO, "logs", f"train__bao__{sdir}.txt")) if os.path.exists(p)),
                os.path.join(BAO, "logs", "author", f"train__bao__{sdir}.txt"))
    trace, t_start, t_end, n_retrain = [], None, None, 0
    if os.path.exists(tr_f):
        train_runs = read_runs(tr_f)
        last_iter = max(r["iter"] for r in train_runs)
        rewards = [r for r in train_runs if r["iter"] == last_iter]  # lần chạy có reward (được Bao học)
        t_start, t_end = min(r["ts"] for r in train_runs), max(r["ts"] for r in train_runs)
        total = len(rewards)
        prev_chunk = None
        for step, r in enumerate(rewards, 1):
            chunk = r["head"].split(",")[0].strip()
            if chunk != "x" and chunk != prev_chunk:
                n_retrain += 1
                trace.append([run_id, MODEL, stype, sid, "retrain", "", n_retrain, "", "", "", "", "OK",
                              f"author log: retrain before chunk {int(chunk) + 1} (baoctl.py --retrain; "
                              f"timestamp not logged by run_queries.py)", ""])
            prev_chunk = chunk
            ok = r["execution"] < PENALTY_MS
            phase = "data_collection_pg" if chunk == "x" else "data_collection_bao"
            trace.append([run_id, MODEL, stype, sid, phase, step, step, total,
                          "execution_time", f"{r['execution']:.3f}" if ok else "", "ms",
                          "OK" if ok else "TIMEOUT",
                          f"author log; query={qid(r['file'])}; planning_ms={r['planning']:.3f}" if ok
                          else f"author log; query={qid(r['file'])}; timeout",
                          iso(r["ts"])])
    else:
        errs.append(f"Không thấy log train của tác giả: {tr_f}")
    write_csv(os.path.join(out, "training_trace.csv"),
              ["run_id", "model", "split_type", "split_id", "phase", "step", "current", "total", "metric_name",
               "metric_value", "metric_unit", "status", "message", "timestamp"], trace)

    # ---- checkpoint_manifest.csv
    src_rel = f"bao/bao_server/bao__job__{sdir}"                       # vị trí trong repo upstream
    ckpt_rel = f"bao/checkpoints/bao__job__{sdir}"                       # artifact trong repo nhóm
    ckpt_abs = os.path.join(ROOT, ckpt_rel)
    need = ["nn_weights", "x_transform", "y_transform", "channels", "n"]
    ck_ok = os.path.isdir(ckpt_abs) and all(os.path.exists(os.path.join(ckpt_abs, x)) for x in need)
    if not ck_ok:
        errs.append(f"Checkpoint thiếu hoặc không đủ file: {ckpt_rel}")
    write_csv(os.path.join(out, "checkpoint_manifest.csv"),
              ["run_id", "model", "split_type", "split_id", "checkpoint_id", "step", "metric_name", "metric_value",
               "path", "is_final", "status"],
              [[run_id, MODEL, stype, sid, ckpt_id, n_retrain or "", "", "", ckpt_rel, "true",
                "OK" if ck_ok else "MISSING"]])

    # ---- raw_training.log (ghép nguyên văn, chỉ thêm dòng tiêu đề nguồn)
    raw = os.path.join(out, "raw_training.log")
    with open(raw, "w", encoding="utf-8") as fo:
        def add(title, path=None, text=None):
            fo.write(f"\n########## SOURCE: {title} ##########\n")
            if path and os.path.exists(path):
                with open(path, encoding="utf-8", errors="replace") as fi:
                    shutil.copyfileobj(fi, fo)
            elif text is not None:
                fo.write(text)
            else:
                fo.write("(không có file)\n")
        add(f"{os.path.relpath(tr_f, ROOT)}  [author training log, unmodified]", tr_f)
        add(f"{os.path.relpath(args.console, ROOT) if args.console.startswith(ROOT) else args.console}  [team evaluation console, unmodified section]",
            text=console_section(args.console, sdir, tag))
        add(f"bao/logs/server__{tag}_ckpt.txt  [Bao server, team evaluation]",
            os.path.join(BAO, "logs", f"server__{tag}_ckpt.txt"))
        add(f"bao/logs/test__postgres__{tag}_ckpt.txt  [team evaluation, raw]", pg_f)
        add(f"bao/logs/test__bao__{tag}_ckpt.txt  [team evaluation, raw]", bao_f)
        if stype == "BaseQuery":  # nỗ lực tự train của nhóm (dừng vì phần cứng)
            for name in ("team_training_attempt_bq1.txt", "train__bao__bq1_run1.txt", "server__bq1_run1.txt"):
                p = os.path.join(BAO, "logs", name)
                if os.path.exists(p):
                    add(f"bao/logs/{name}  [team training attempt, stopped: hardware too slow]", p)

    # ---- run_manifest.json
    all_eval = bao_runs + pg_runs
    mpath = os.path.join(out, "run_manifest.json")
    try:  # giữ lại git_commit / postgres_version đã ghi lần trước nếu không truyền tham số
        prev = json.load(open(mpath, encoding="utf-8"))
    except (OSError, ValueError):
        prev = {}
    status = "completed" if not errs else "completed_with_warnings"
    manifest = {
        "run_id": run_id, "model": MODEL, "split_type": stype, "split_id": sid,
        "model_source": MODEL_SOURCE,
        "train_query_count": len(tr), "test_query_count": len(te),
        "seed": 42,
        "git_repo": "edualc/lqo_ml_perspective", "git_commit": args.git_commit or prev.get("git_commit", "UNKNOWN"),
        "training_command": (f"NOT RUN BY TEAM (author_checkpoint). Author protocol: python3 run_queries.py "
                             f"--query_dir queries/job__{sdir}/train --output_file logs/train__bao__{sdir}.txt"),
        "evaluation_command": (f"bash test_ckpt.sh {tag}_ckpt {sdir}  ->  run_test_queries.py --use_postgres / "
                               f"--use_bao --query_dir queries/job__{sdir}/test"),
        "postgres_version": args.pg_version or prev.get("postgres_version", "UNKNOWN"),
        "start_time": iso(min(r["ts"] for r in all_eval)) if all_eval else None,
        "end_time": iso(max(r["ts"] for r in all_eval)) if all_eval else None,
        "time_zone": "UTC (container clock)",
        "author_training_start_time": iso(t_start), "author_training_end_time": iso(t_end),
        "author_training_time_s": round((t_end - t_start).total_seconds(), 1) if t_start else None,
        "checkpoint_path": ckpt_rel,
        "source_checkpoint_path": f"edualc/lqo_ml_perspective:{src_rel}",
        "artifact_checkpoint_path": ckpt_rel,
        "training_trace_source": "author_log",
        "split_source": "analysis/data/split_master.csv",
        "timeout_ms": TIMEOUT_MS,
        "runs_per_query": 3,
        "aggregation": "median planning_ms + median execution_ms over 3 runs; any timed-out run => TIMEOUT",
        "inference_in_planning": True,
        "chosen_plan_id_meaning": ("Bao arm (arm_0 = no hint ... arm_4) from 'Bao recommended hint'; "
                                   "MULTIPLE if the 3 runs chose different arms, see chosen_plan_ids"),
        "hardware": "Laptop 8 GB RAM, 8 CPU cores, WSL2 ~3.7 GB RAM, Docker Desktop",
        "notes": ("Training could not be reproduced on team hardware (~22 h/split estimated: 40/1500 "
                  "log lines in 35 min). Evaluation and PostgreSQL baseline measured by team."),
        "status": status,
        "warnings": errs,
    }
    with open(os.path.join(out, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    ok = sum(r[11] == "OK" for r in ev); to = sum(r[11] == "TIMEOUT" for r in ev)
    pok = sum(r[7] == "OK" for r in pgr); pto = sum(r[7] == "TIMEOUT" for r in pgr)
    report.append(f"[{label}] {run_id}: train/test={len(tr)}/{len(te)} | eval rows={len(ev)} "
                  f"(Bao OK={ok}, TIMEOUT={to}; PG OK={pok}, TIMEOUT={pto}) | trace rows={len(trace)} "
                  f"| checkpoint={'OK' if ck_ok else 'MISSING'}")
    report.extend(f"    ! {e}" for e in errs)
    return ev, pgr


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--console", default=os.path.join(BAO, "logs", "console_evaluation.log"))
    ap.add_argument("--git-commit", default="", help="commit upstream; bỏ trống = giữ giá trị trong manifest cũ")
    ap.add_argument("--pg-version", default="", help="bỏ trống = giữ giá trị trong manifest cũ")
    ap.add_argument("--upstream-queries", default="", help="(tuỳ chọn) <lqo_ml_perspective>/bao/queries để đối chiếu")
    ap.add_argument("--out", default=os.path.join(BAO, "results"))
    args = ap.parse_args()
    for need in [os.path.join(BAO, "logs"), os.path.join(BAO, "checkpoints"),
                 os.path.join(ROOT, "analysis", "data", "split_master.csv")]:
        if not os.path.exists(need):
            sys.exit(f"Không thấy {need}. Script phải nằm ở <team-repo>/bao/scripts/ (BAO={BAO}).")
    sm_path = os.path.join(ROOT, "analysis", "data", "split_master.csv")
    args.split_master_rows = list(csv.DictReader(open(sm_path, encoding="utf-8-sig")))

    report, all_ev, all_pg = [], [], []
    for s in SPLITS:
        ev, pg = build(s, args, report)
        all_ev += ev
        all_pg += pg
    runs = args.out
    write_csv(os.path.join(runs, "ALL_evaluation_results.csv"),
              ["run_id", "model", "split_type", "split_id", "query_id", "family", "checkpoint_id", "planning_ms",
               "inference_ms", "execution_ms", "end_to_end_ms", "status", "timeout_ms", "chosen_plan_id", "error", "warning",
               "chosen_plan_ids"],
              all_ev)
    write_csv(os.path.join(runs, "ALL_postgres_results.csv"),
              ["split_type", "split_id", "query_id", "family", "planning_ms", "execution_ms", "end_to_end_ms",
               "status", "timeout_ms"], all_pg)
    report.append(f"TOTAL evaluation rows = {len(all_ev)} (mong đợi 74)")
    nov_path = os.path.join(ROOT, "analysis", "data", "novelty.csv")
    if os.path.exists(nov_path):
        nov = {(r["split_type"], str(r["split_id"]), r["query_id"])
               for r in csv.DictReader(open(nov_path, encoding="utf-8-sig"))}
        keys = {(r[2], str(r[3]), r[4]) for r in all_ev}
        miss = sorted(keys - nov)
        report.append(f"MERGE novelty.csv on (split_type, split_id, query_id): {len(keys & nov)}/{len(keys)} khớp"
                      + (f"  ! thiếu: {miss}" if miss else ""))
    else:
        report.append(f"  ! không thấy {nov_path}")
    multi = [f"{r[2]}-{r[4]}:{r[16]}" for r in all_ev if r[13] == "MULTIPLE"]
    report.append(f"chosen_plan_id = MULTIPLE: {len(multi)} query {multi if multi else ''}")
    text = "\n".join(report) + "\n"
    open(os.path.join(runs, "VALIDATION.txt"), "w", encoding="utf-8").write(text)
    print(text)
    print("Output:", runs)
    sys.exit(0 if len(all_ev) == 74 and "!" not in text else 1)


if __name__ == "__main__":
    main()
