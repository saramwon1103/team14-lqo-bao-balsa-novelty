#!/usr/bin/env python3
"""Rút gọn experiment log đầy đủ thành 1 khối/mục, đúng template nhóm."""
import re, sys

def blocks(path):
    head, cur, out = {}, None, []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("====="):
            continue
        m = re.match(r"^\[(1|2|3|4|POSTGRESQL)\]", line)
        if m:
            cur = {"sec": m.group(1), "title": line, "kv": {}}
            out.append(cur)
        elif ":" in line:
            k, v = line.split(":", 1)
            (cur["kv"] if cur else head)[k.strip()] = v.strip()
    return head, out

def num(v):
    m = re.match(r"^\s*(-?[\d.]+)", v or "")
    return float(m.group(1)) if m else None

def main(path):
    head, bs = blocks(path)
    split, model = head.get("Split", "?"), head.get("Model", "Bao")
    b1 = [b["kv"] for b in bs if b["sec"] == "1"]
    b2 = [b["kv"] for b in bs if b["sec"] == "2"]
    b3 = next((b["kv"] for b in bs if b["sec"] == "3"), {})
    tests = []
    for i, b in enumerate(bs):
        if b["sec"] == "4":
            pg = bs[i + 1]["kv"] if i + 1 < len(bs) and bs[i + 1]["sec"] == "POSTGRESQL" else {}
            tests.append((b["kv"], pg))

    yes = lambda L, k: sum(x.get(k) == "YES" for x in L)
    avg = lambda L, k: (lambda v: sum(v) / len(v) if v else None)([num(x.get(k)) for x in L if num(x.get(k)) is not None])
    f1 = lambda v: "N/A" if v is None else f"{v:.1f}"

    both = [(b, p) for b, p in tests if num(b.get("End-to-End")) is not None and num(p.get("End-to-End")) is not None]
    s = lambda L, k: sum(num(x.get(k)) for x in L)
    bl, pl = [b for b, _ in both], [p for _, p in both]
    to_b = [b["Test query"] for b, _ in tests if b.get("Status") != "COMPLETED"]
    to_p = [b["Test query"] for b, p in tests if num(p.get("End-to-End")) is None]
    bao_e2e, pg_e2e = s(bl, "End-to-End"), s(pl, "End-to-End")

    L = ["===== EXPERIMENT START =====", "", f"Split: {split}", f"Model: {model}", "",
         "[1] TRAINING DATA GENERATION",
         f"Train query: {len(b1)} train queries",
         f"Plan collected: {'YES' if yes(b1, 'Plan collected') == len(b1) else 'PARTIAL'} ({yes(b1, 'Plan collected')}/{len(b1)})",
         f"Cost: {f1(avg(b1, 'Cost'))} (trung bình)",
         f"Cardinality: {f1(avg(b1, 'Cardinality'))} (trung bình)",
         f"Execution time: {f1(avg(b1, 'Execution time'))} ms (trung bình)", "",
         "[2] QUERY & PLAN ENCODING",
         f"Query: {len(b2)} train queries",
         f"Query/Plan encoded: {'YES' if yes(b2, 'Query/Plan encoded') == len(b2) else 'PARTIAL'} ({yes(b2, 'Query/Plan encoded')}/{len(b2)})",
         f"Model input generated: {'YES' if yes(b2, 'Model input generated') == len(b2) else 'PARTIAL'} ({yes(b2, 'Model input generated')}/{len(b2)})", "",
         "[3] LQO TRAINING",
         f"Model: {b3.get('Model', model)}", f"Split: {b3.get('Split', split)}",
         f"Training status: {b3.get('Training status', 'N/A')}",
         f"Training time: {b3.get('Training time', 'N/A')}",
         f"Model/state: {b3.get('Model/state', 'N/A')}", "",
         "[4] LQO EVALUATION",
         f"Test query: {len(tests)} test queries (so sánh trên {len(both)} query cả 2 cùng chạy xong)",
         f"{model} inference/planning: {s(bl, model + ' inference/planning'):.1f} ms",
         f"Execution time: {s(bl, 'Execution time'):.1f} ms",
         f"End-to-End: {bao_e2e:.1f} ms",
         f"Status: COMPLETED {len(tests) - len(to_b)}/{len(tests)}" + (f" (TIMEOUT: {', '.join(to_b)})" if to_b else ""), "",
         "[POSTGRESQL]",
         f"Planning: {s(pl, 'Planning'):.1f} ms",
         f"Execution: {s(pl, 'Execution'):.1f} ms",
         f"End-to-End: {pg_e2e:.1f} ms",
         f"Status: COMPLETED {len(tests) - len(to_p)}/{len(tests)}" + (f" (TIMEOUT: {', '.join(to_p)})" if to_p else ""),
         f"Speedup (PostgreSQL / {model}): {pg_e2e / bao_e2e:.2f}x", "",
         "===== EXPERIMENT END ====="]
    out = re.sub(r"\.log$", "", path) + "_summary.log"
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(open(out, encoding="utf-8").read())
    print(f"-> {out}\n")

for p in sys.argv[1:]:
    main(p)
