"""
Score one test fold of the decision-model data with a decision model served through Unsloth Studio's Decision API, and compare it with "keep the last take".

    UNSLOTH_API_KEY=sk-unsloth-... python training/score_decision_fold.py --fold 2 \
        --test data/training/sft_decision/decision_fold2_test.jsonl --keys data/training/sft_decision/decision_fold2_test.keys.txt
    python training/score_decision_fold.py --fold 2 --test ... --keys ... --probs-only     # rescore from the saved probabilities, no API call

API (Studio /openapi.json): POST {base-url}/systemone  {"model", "state", "questions"}  ->  {"model", "answers": {"keep_<id>": {"type": "noul", "noul": p_yes}}}.
`--model default` routes to the model selected in Studio Settings > API; the response's "model" field is recorded so the model that answered is visible.
Per test row (file order) one request asks one yes/no question per unit; the gold is never sent. Rule (training/decision_eval.py, not modified): keep = units with
p >= 0.5, an empty set = keep_none, unclear = top two within 0.15 or max p < 0.6. A group that cannot be scored is logged, kept in the output with its error, and
counts as WRONG (never dropped). Resumable: groups already in results/fold{k}_probs.jsonl without an error are not asked again. Retries with backoff; a model that
is still loading (HTTP 503 model_loading) is waited for up to --load-wait seconds.

Outputs next to the test file, in results/: fold{k}_probs.jsonl (resume store), decision_fold{k}_predictions.csv (one row per group), decision_fold_summary.csv
(one row per fold; re-running a fold replaces its row so pooled counts are never doubled).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from training.decision_eval import decide, exact_match, gold_keep, score  # noqa: E402

DEFAULT_BASE = "http://127.0.0.1:8888/v1"
KEY_ENV = "UNSLOTH_API_KEY"


def _clean(msg: str, key: str | None) -> str:
    return str(msg).replace(key, "***") if key else str(msg)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Retry(Exception):
    pass


def ask(session: requests.Session, url: str, model: str, row: dict, headers: dict, timeout: float) -> tuple[dict[str, float], str]:
    questions = json.loads(row["questions"])                          # the gold is never part of the payload
    r = session.post(url, json={"model": model, "state": row["state"], "questions": questions}, headers=headers, timeout=timeout)
    if r.status_code == 503 and "model_loading" in r.text:
        raise Retry("model still loading")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    body = r.json()
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise RuntimeError("response has no 'answers' object")
    probs: dict[str, float] = {}
    for qid in questions:
        a = answers.get(qid)
        v = a.get("noul") if isinstance(a, dict) else None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
            raise RuntimeError(f"answer for {qid} is not a probability: {a!r}"[:200])
        probs[qid[len("keep_"):] if qid.startswith("keep_") else qid] = float(v)
    return probs, str(body.get("model", "?"))


def call_api(rows: list[dict], keys: list[str], done: dict[str, dict], args, key: str | None, store: Path) -> list[dict]:
    url = args.base_url.rstrip("/") + "/systemone"
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    session = requests.Session()
    out = []
    n_new = 0
    for i, (row, gid) in enumerate(zip(rows, keys)):
        if gid in done and not done[gid].get("error"):                  # resume: keep what was already scored
            out.append(done[gid])
            continue
        rec: dict = {"group_id": gid, "probs": {}, "error": None, "served_by": None}
        deadline = time.time() + args.load_wait
        attempt = 0
        while True:
            try:
                rec["probs"], rec["served_by"] = ask(session, url, args.model, row, headers, args.timeout)
                rec["error"] = None
                break
            except Retry as exc:                                         # the model is loading: wait without using up the retries
                rec["error"] = str(exc)
                if time.time() > deadline:
                    break
                time.sleep(15.0)
            except Exception as exc:
                rec["error"] = _clean(f"{type(exc).__name__}: {exc}", key)
                attempt += 1
                if attempt > args.retries:
                    break
                time.sleep(min(60.0, 5.0 * 2 ** (attempt - 1)))
        if rec["error"]:
            print(f"  FAILED {gid}: {rec['error'][:160]}", flush=True)
        out.append(rec)
        n_new += 1
        store.parent.mkdir(parents=True, exist_ok=True)                  # write after every group so a crash loses nothing
        store.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in out + [done[g] for g in keys[len(out):] if g in done]), encoding="utf-8", newline="\n")
        if n_new % 10 == 0:
            print(f"  {i + 1}/{len(rows)} groups done, {sum(1 for o in out if o['error'])} failed", flush=True)
    return out


def group_type(gold: dict) -> str:
    n = sum(1 for v in gold.values() if v)
    return "keep_none" if n == 0 else "single" if n == 1 else "multi"


def build_table(rows: list[dict], keys: list[str], probs_rows: list[dict], meta: dict[str, dict]) -> list[dict]:
    table = []
    for row, gid, pr in zip(rows, keys, probs_rows):
        gold = json.loads(row["gold"])
        units = [u[len("keep_"):] for u in gold]
        gk = gold_keep(gold)
        failed = bool(pr.get("error"))
        d: dict | None = None if failed else decide(pr["probs"])
        keep_last = [units[-1]]
        table.append({
            "group_id": gid, "n_units": len(units), "group_type": group_type(gold), "synthetic": bool(meta.get(gid, {}).get("synthetic")),
            "gold_keep": "|".join(gk), "model_keep": "" if d is None else "|".join(d["keep"]), "model_keep_none": "" if d is None else d["keep_none"],
            "unclear": "" if d is None else d["unclear"], "keep_last": "|".join(keep_last), "failed": failed, "error": pr.get("error") or "",
            "model_correct": bool(d is not None and exact_match(d["keep"], gold)), "keep_last_correct": exact_match(keep_last, gold),
            "probs": json.dumps(pr["probs"]), "served_by": pr.get("served_by") or "",
        })
    return table


def bootstrap_diff(model_ok: np.ndarray, last_ok: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = model_ok.astype(float) - last_ok.astype(float)
    idx = rng.integers(0, len(d), size=(n, len(d)))                     # resample groups with replacement
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def pct(x: float) -> str:
    return f"{x:.1%}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--meta", default=None, help="decision_meta.jsonl (default: next to --test); used for natural vs synthetic")
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default="default")
    ap.add_argument("--out-dir", default=None, help="default: <test dir>/results")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--load-wait", type=float, default=600.0, help="seconds to wait for a model that is still loading")
    ap.add_argument("--probs-only", action="store_true")
    ap.add_argument("--no-auth", action="store_true")
    args = ap.parse_args()

    test, keys_path = Path(args.test), Path(args.keys)
    rows = read_rows(test)
    keys = keys_path.read_text(encoding="utf-8").split()
    if len(rows) != len(keys):
        print(f"error: {len(rows)} rows in {test.name} but {len(keys)} keys", file=sys.stderr)
        return 2
    meta_path = Path(args.meta) if args.meta else test.parent / "decision_meta.jsonl"
    meta = {m["group_id"]: m for m in read_rows(meta_path)} if meta_path.exists() else {}
    out_dir = Path(args.out_dir) if args.out_dir else test.parent / "results"
    store = out_dir / f"fold{args.fold}_probs.jsonl"
    done = {p["group_id"]: p for p in read_rows(store)} if store.exists() else {}

    if args.probs_only:
        if not store.exists() or any(g not in done for g in keys):
            print("error: --probs-only needs a complete probs file for this fold", file=sys.stderr)
            return 2
        probs_rows = [done[g] for g in keys]
    else:
        key = None if args.no_auth else os.environ.get(KEY_ENV)
        if not key and not args.no_auth:
            print(f"error: set {KEY_ENV} or pass --no-auth", file=sys.stderr)
            return 2
        print(f"fold {args.fold}: {len(rows)} groups ({len([g for g in keys if g in done and not done[g].get('error')])} already scored) -> POST {args.base_url.rstrip('/')}/systemone (model '{args.model}')", flush=True)
        probs_rows = call_api(rows, keys, done, args, key, store)
        store.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in probs_rows), encoding="utf-8", newline="\n")
        served = sorted({p["served_by"] for p in probs_rows if p.get("served_by")})
        print(f"answered by model(s): {served}" + ("   <-- WARNING: Laya is the stock model, not yours" if any("laya" in s.lower() for s in served) else ""))

    table = build_table(rows, keys, probs_rows, meta)
    n = len(table)
    failed = sum(t["failed"] for t in table)
    m_ok = np.array([t["model_correct"] for t in table])
    l_ok = np.array([t["keep_last_correct"] for t in table])
    if failed == 0:                                                      # cross-check against the shared scorer
        ref = score([r["gold"] for r in rows], [p["probs"] for p in probs_rows])
        assert abs(ref["exact_match"] - m_ok.mean()) < 1e-12, "table disagrees with decision_eval.score"
    scored = [t for t in table if not t["failed"]]
    unclear = sum(1 for t in scored if t["unclear"] is True)

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"decision_fold{args.fold}_predictions.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    diff, lo, hi = bootstrap_diff(m_ok, l_ok)
    print(f"\n=== fold {args.fold}: {n} test groups, {failed} failed to score (counted wrong) ===")
    print(f"natural / synthetic groups in the test file: {sum(not t['synthetic'] for t in table)} / {sum(t['synthetic'] for t in table)}"
          + ("" if not any(t["synthetic"] for t in table) else "   <-- the test set should be natural only"))
    print(f"model accuracy     : {pct(m_ok.mean())}  ({int(m_ok.sum())}/{n})")
    print(f"keep-last accuracy : {pct(l_ok.mean())}  ({int(l_ok.sum())}/{n})")
    print(f"difference         : {diff * 100:+.1f} points   95% bootstrap interval by group [{lo * 100:+.1f}, {hi * 100:+.1f}] (1,000 resamples, seed 0)")
    print(f"unclear predictions: {unclear}/{len(scored)} ({pct(unclear / max(len(scored), 1))})")
    print("by group type (model correct / keep-last correct / groups):")
    for kind in ("single", "multi", "keep_none"):
        sel = [t for t in table if t["group_type"] == kind]
        print(f"   {kind:<10}: {sum(t['model_correct'] for t in sel)} / {sum(t['keep_last_correct'] for t in sel)} / {len(sel)}")
    for nat, label in ((True, "natural"), (False, "synthetic")):
        sel = [t for t in table if t["synthetic"] != nat]
        if sel:
            print(f"   {label:<10}: {sum(t['model_correct'] for t in sel)} / {sum(t['keep_last_correct'] for t in sel)} / {len(sel)}")
    dis = [t for t in table if t["model_correct"] != t["keep_last_correct"] or (t["model_keep"] != t["keep_last"])]
    print(f"\ngroups where the model and keep-last differ ({len(dis)}):")
    for t in dis:
        pr = json.loads(t["probs"])
        who = "model right" if t["model_correct"] else ("keep-last right" if t["keep_last_correct"] else "both wrong")
        print(f"  {t['group_id']:<16} [{who}] gold={t['gold_keep'] or '(none)'}  model={t['model_keep'] or ('FAILED' if t['failed'] else '(none)')}  keep-last={t['keep_last']}  p={pr}")

    summary = out_dir / "decision_fold_summary.csv"
    cols = ["fold", "n_groups", "failed", "model_correct", "keep_last_correct", "model_accuracy", "keep_last_accuracy", "unclear_count", "diff_points", "ci_low_points", "ci_high_points"]
    old = []
    if summary.exists():
        with open(summary, newline="", encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f) if r["fold"] != str(args.fold)]      # one row per fold: a re-run replaces it
    new = {"fold": args.fold, "n_groups": n, "failed": failed, "model_correct": int(m_ok.sum()), "keep_last_correct": int(l_ok.sum()), "model_accuracy": round(float(m_ok.mean()), 4),
           "keep_last_accuracy": round(float(l_ok.mean()), 4), "unclear_count": unclear, "diff_points": round(diff * 100, 2), "ci_low_points": round(lo * 100, 2), "ci_high_points": round(hi * 100, 2)}
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(old + [new])
    print(f"\nwrote {pred_path}\nwrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
