"""Ambiguous-band comparison with FREE Groq models (resumable, quota-aware).

Re-runs the earlier band comparison with openai/gpt-oss-120b, qwen/qwen3.8-27b (and optionally gpt-oss-20b), reusing
the fixed prompts from adjudication_compare.py (extracted from its source with ast, so the text is byte-identical)
and results/fewshot_examples.json. Writes NEW files only; it reads (never modifies) adjudication_scores.csv,
baseline_scores.csv, ce_scores.csv, stage2_errors.csv, gemini_cache.jsonl (llama3 verdicts) and folds.json.

Usage:  python adjudication_groq.py          (rerun the same command to resume after a daily-quota stop)

Stages, in order (each writes a short interim report before the next starts):
  1 gpt-oss-120b zero-shot on the 861 band pairs
  2 qwen3.8-27b zero-shot on the same pairs
  3 gpt-oss-120b few-shot, only if stage 1 band F1 > 0.30 OR (recall > 0.6 and precision > 0.3)
  4 extra: the better model, zero-shot, on all 79 Stage 2 errors (the 31 out-of-band ones are new calls)
The key is read from GROQ_API_KEY (environment, else the repo's git-ignored .env) and is never printed or logged.
"""
import ast
import collections
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import precision_recall_curve

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
CACHE, STATUS = RES / "groq_cache.jsonl", RES / "groq_status.txt"
BASE_URL = "https://api.groq.com/openai/v1"
LO, HI, S2_T = 0.45, 0.65, 0.65
MODELS = {"gpt120": "openai/gpt-oss-120b", "qwen": "qwen/qwen3.8-27b", "gpt20": "openai/gpt-oss-20b"}
ZERO, FEW = "zero_v1", "few_v1"            # same prompt text as the llama3 runs; the cache key also carries the model
MAX_RPM, TPM_BUDGET, MAX_TOKENS = 15, 7000, 400
MIN_INTERVAL = 60.0 / MAX_RPM
HARD_CAP = 2500                            # calls per (model, variant), so a bug can never loop
N_BOOT, SEED = 1000, 0
TODAY = date.today().isoformat()

# ---------------- key (never printed) ----------------
KEY = os.environ.get("GROQ_API_KEY", "")
if not KEY and (REPO / ".env").exists():
    for line in (REPO / ".env").read_text().splitlines():
        if line.strip().startswith("GROQ_API_KEY="):
            KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
if not KEY:
    raise SystemExit("GROQ_API_KEY is missing (environment and .env); stopping.")


def scrub(x):
    return str(x).replace(KEY, "<key>")


def set_status(text):
    STATUS.write_text(text + "\n", encoding="utf-8")


# ---------------- fixed prompt, extracted unchanged from adjudication_compare.py ----------------
def load_prompt_fn():
    tree = ast.parse((HERE / "adjudication_compare.py").read_text(encoding="utf-8"))
    nodes = [n for n in tree.body
             if (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in {"HEAD", "TAIL"} for t in n.targets))
             or (isinstance(n, ast.FunctionDef) and n.name in {"pair_block", "make_prompt"})]
    ns = {"json": json, "ZERO": "zero", "FEW": "few", "fewshot": json.loads((RES / "fewshot_examples.json").read_text())}
    exec(compile(ast.Module(nodes, []), "prompt_extract", "exec"), ns)
    return lambda row, variant: ns["make_prompt"](row, "zero" if variant == ZERO else "few")


make_prompt = load_prompt_fn()

# ---------------- data and alignment checks ----------------
sc = pd.read_csv(RES / "adjudication_scores.csv", dtype={"pair_id": str, "video_id": str})
base = pd.read_csv(RES / "baseline_scores.csv", dtype={"pair_id": str})
ce = pd.read_csv(RES / "ce_scores.csv", dtype={"pair_id": str}).drop_duplicates("pair_id")
assert set(sc.pair_id) == set(base.pair_id) == set(ce.pair_id) and len(sc) == len(base) == 3570, "files do not align on pair_id"
chk = sc.merge(base[["pair_id", "current_stage2"]], on="pair_id", suffixes=("", "_b")).merge(ce[["pair_id", "prob_mean"]], on="pair_id")
assert np.allclose(chk.current_stage2, chk.current_stage2_b) and np.allclose(chk.ce_prob, chk.prob_mean), "score mismatch"
pv = pd.read_csv(HERE / "pairs_v1.csv", dtype=str, keep_default_na=False)[["pair_id", "text_a", "text_b"]]
df = sc.merge(pv, on="pair_id", validate="1:1").reset_index(drop=True)
df["fold"] = df.fold.astype(int)
df["in_band"] = (df.current_stage2 >= LO) & (df.current_stage2 < HI)
assert int(df.in_band.sum()) == 861 and (df.in_band == (df.band_flag.astype(str) == "True")).all()
df["s2_pred"] = (df.current_stage2 >= S2_T).astype(int)
BAND = df[df.in_band]
ERRORS_OUT = df[(df.s2_pred != df.label) & ~df.in_band]                  # the 31 Stage 2 errors outside the band
errs = pd.read_csv(RES / "stage2_errors.csv", dtype=str)
assert len(errs) == 79 and len(ERRORS_OUT) == 31 and int(((df.s2_pred != df.label)).sum()) == 79

# ---------------- cache ----------------
cache = {}
if CACHE.exists():
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        cache[(r["pair_id"], r["model"], r["variant"])] = r


def append_cache(rec):
    cache[(rec["pair_id"], rec["model"], rec["variant"])] = rec
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")                                   # written immediately after each call


# ---------------- rate limiting ----------------
last_start = [0.0]
window = collections.deque()                                              # (time, total_tokens) over the last 60 s
run_stats = collections.defaultdict(lambda: {"calls": 0, "failed": 0, "attempts": 0, "tokens": 0})
headers_seen = {}


class DailyQuota(Exception):
    pass


class Stop(Exception):
    pass


def throttle(est_tokens):
    """At most MAX_RPM requests/min and a rolling tokens-per-minute count under TPM_BUDGET."""
    while True:
        now = time.time()
        while window and now - window[0][0] > 60:
            window.popleft()
        used = sum(t for _, t in window)
        wait_rpm = max(0.0, last_start[0] + MIN_INTERVAL - now)
        wait_tpm = (window[0][0] + 60 - now + 0.5) if (used + est_tokens > TPM_BUDGET and window) else 0.0
        wait = max(wait_rpm, wait_tpm)
        if wait <= 0:
            last_start[0] = time.time()
            return
        time.sleep(min(wait, 5.0))


def parse_verdict(content):
    """First JSON object in the reply (a <think> block is dropped); must hold a boolean same_intended_line."""
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    s, e = content.find("{"), content.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("no JSON object")
    obj = json.loads(content[s:e + 1])
    if not isinstance(obj.get("same_intended_line"), bool):
        raise ValueError("same_intended_line missing or not boolean")
    return obj


def request_body(model, prompt, with_format=True):
    body = {"model": model, "temperature": 0, "max_completion_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}]}
    if with_format:
        body["response_format"] = {"type": "json_object"}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    elif model.startswith("qwen/"):
        body["reasoning_effort"] = "none"                                # keep tokens low; qwen would otherwise think at length
    return body


def one_call(model, variant, row):
    """One cached call with retries. Returns the cache record; raises DailyQuota / Stop."""
    key = (row.pair_id, model, variant)
    if key in cache:
        return cache[key]
    prompt = make_prompt(row, variant)
    tokens_est = int(np.mean([t for _, t in window])) if window else 450
    parse_fail, transient, last_err = 0, 0, ""
    while True:
        throttle(tokens_est)
        run_stats[(model, variant)]["attempts"] += 1
        try:
            r = requests.post(f"{BASE_URL}/chat/completions", headers={"Authorization": f"Bearer {KEY}"},
                              json=request_body(model, prompt), timeout=90)
        except requests.RequestException as e:
            transient += 1
            last_err = scrub(e)[:200]
            if transient >= 5:
                break
            time.sleep(min(60, 2 ** transient * 2))
            continue
        for h, v in r.headers.items():
            if h.lower().startswith("x-ratelimit-limit"):
                headers_seen[h.lower()] = v
        if r.status_code == 429:
            msg = scrub(r.text).lower()
            if "per day" in msg or "(tpd)" in msg or "(rpd)" in msg:
                raise DailyQuota(msg[:200])
            time.sleep(min(60, float(r.headers.get("retry-after", 20)) + 1))   # per-minute limit: back off, retry, not counted
            continue
        if r.status_code in (402,) or (r.status_code >= 400 and any(w in scrub(r.text).lower() for w in ("billing", "payment required", "paid plan"))):
            raise Stop(f"API indicates billing/paid plan (HTTP {r.status_code}); stopping to ask the user")
        if r.status_code >= 500:
            transient += 1
            last_err = f"HTTP {r.status_code}"
            if transient >= 5:
                break
            time.sleep(min(60, 2 ** transient * 2))
            continue
        if r.status_code != 200:
            raise Stop(f"HTTP {r.status_code}: {scrub(r.text)[:300]}")
        j = r.json()
        u = j.get("usage", {})
        window.append((time.time(), int(u.get("total_tokens", tokens_est))))
        rs = run_stats[(model, variant)]
        rs["tokens"] += int(u.get("total_tokens", 0))
        try:
            verdict = parse_verdict(j["choices"][0]["message"].get("content") or "")
            rec = {"pair_id": row.pair_id, "model": model, "variant": variant, "failed": False,
                   "same_intended_line": verdict["same_intended_line"], "confidence": verdict.get("confidence"),
                   "usage": {"prompt": u.get("prompt_tokens", 0), "completion": u.get("completion_tokens", 0),
                             "total": u.get("total_tokens", 0),
                             "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)},
                   "date": TODAY}
            rs["calls"] += 1
            append_cache(rec)
            return rec
        except Exception as e:                                           # parse failure: retry at most twice
            parse_fail += 1
            last_err = f"parse: {e}"
            if parse_fail > 2:
                break
    rec = {"pair_id": row.pair_id, "model": model, "variant": variant, "failed": True, "error": last_err,
           "usage": {"prompt": 0, "completion": 0, "total": 0, "reasoning": 0}, "date": TODAY}
    run_stats[(model, variant)]["failed"] += 1
    append_cache(rec)
    return rec


def run_stage(model, variant, rows, label):
    """Run (or resume) calls for `rows`. Returns True if every row now has a cache entry."""
    n_have = sum(1 for r in rows.itertuples() if (r.pair_id, model, variant) in cache)
    total_cached = sum(1 for k in cache if k[1] == model and k[2] == variant)
    print(f"[{label}] {model} {variant}: {n_have}/{len(rows)} already cached")
    reported50 = False
    for r in rows.itertuples():
        if (r.pair_id, model, variant) in cache:
            continue
        if total_cached >= HARD_CAP:
            raise Stop(f"hard cap of {HARD_CAP} calls reached for {model} {variant}")
        one_call(model, variant, r)
        total_cached += 1
        done = sum(1 for x in rows.itertuples() if (x.pair_id, model, variant) in cache)
        rs = run_stats[(model, variant)]
        if not reported50 and rs["calls"] + rs["failed"] >= 50:
            reported50 = True
            per_call = rs["tokens"] / max(1, rs["calls"])
            remaining = len(rows) - done
            days = max(remaining / 1000.0, remaining * per_call / 200000.0)
            msg = (f"after 50 calls: {per_call:.0f} tokens/call measured; {remaining} calls remain; projected "
                   f"{days:.1f} day(s) if 1,000 req/day and 200,000 tokens/day per model apply")
            print(f"[{label}] {msg}")
            set_status(f"partial: running {model} {variant}; {msg}")
        if done % 100 == 0:
            print(f"[{label}] {done}/{len(rows)}")
    return True


# ---------------- evaluation helpers ----------------
def verdicts(model, variant, rows=None):
    """pair_id -> True/False (None for a failed call, missing if not attempted)."""
    out = {}
    for k, r in cache.items():
        if k[1] == model and k[2] == variant:
            out[k[0]] = None if r["failed"] else r["same_intended_line"]
    return out


def prf(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": pr, "recall": rc, "F1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "n": len(y), "flag_rate": (tp + fp) / max(1, len(y))}


def band_complete(model, variant):
    v = verdicts(model, variant)
    return all(p in v for p in BAND.pair_id)


def band_metrics(model, variant):
    v = verdicts(model, variant)
    b = BAND[BAND.pair_id.map(lambda p: v.get(p) is not None)]
    return prf(b.label, b.pair_id.map(lambda p: int(v[p]))), int(sum(1 for p in BAND.pair_id if v.get(p) is None))


def write_interim(stage, model, variant, extra=""):
    m, failed = band_metrics(model, variant)
    rs = [r for k, r in cache.items() if k[1] == model and k[2] == variant and not r["failed"]]
    tok = sum(r["usage"]["total"] for r in rs)
    lines = [f"# Interim report, stage {stage}: {model} {variant}", "", f"Run date {TODAY}. Band [{LO}, {HI}): {len(BAND)} pairs.", "",
             "| precision | recall | F1 | TP | FP | FN | TN | flag rate | failed calls |", "|---|---|---|---|---|---|---|---|---|",
             f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['F1']:.3f} | {m['TP']} | {m['FP']} | {m['FN']} | {m['TN']} | {m['flag_rate']:.1%} | {failed} |",
             "", f"Measured tokens: {tok:,} over {len(rs)} successful calls ({tok / max(1, len(rs)):.0f} per call).", extra]
    (RES / f"adjudication_groq_interim_stage{stage}.md").write_text("\n".join(lines), encoding="utf-8")
    return m


def decide_next(m):
    return m["F1"] > 0.30 or (m["recall"] > 0.6 and m["precision"] > 0.3)


# ---------------- run the stages ----------------
state = {"stage": "start"}
available = {}
try:
    ids = {m["id"] for m in requests.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {KEY}"}, timeout=30).json().get("data", [])}
    available = {k: v for k, v in MODELS.items() if v in ids}
    print("models available:", {k: v in ids for k, v in MODELS.items()})
    assert "gpt120" in available, "primary model not available for this key"

    state["stage"] = "1 gpt120 zero"
    run_stage(MODELS["gpt120"], ZERO, BAND, "stage 1")
    m1 = write_interim(1, MODELS["gpt120"], ZERO)
    run3 = decide_next(m1)

    best = MODELS["gpt120"]
    if "qwen" in available:
        state["stage"] = "2 qwen zero"
        run_stage(MODELS["qwen"], ZERO, BAND, "stage 2")
        m2 = write_interim(2, MODELS["qwen"], ZERO)
        if m2["F1"] > m1["F1"]:
            best = MODELS["qwen"]

    if run3:
        state["stage"] = "3 gpt120 few"
        run_stage(MODELS["gpt120"], FEW, BAND, "stage 3")
        write_interim(3, MODELS["gpt120"], FEW)
    else:
        (RES / "adjudication_groq_interim_stage3.md").write_text(
            f"# Stage 3 skipped\n\ngpt-oss-120b zero-shot band F1 {m1['F1']:.3f} (precision {m1['precision']:.3f}, recall {m1['recall']:.3f}) "
            "did not meet the condition (F1 > 0.30, or recall > 0.6 with precision > 0.3).\n", encoding="utf-8")

    state["stage"] = f"4 extra {best} zero on out-of-band errors"
    run_stage(best, ZERO, ERRORS_OUT, "stage 4")
    status = "done"
except DailyQuota as e:
    status = f"partial: daily quota reached while on stage {state['stage']}; rerun the same command after the quota resets. ({e})"
except Stop as e:
    status = f"failed: {e}"
except AssertionError as e:
    status = f"failed: {e}"
except KeyboardInterrupt:
    status = f"partial: interrupted during stage {state['stage']}; rerun to resume"
print("status:", status)

# remaining-call bookkeeping for the status file
todo = [("stage 1", MODELS["gpt120"], ZERO, BAND), ("stage 2", MODELS["qwen"], ZERO, BAND), ("stage 3", MODELS["gpt120"], FEW, BAND)]
remaining = {f"{m} {v}": sum(1 for p in rows.pair_id if (p, m, v) not in cache) for _, m, v, rows in todo}
if status.startswith("partial"):
    status += " | remaining calls: " + ", ".join(f"{k}: {n}" for k, n in remaining.items() if n)
set_status(status)

# ---------------- evaluation and report (uses whatever is complete) ----------------
# cross-encoder fold thresholds, recomputed exactly as in adjudication_compare.py and checked against ce_decision
thr = {}
for k in sorted(df.fold.unique()):
    tr = df[(df.fold != k) & df.in_band]
    prec, rec, th = precision_recall_curve(tr.label, tr.ce_prob)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    thr[k] = float(th[np.argmax(f1)])
df["ce_thr"] = df.fold.map(thr)
assert ((df.ce_prob >= df.ce_thr).astype(int) == df.ce_decision.astype(int)).all(), "cross-encoder decisions not reproduced"
df["ce_pred"] = df.ce_decision.astype(int)

systems = {"stage2": df.s2_pred.values.astype(float), "cross_encoder": np.where(df.in_band, df.ce_pred, df.s2_pred).astype(float)}
band_dec = {}                                                              # name -> series over df (NaN = no verdict)
for col, name in (("llm_zero", "llama3_zero"), ("llm_few", "llama3_few")):
    band_dec[name] = pd.to_numeric(df[col].map({True: 1, False: 0, "True": 1, "False": 0}), errors="coerce").where(df.in_band)
short = {MODELS["gpt120"]: "gptoss120b", MODELS["qwen"]: "qwen27b", MODELS["gpt20"]: "gptoss20b"}
used_groq = []
for model, variant in [(MODELS["gpt120"], ZERO), (MODELS["qwen"], ZERO), (MODELS["gpt120"], FEW), (MODELS["gpt20"], ZERO)]:
    v = verdicts(model, variant)
    if band_complete(model, variant) and sum(1 for k in cache if k[1] == model and k[2] == variant) >= len(BAND):
        name = f"{short[model]}_{'zero' if variant == ZERO else 'few'}"
        band_dec[name] = df.pair_id.map(lambda p: (None if v.get(p) is None else int(v[p])) if p in v else np.nan).astype(float).where(df.in_band)
        used_groq.append(name)
for name, ser in band_dec.items():
    systems[name] = np.where(df.in_band, ser.fillna(df.s2_pred), df.s2_pred).astype(float)   # failed/missing pair -> Stage 2 decision
    # combinations with the cross-encoder (saved verdicts only, no new calls)
    ce_ = df.ce_pred.values.astype(float)
    llm = ser.values
    have = ~np.isnan(llm)
    near = np.abs(df.ce_prob.values - df.ce_thr.values) <= 0.1
    comb = {"AND": np.where(have, (ce_ == 1) & (llm == 1), ce_ == 1),
            "CEtie": np.where(have & near, llm == 1, ce_ == 1),
            "OR": np.where(have, (ce_ == 1) | (llm == 1), ce_ == 1)}
    for cname, arr in comb.items():
        systems[f"{name}+CE_{cname}"] = np.where(df.in_band, arr.astype(float), df.s2_pred)

y_all = df.label.values
rows = []
for name, arr in systems.items():
    e2e = prf(y_all, arr)
    rows.append((name, e2e))

rng = np.random.default_rng(SEED)
vids = df.video_id.unique()
idx_by_v = {v: np.where(df.video_id.values == v)[0] for v in vids}
boot = {n: [] for n in systems}
for _ in range(N_BOOT):
    idx = np.concatenate([idx_by_v[v] for v in rng.choice(vids, len(vids), replace=True)])
    for n, arr in systems.items():
        boot[n].append(prf(y_all[idx], arr[idx])["F1"])
diff_ci = {}
for n in systems:
    if n == "stage2":
        continue
    d = np.array(boot[n]) - np.array(boot["stage2"])
    lo, hi = np.percentile(d, [2.5, 97.5])
    diff_ci[n] = (prf(y_all, systems[n])["F1"] - prf(y_all, systems["stage2"])["F1"], lo, hi)

# ---------------- markdown ----------------
md = ["# Ambiguous band with free Groq models", "",
      f"Run date {TODAY}. Band = Stage 2 score in [{LO}, {HI}): {len(BAND)} pairs ({int(BAND.label.sum())} positives). Outside the band Stage 2 decides. "
      "Prompts are the fixed ones from the llama3 runs (extracted unchanged from adjudication_compare.py; each is sent as one user message). "
      "API-format additions only: temperature 0, max_completion_tokens 400, response_format json_object, reasoning_effort low (gpt-oss) / none (qwen). "
      f"Models: {', '.join(sorted(v for v in MODELS.values() if v in available))}. Status: **{status.split('|')[0].strip()}**", ""]
band_sets = {"cross_encoder": df.ce_pred.where(df.in_band)}
band_sets.update({n: s for n, s in band_dec.items()})
md += ["### Band pairs only (failed/missing LLM calls excluded; counts shown)", "",
       "| system | scope | n | precision | recall | F1 | TP | FP | FN | TN | flag rate |", "|---|---|---|---|---|---|---|---|---|---|---|"]
flags = []
for name, ser in band_sets.items():
    for scope, mask in [("pooled", df.in_band)] + [(f"fold{k}", df.in_band & (df.fold == k)) for k in sorted(df.fold.unique())]:
        m = mask & ser.notna()
        r = prf(df.label[m], ser[m])
        md.append(f"| {name} | {scope} | {r['n']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['F1']:.3f} | {r['TP']} | {r['FP']} | {r['FN']} | {r['TN']} | {r['flag_rate']:.0%} |")
        if scope != "pooled" and r["flag_rate"] > 0.6:
            flags.append(f"{name} {scope} ({r['flag_rate']:.0%})")
md += ["", "Folds where a system flags more than 60% of band pairs: " + ("; ".join(flags) if flags else "none"), ""]
md += ["### End to end (all 3570 rows; outside the band and for failed calls Stage 2 decides)", "",
       "| system | precision | recall | F1 | TP | FP | FN | F1 minus Stage 2 [95% CI] | CI excludes 0 |", "|---|---|---|---|---|---|---|---|---|"]
for name, e in rows:
    if name == "stage2":
        md.append(f"| stage2 | {e['precision']:.3f} | {e['recall']:.3f} | {e['F1']:.3f} | {e['TP']} | {e['FP']} | {e['FN']} | - | - |")
    else:
        d, lo, hi = diff_ci[name]
        md.append(f"| {name} | {e['precision']:.3f} | {e['recall']:.3f} | {e['F1']:.3f} | {e['TP']} | {e['FP']} | {e['FN']} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {lo > 0 or hi < 0} |")
md += ["", "Combination rules use saved verdicts only: AND = cross-encoder and LLM both yes; CEtie = cross-encoder decides, the LLM decides when "
       "the cross-encoder probability is within +-0.1 of its fold-chosen threshold; OR for reference. Combos start from the cross-encoder when an LLM verdict is missing.", ""]
md += ["### Agreement of each LLM with the cross-encoder (band pairs)", "",
       "| LLM | n | agree | accuracy when agree | accuracy when disagree | LLM acc on disagreements | CE acc on disagreements |", "|---|---|---|---|---|---|---|"]
for name, ser in band_dec.items():
    m = df.in_band & ser.notna()
    g, c, yy = ser[m].astype(int).values, df.ce_pred[m].values, df.label[m].values
    ag = g == c
    md.append(f"| {name} | {int(m.sum())} | {ag.mean():.1%} | {((g[ag] == yy[ag]).mean() if ag.any() else float('nan')):.3f} | "
              f"{(((g[~ag] == yy[~ag]).mean() + (c[~ag] == yy[~ag]).mean()) / 2 if (~ag).any() else float('nan')):.3f} | "
              f"{((g[~ag] == yy[~ag]).mean() if (~ag).any() else float('nan')):.3f} | {((c[~ag] == yy[~ag]).mean() if (~ag).any() else float('nan')):.3f} |")
md += ["", "### Stage 2 errors", "",
       f"- 49 FN: {int(((df.s2_pred == 0) & (df.label == 1) & df.in_band).sum())} inside the band, {int(((df.s2_pred == 0) & (df.label == 1) & ~df.in_band).sum())} outside (below {LO}).",
       f"- 30 FP: {int(((df.s2_pred == 1) & (df.label == 0) & df.in_band).sum())} inside the band (FPs score >= {S2_T}, so a band ending at {S2_T} can never contain one).", "",
       "| system | FN in band recovered (of 48) | new FP in band | combos: recovered / new FP |", "|---|---|---|---|"]
for name in ["cross_encoder"] + list(band_dec):
    arr = systems[name]
    rec_ = int(((df.label == 1) & df.in_band & (arr == 1)).sum()); newfp = int(((df.label == 0) & df.in_band & (arr == 1)).sum())
    cs = "; ".join(f"{c}: {int(((df.label == 1) & df.in_band & (systems[f'{name}+CE_{c}'] == 1)).sum())}/{int(((df.label == 0) & df.in_band & (systems[f'{name}+CE_{c}'] == 1)).sum())}"
                   for c in ("AND", "CEtie", "OR") if f"{name}+CE_{c}" in systems)
    md.append(f"| {name} | {rec_} | {newfp} | {cs} |")
md.append("")

# extra analysis: best available model, zero-shot, on all 79 Stage 2 errors
all_err = df[df.s2_pred != df.label]
ex_lines = []
for model in (MODELS["gpt120"], MODELS["qwen"]):
    v = verdicts(model, ZERO)
    if all(p in v for p in all_err.pair_id):
        vv = all_err.pair_id.map(v)
        ok = vv.notna()
        fixed_fn = int(((all_err.label == 1) & (vv == True)).sum()); fixed_fp = int(((all_err.label == 0) & (vv == False)).sum())
        ex_lines.append(f"- {model} zero-shot corrects {fixed_fn + fixed_fp} of {int(ok.sum())} judged Stage 2 errors "
                        f"(FN {fixed_fn} of {int(((all_err.label == 1) & ok).sum())}; FP {fixed_fp} of {int(((all_err.label == 0) & ok).sum())}).")
md += ["### Extra analysis (separate): zero-shot on all 79 Stage 2 errors regardless of band", ""] + (ex_lines or ["- not run yet (needs the stage 4 calls)."]) + [
    "- This is a selected sample of errors only, so it measures recoverability, not net accuracy.", ""]

# usage
md += ["### Calls, failures, tokens, cost", "", "| model | variant | cached calls | failed | tokens (all runs) | tokens/call |", "|---|---|---|---|---|---|"]
for model in MODELS.values():
    for variant in (ZERO, FEW):
        rs = [r for k, r in cache.items() if k[1] == model and k[2] == variant]
        if rs:
            ok = [r for r in rs if not r["failed"]]; tok = sum(r["usage"]["total"] for r in ok)
            md.append(f"| {model} | {variant} | {len(ok)} | {len(rs) - len(ok)} | {tok:,} | {tok / max(1, len(ok)):.0f} |")
md += ["", f"Cost: $0.00 (free tier; all calls on the on-demand free plan). Observed rate-limit headers: {headers_seen or 'none this run'}. "
       f"Throttle: <= {MAX_RPM} requests/min and < {TPM_BUDGET} tokens/min.", ""]

best_n = max(diff_ci, key=lambda n: diff_ci[n][0]) if diff_ci else None
beats = [n for n, (d, lo, hi) in diff_ci.items() if lo > 0]
md += ["## Summary", ""]
llm_names = [n for n in systems if n.split("+")[0] in used_groq]
md += [f"1. Complete Groq systems in this report: {', '.join(used_groq) or 'none yet'}; status: {status.split('|')[0].strip()}.",
       "2. Band F1: " + "; ".join(f"{n} {prf(df.label[df.in_band & ser.notna()], ser[df.in_band & ser.notna()])['F1']:.3f}" for n, ser in band_sets.items()) + ".",
       f"3. Best end-to-end F1 minus Stage 2: {best_n} {diff_ci[best_n][0]:+.3f} [{diff_ci[best_n][1]:+.3f}, {diff_ci[best_n][2]:+.3f}]." if best_n else "3. No comparison yet.",
       ("4. Beats Stage 2 with a CI excluding 0: " + ", ".join(beats) + ".") if beats else "4. No LLM or combination beats Stage 2 with a 95% CI that excludes 0.",
       f"5. Calls this run: {sum(v['calls'] for v in run_stats.values())} ok, {sum(v['failed'] for v in run_stats.values())} failed; cost $0."]
(RES / "adjudication_groq.md").write_text("\n".join(md), encoding="utf-8")

# scores and error lists (new files)
out = df[["pair_id", "video_id", "fold", "label", "in_band", "current_stage2", "ce_prob", "ce_pred"]].rename(columns={"in_band": "band_flag", "ce_pred": "ce_decision"}).copy()
for n, ser in band_dec.items():
    out[n] = ser
out.to_csv(RES / "adjudication_groq_scores.csv", index=False)
er = df[df.in_band | (df.s2_pred != df.label)][["pair_id", "video_id", "fold", "label", "in_band", "current_stage2", "text_a", "text_b"]].copy()
er["stage2_error"] = np.where((df.s2_pred != df.label)[er.index], np.where(df.label[er.index] == 1, "FN", "FP"), "")
for n in ["cross_encoder"] + list(band_dec):
    ser = df.ce_pred if n == "cross_encoder" else band_dec[n]
    er[n] = ser[er.index]
    er[n + "_outcome"] = np.select([(er.in_band) & (er.label == 1) & (er[n] == 1), (er.in_band) & (er.label == 1),
                                    (er.in_band) & (er.label == 0) & (er[n] == 1)], ["recovered_FN", "still_FN", "new_FP"], default="")
er.to_csv(RES / "adjudication_groq_errors.csv", index=False)
print("\n".join(md[-14:]))
