"""
Score one test fold of the SFT v2 keeper data against a chat model: Unsloth Studio (OpenAI-compatible POST {base-url}/chat/completions, the default) or
Gemini (--provider gemini: POST generativelanguage.googleapis.com/v1beta/models/<model>:generateContent, key from GEMINI_API_KEY). Every model gets the
identical system + user messages of the identical test rows, temperature 0.

    UNSLOTH_API_KEY=sk-unsloth-... python training/score_sft_fold.py --fold 1 --variant short
    python training/score_sft_fold.py --fold 1 --variant full --model Qwen/Qwen3-4B-Instruct-2507 --tag base_qwen
    GEMINI_API_KEY=... python training/score_sft_fold.py --fold 1 --variant full --provider gemini --model gemini-3.5-flash --tag gemini
    python training/score_sft_fold.py --fold 1 --variant full --tag base_qwen --preds-only     # rescore an existing predictions file, no API call

Sends only the system + user messages of each test row (the assistant message, i.e. the gold answer, is never sent), temperature 0, one row at a time in file
order; a failed request is retried once. The model must answer {"keep": [...], "keep_none": bool, "unclear": bool}. A request error or an unparsable /
invalid answer (not JSON, wrong keys, an id that is not one of the group's units, keep_none inconsistent with keep) counts as WRONG, never skipped.
The key is read from UNSLOTH_API_KEY only and is never printed or written. Default model: the first model /v1/models reports as loaded.
Writes fold{k}_{variant}_preds[_{tag}].jsonl next to the test file: {"group_id", "raw", "pred", "error", "model"}.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "http://127.0.0.1:8888/v1"
KEY_ENV = "UNSLOTH_API_KEY"
UNIT_IDS = re.compile(r"^Unit ids \(the only valid answers\): \[(.*)\]$", re.M)
WANT_KEYS = {"keep", "keep_none", "unclear"}


def _clean(msg: str, key: str | None) -> str:
    return str(msg).replace(key, "***") if key else str(msg)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_answer(raw: str, group_ids: list[str]) -> dict:
    """Raw model text -> {"keep": [...], "keep_none": bool, "unclear": bool}; raises ValueError if it is not a valid answer for this group."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)                         # tolerate a code fence or a leading sentence
    if not m:
        raise ValueError("no JSON object in the answer")
    ans = json.loads(m.group(0))
    if set(ans) != WANT_KEYS:
        raise ValueError(f"keys {sorted(ans)} != {sorted(WANT_KEYS)}")
    if not isinstance(ans["keep"], list) or not all(isinstance(k, str) for k in ans["keep"]) or not all(isinstance(ans[k], bool) for k in ("keep_none", "unclear")):
        raise ValueError("wrong value types")
    bad = [k for k in ans["keep"] if k not in group_ids]
    if bad:
        raise ValueError(f"ids {bad} are not in the group")
    if ans["keep_none"] != (not ans["keep"]):
        raise ValueError("keep_none is inconsistent with keep")
    return ans


def chat(session: requests.Session, url: str, model: str, messages: list[dict], headers: dict, timeout: float, max_tokens: int) -> tuple[str, str]:
    r = session.post(url, json={"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": False}, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    body = r.json()
    return str(body["choices"][0]["message"]["content"]), str(body.get("model", model))


def gemini_chat(session: requests.Session, model: str, messages: list[dict], key: str, timeout: float, max_tokens: int) -> tuple[str, str]:
    """system -> systemInstruction, user -> contents; temperature 0. The key travels in a header, never in the URL."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n".join(m["content"] for m in messages if m["role"] == "user")
    body = {"contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    r = session.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", json=body,
                     headers={"x-goog-api-key": key, "Content-Type": "application/json"}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
    cands = r.json().get("candidates") or []
    parts = (cands[0].get("content", {}).get("parts") if cands else None) or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise RuntimeError(f"empty answer (finishReason {cands[0].get('finishReason') if cands else 'none'})")
    return text, model


def evaluate(rows: list[dict], preds: list[dict]) -> dict:
    hits = {"all": 0, "none": 0, "keepers": 0}
    counts = {"none": 0, "keepers": 0}
    baseline = errors = unclear_hits = unclear_n = 0
    for row, p in zip(rows, preds):
        gold = json.loads(row["messages"][-1]["content"])
        ids = UNIT_IDS.search(row["messages"][1]["content"])
        group_ids = re.findall(r'"([^"]+)"', ids.group(1))
        kind = "none" if gold["keep_none"] else "keepers"
        counts[kind] += 1
        baseline += gold["keep"] == [group_ids[-1]]
        if p.get("error") or p.get("pred") is None:
            errors += 1
            continue
        ok = set(p["pred"]["keep"]) == set(gold["keep"])
        hits["all"] += ok
        hits[kind] += ok
        unclear_n += 1
        unclear_hits += p["pred"]["unclear"] == gold["unclear"]
    n = len(rows)
    rate = lambda a, b: None if b == 0 else a / b
    return {"n": n, "errors": errors, "exact_match": rate(hits["all"], n), "exact_match_keep_none_rows": rate(hits["none"], counts["none"]),
            "exact_match_rows_with_keepers": rate(hits["keepers"], counts["keepers"]), "n_keep_none": counts["none"], "keep_last_baseline": rate(baseline, n),
            "unclear_flag_agreement": rate(unclear_hits, unclear_n)}


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--variant", choices=["full", "short"], required=True, help="the prompt variant the model was trained on")
    ap.add_argument("--test", default=None, help="default: data/training/sft_v2/<variant>/keeper_sft_fold<k>_test.jsonl")
    ap.add_argument("--provider", choices=["studio", "gemini"], default="studio")
    ap.add_argument("--tag", default=None, help="suffix of the predictions file, e.g. base_qwen or gemini")
    ap.add_argument("--retries", type=int, default=None, help="extra attempts per row (default 1 for studio, 3 with backoff for gemini)")
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default=None, help="default: the first model /v1/models reports as loaded")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-tokens", type=int, default=None, help="default 120 for studio, 2048 for gemini (thinking tokens count)")
    ap.add_argument("--preds-only", action="store_true")
    args = ap.parse_args()

    test = Path(args.test) if args.test else REPO / "data" / "training" / "sft_v2" / args.variant / f"keeper_sft_fold{args.fold}_test.jsonl"
    rows = read_rows(test)
    out = test.parent / (f"fold{args.fold}_{args.variant}_preds" + (f"_{args.tag}" if args.tag else "") + ".jsonl")
    if args.preds_only:
        preds = read_rows(out)
        if [p["group_id"] for p in preds] != [r["meta"]["group_id"] for r in rows]:
            print("error: predictions do not match the test rows (same rows, same order required)", file=sys.stderr)
            return 2
    else:
        gem = args.provider == "gemini"
        key = os.environ.get("GEMINI_API_KEY" if gem else KEY_ENV)
        if not key:
            print(f"error: set {'GEMINI_API_KEY' if gem else KEY_ENV}", file=sys.stderr)
            return 2
        headers, base = {"Authorization": f"Bearer {key}"}, args.base_url.rstrip("/")
        session = requests.Session()
        model = args.model
        if gem and not model:
            print("error: --model is required with --provider gemini (e.g. gemini-3.5-flash)", file=sys.stderr)
            return 2
        if not model:
            items = session.get(base + "/models", headers=headers, timeout=30).json().get("data", [])
            loaded = [m["id"] for m in items if m.get("loaded")]
            if not loaded:
                print("error: /v1/models reports no loaded model; load one in Studio or pass --model", file=sys.stderr)
                return 2
            model = loaded[0]
        max_tokens = args.max_tokens or (2048 if gem else 120)
        extra = args.retries if args.retries is not None else (3 if gem else 1)
        where = "generateContent (Gemini)" if gem else f"{base}/chat/completions"
        print(f"fold {args.fold} ({args.variant}): {len(rows)} rows -> {where} (model '{model}')", flush=True)
        preds = []
        for i, row in enumerate(rows):
            group_ids = re.findall(r'"([^"]+)"', UNIT_IDS.search(row["messages"][1]["content"]).group(1))
            messages = row["messages"][:2]                              # system + user only: the gold answer is never sent
            rec = {"group_id": row["meta"]["group_id"], "raw": None, "pred": None, "error": None, "model": None}
            for attempt in range(extra + 1):
                try:
                    if gem:
                        rec["raw"], rec["model"] = gemini_chat(session, model, messages, key, args.timeout, max_tokens)
                    else:
                        rec["raw"], rec["model"] = chat(session, base + "/chat/completions", model, messages, headers, args.timeout, max_tokens)
                    rec["pred"], rec["error"] = parse_answer(rec["raw"], group_ids), None
                    break
                except Exception as exc:
                    rec["error"] = _clean(f"{type(exc).__name__}: {exc}", key)
                    if attempt < extra:
                        time.sleep(2.0 * (attempt + 1) if gem else 1.0)
            preds.append(rec)
            if (i + 1) % 10 == 0 or i + 1 == len(rows):
                print(f"  {i + 1}/{len(rows)} rows sent, {sum(1 for p in preds if p['error'])} errors", flush=True)
        out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8", newline="\n")
        print(f"wrote {out}")
        print(f"answered by model(s): {sorted({p['model'] for p in preds if p['model']})}")

    m = evaluate(rows, preds)
    diff = None if m["exact_match"] is None or m["keep_last_baseline"] is None else m["exact_match"] - m["keep_last_baseline"]
    print(f"\nfold {args.fold} ({args.variant})  n={m['n']}  errors/invalid answers={m['errors']} (counted as wrong)")
    print(f"exact-match overall            : {fmt(m['exact_match'])}")
    print(f"exact-match keep_none rows only: {fmt(m['exact_match_keep_none_rows'])}   (n_keep_none = {m['n_keep_none']})")
    print(f"exact-match rows with keepers  : {fmt(m['exact_match_rows_with_keepers'])}")
    print(f"keep-last baseline             : {fmt(m['keep_last_baseline'])}   difference: {'n/a' if diff is None else f'{diff * 100:+.1f} points'}")
    print(f"unclear flag agreement (valid answers): {fmt(m['unclear_flag_agreement'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
