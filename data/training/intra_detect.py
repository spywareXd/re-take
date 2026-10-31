"""Part A: rule-based detection of retakes INSIDE a single ComparableUnit (no training, no API calls).

Inputs (read only): processed/<video>/units.json (the per-video comparable units; the task called it
comparable_units.json) and processed/<video>/transcript.json (word timestamps + confidence), pairs_v1.csv,
folds.json. Writes only to results/intra/. Rules R1-R4 flag a unit; R0 (adjacent stutter) is reported
separately and does not count as flagged. Fixed seeds throughout.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
OUT = HERE / "results" / "intra"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
MIN_TOKENS = 6
GAP_SHORT = 0.35            # R2: adjacent repeat counts if the silence between them is shorter than this
LOW_CONF = 0.5
RNG = np.random.default_rng(SEED)

# ---- stoplists / cue lists ----
DOUBLINGS = {("very", "very"), ("no", "no"), ("bye", "bye"), ("go", "go")}   # common intentional doublings
NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "twenty", "thirty", "forty", "fifty", "hundred", "thousand", "million"}
FUNCTION_WORDS = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "is", "are", "was",
                  "were", "be", "it", "its", "i", "you", "he", "she", "we", "they", "that", "this", "so", "if", "as",
                  "do", "have", "my", "your", "me", "um", "uh", "like"}
CUES = [("sorry",), ("again",), ("let", "me"), ("i", "mean"), ("no", "wait"), ("okay", "so")]


def is_number(t):
    return t.isdigit() or t in NUMBER_WORDS


def has_content(gram):
    """An n-gram counts only if it has a non-function, non-number word (avoids 'and the ... and the')."""
    return any(t not in FUNCTION_WORDS and not is_number(t) for t in gram)


# ---------------- detector ----------------
def detect(words):
    """words: list of dicts {tok, raw, start, end, conf}. Returns (fired_rules, split) or ([], None).
    split = (discard_start_idx, split_idx) of the first rule that fires (priority R1, R2, R3, R4)."""
    toks = [w["tok"] for w in words]
    n = len(toks)
    fired, splits = [], {}

    # R1 prefix restart: first k (2..5) tokens reappear later (largest k wins), not a stutter of one repeated word
    for k in range(min(5, n // 2), 1, -1):
        pre = toks[:k]
        if len(set(pre)) == 1 or tuple(pre) in DOUBLINGS or not has_content(pre):
            continue
        j = next((j for j in range(k, n - k + 1) if toks[j:j + k] == pre), None)
        if j is not None:
            fired.append("R1"); splits["R1"] = (0, j)
            break

    # R2 repeated n-gram (n >= 3), either a token between the occurrences or a short silence if adjacent
    done = False
    for g in range(min(8, n // 2), 2, -1):
        for i in range(0, n - g + 1):
            gram = toks[i:i + g]
            if not has_content(gram):
                continue
            for j in range(i + g, n - g + 1):
                if toks[j:j + g] != gram:
                    continue
                gap = words[j]["start"] - words[i + g - 1]["end"]
                if j > i + g or gap < GAP_SHORT:
                    fired.append("R2"); splits["R2"] = (i, j); done = True
                    break
            if done:
                break
        if done:
            break

    # R3 restart cue followed by a repeat of earlier tokens (the 2 tokens after the cue occur before the cue)
    found = False
    for cue in CUES:
        c = len(cue)
        for p in range(0, n - c + 1):
            if tuple(toks[p:p + c]) != cue:
                continue
            after = toks[p + c:p + c + 2]
            if len(after) < 2 or not has_content(after):
                continue
            i = next((i for i in range(0, p - 1) if toks[i:i + 2] == after and i + 2 <= p), None)
            if i is not None:
                fired.append("R3"); splits["R3"] = (i, p + c); found = True
                break
        if found:
            break

    # R4 >= 2 consecutive low-confidence words followed by a repeat of earlier tokens
    q = 0
    found = False
    while q < n and not found:
        if words[q]["conf"] < LOW_CONF:
            r = q
            while r < n and words[r]["conf"] < LOW_CONF:
                r += 1
            if r - q >= 2:
                after = toks[r:r + 2]
                if len(after) == 2 and has_content(after):
                    i = next((i for i in range(0, q - 1) if toks[i:i + 2] == after and i + 2 <= q), None)
                    if i is not None:
                        fired.append("R4"); splits["R4"] = (i, r); found = True
            q = r
        else:
            q += 1
    split = None
    for rule in ("R1", "R2", "R3", "R4"):
        if rule in splits:
            split = splits[rule]
            break
    return fired, split


def stutter_r0(toks):
    """R0: adjacent duplicate tokens (reported separately), minus numbers and intentional doublings."""
    return any(toks[i] == toks[i + 1] and (toks[i], toks[i + 1]) not in DOUBLINGS and not is_number(toks[i])
               for i in range(len(toks) - 1))


# ---------------- load words per unit ----------------
def clean(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


units_by_video, rows, missing = {}, [], []
for vdir in sorted((HERE / "processed").iterdir()):
    if not (vdir / "units.json").exists():
        continue
    v = vdir.name
    units = json.loads((vdir / "units.json").read_text())
    segs = {s["id"]: s for s in json.loads((vdir / "transcript.json").read_text())["segments"]}
    units_by_video[v] = units
    for u in units:
        seg = segs.get(u["source_segment_id"])
        ws = [] if not seg or not seg.get("words") else [
            {"tok": clean(w["word"]), "raw": w["word"], "start": w["start"], "end": w["end"], "conf": w.get("confidence", 1.0)}
            for w in seg["words"] if w["start"] >= u["start"] - 0.03 and w["end"] <= u["end"] + 0.03]
        ws = [w for w in ws if w["tok"]]
        if not ws:
            missing.append((v, u["unit_id"]))
        rows.append({"video_id": v, "unit_id": u["unit_id"], "unit": u, "words": ws})
print(f"{len(rows)} units in {len(units_by_video)} videos; units without word timestamps: {len(missing)}")

# ---------------- run on real units ----------------
recs = []
for r in rows:
    ws = r["words"]
    if len(ws) < MIN_TOKENS:
        continue
    fired, split = detect(ws)
    rec = {"video_id": r["video_id"], "unit_id": r["unit_id"], "n_tokens": len(ws), "R0": stutter_r0([w["tok"] for w in ws]),
           "rules": fired, "flagged": bool(fired), "text": r["unit"]["raw_text"], "start": r["unit"]["start"], "end": r["unit"]["end"]}
    if split:
        i, j = split
        rec.update({
            "split": split,
            "marked_text": " ".join(w["raw"] if x != j else "|| " + w["raw"] for x, w in enumerate(ws)),
            "discard_start": ws[i]["start"], "discard_end": ws[j]["start"],
            "discard_text": " ".join(w["raw"] for w in ws[i:j]), "kept_text": " ".join(w["raw"] for w in ws[j:]),
            "pause_at_split": round(ws[j]["start"] - ws[j - 1]["end"], 3) if j > 0 else np.nan,
            "mean_conf_discard": float(np.mean([w["conf"] for w in ws[i:j]])) if j > i else np.nan})
    recs.append(rec)
det = pd.DataFrame(recs)
flag = det[det.flagged].copy()
print(f"eligible units (>= {MIN_TOKENS} tokens): {len(det)}; flagged R1-R4: {len(flag)}; R0-only/any R0: {int(det.R0.sum())}")

# ---------------- A2 review sheets ----------------
if len(flag) <= 150:
    chosen = flag
else:
    # stratified, seeded: round-robin over (primary rule, video) cells until 150
    flag["primary"] = flag.rules.map(lambda x: x[0])
    cells = {k: g.sample(frac=1, random_state=SEED) for k, g in flag.groupby(["primary", "video_id"])}
    keys = sorted(cells)
    picked, level = [], 0
    while len(picked) < 150 and any(len(cells[k]) > level for k in keys):
        for k in keys:
            if len(cells[k]) > level and len(picked) < 150:
                picked.append(cells[k].iloc[level])
        level += 1
    chosen = pd.DataFrame(picked)
pool = det[~det.flagged & ~det.R0]
unflagged = pool.sample(n=min(40, len(pool)), random_state=SEED)
sheet = pd.concat([chosen.assign(flagged=1), unflagged.assign(flagged=0)], ignore_index=True)
sheet = sheet.sample(frac=1, random_state=SEED).reset_index(drop=True)
sheet["row_id"] = range(1, len(sheet) + 1)
sheet["rule_fired"] = sheet.rules.map(lambda x: "+".join(x) if isinstance(x, list) else "")
sheet["text_marked"] = sheet.marked_text.where(sheet.marked_text.notna(), sheet.text)
review = sheet.rename(columns={"text_marked": "text_with_split", "discard_start": "proposed_discard_start",
                               "discard_end": "proposed_discard_end", "pause_at_split": "pause_at_split_seconds"})
review["verdict"], review["notes"] = "", ""
cols = ["row_id", "video_id", "unit_id", "flagged", "rule_fired", "text_with_split", "proposed_discard_start",
        "proposed_discard_end", "discard_text", "kept_text", "pause_at_split_seconds", "mean_conf_discard", "verdict", "notes"]
review[cols].to_csv(OUT / "intra_review.csv", index=False)
# blind sheet: no flag, no rule, no proposed split (blank proposal fields would reveal the flag)
blind = sheet[["row_id", "video_id", "unit_id", "text", "start", "end"]].copy()
blind["verdict"], blind["notes"] = "", ""
blind.to_csv(OUT / "intra_review_blind.csv", index=False)

# ---------------- A3 synthetic recall (text only, is_synthetic) ----------------
cand = det[~det.flagged & ~det.R0 & (det.n_tokens >= 8)]
base_units = cand.sample(n=min(200, len(cand)), random_state=SEED)
tok_by_unit = {(r["video_id"], r["unit_id"]): r["words"] for r in rows}


def mk(tokens):
    """Text-only synthetic words: contiguous 0.3 s per word, confidence 0.9 (no pause, no low-confidence cues)."""
    return [{"tok": t, "raw": t, "start": 0.3 * i, "end": 0.3 * i + 0.3, "conf": 0.9} for i, t in enumerate(tokens)]


syn = []
for r in base_units.itertuples():
    t = [w["tok"] for w in tok_by_unit[(r.video_id, r.unit_id)]]
    k = int(RNG.integers(3, 6))
    cue = [["i", "mean"], ["sorry"], []][int(RNG.integers(0, 3))]
    syn.append(("a_false_start", t[:k] + cue + t))
    syn.append(("b_mid_restart", t[:len(t) // 2] + t[:3] + t))
    syn.append(("c_control", t))
synres = []
for kind, toks in syn:
    fired, _ = detect(mk(toks))
    synres.append({"kind": kind, "any": bool(fired), **{rl: rl in fired for rl in ("R1", "R2", "R3", "R4")}})
syn_df = pd.DataFrame(synres)
syn_tab = syn_df.groupby("kind")[["any", "R1", "R2", "R3", "R4"]].mean().round(3)
syn_n = syn_df.groupby("kind").size()

# ---------------- A4 impact simulation ----------------
from sentence_transformers import SentenceTransformer  # noqa: E402
from pipeline.step_03_token_matching.config import LEXICAL_WEIGHTS, TokenMatchingConfig  # noqa: E402
from pipeline.step_03_token_matching.edit_distance import containment_score, sequence_similarity_score  # noqa: E402
from pipeline.step_03_token_matching.exact_match import exact_match_score, near_exact_match_score  # noqa: E402
from pipeline.step_03_token_matching.jaccard import jaccard_score  # noqa: E402
from pipeline.step_03_token_matching.ngram_match import ngram_overlap_score  # noqa: E402
from pipeline.step_02_preprocessing.normalize import tokenize  # noqa: E402
from pipeline.step_02_preprocessing.schemas import ComparableUnit  # noqa: E402
from pipeline.step_04_semantic_matching.config import SEMANTIC_WEIGHTS  # noqa: E402
from pipeline.step_04_semantic_matching.semantic_matching import get_model, semantic_similarity_score  # noqa: E402

fillers = set(TokenMatchingConfig().filler_words)


def cu(uid, text):
    t = tokenize(text)
    return ComparableUnit(unit_id=uid, source_segment_id=0, start=0.0, end=0.0, raw_text=text,
                          normalized_text=" ".join(t), tokens=t, token_set=set(t))


model = get_model()
sim = []
for r in flag.itertuples():
    a, b = cu("d", r.discard_text), cu("k", r.kept_text)
    ea, eb = model.encode([r.discard_text, r.kept_text], show_progress_bar=False)
    sem = semantic_similarity_score(ea, eb)
    seq, jac, con, big = sequence_similarity_score(a, b), jaccard_score(a, b), containment_score(a, b), ngram_overlap_score(a, b, 2)
    lex = max(LEXICAL_WEIGHTS["sequence"] * seq + LEXICAL_WEIGHTS["jaccard"] * jac + LEXICAL_WEIGHTS["containment"] * con
              + LEXICAL_WEIGHTS["bigram"] * big, exact_match_score(a, b), near_exact_match_score(a, b, fillers))
    s2 = (SEMANTIC_WEIGHTS["semantic_similarity"] * sem + SEMANTIC_WEIGHTS["lexical_combined_score"] * lex
          + SEMANTIC_WEIGHTS["containment_bonus"] * con)
    sim.append({"video_id": r.video_id, "unit_id": r.unit_id, "lexical": lex, "containment": con, "semantic": sem, "stage2": s2,
                "discard_tokens": len(a.tokens), "kept_tokens": len(b.tokens)})
sim = pd.DataFrame(sim)
sim.to_csv(OUT / "intra_split_pair_scores.csv", index=False)

# which flagged units does the current pipeline already see? (pairs_v1 real rows, any label)
pv = pd.read_csv(HERE / "pairs_v1.csv", dtype=str, keep_default_na=False)
pv = pv[pv.is_synthetic == "0"]
unit_in = {}
for p in pv.itertuples():
    for u in p.pair_id.split("__")[1:3]:
        uid = u.replace("-", ":")
        e = unit_in.setdefault((p.video_id, uid), {"pairs": 0, "pos": 0})
        e["pairs"] += 1
        e["pos"] += int(p.label == "1")
flag["in_any_pair"] = [(v, u) in unit_in for v, u in zip(flag.video_id, flag.unit_id)]
flag["in_positive_pair"] = [unit_in.get((v, u), {}).get("pos", 0) > 0 for v, u in zip(flag.video_id, flag.unit_id)]

# ---------------- report ----------------
rule_counts = Counter(rl for rs in det.rules for rl in rs)
md = ["# Intra-unit retake detector (Part A)", "",
      "Rules only, no training. Units come from processed/<video>/units.json with word timestamps from transcript.json "
      f"(all 15 videos have word timestamps; {len(missing)} units had no matching words). Eligible = units with >= {MIN_TOKENS} word tokens.",
      "Design choices not fixed by the task: tokens are lowercase words without punctuation (contractions kept whole); an "
      "n-gram/prefix must contain at least one non-function, non-number word; R3/R4 'repeat' = the 2 tokens after the cue/cluster occur "
      "earlier in the unit; R0 stutters are reported separately and do not count as flagged.", "",
      "## A1/A5 rates", "",
      f"- Eligible units: {len(det)}; flagged (R1-R4): {len(flag)} ({100 * len(flag) / len(det):.1f}% of eligible); R0 stutters: {int(det.R0.sum())}.",
      "- Rule counts (a unit can fire several): " + ", ".join(f"{k}={rule_counts.get(k, 0)}" for k in ("R1", "R2", "R3", "R4")), "",
      "| video | units | eligible | flagged | flagged per 100 units | R1 | R2 | R3 | R4 | R0 |", "|---|---|---|---|---|---|---|---|---|---|"]
for v, us in units_by_video.items():
    d = det[det.video_id == v]
    rc = Counter(rl for rs in d.rules for rl in rs)
    md.append(f"| {v} | {len(us)} | {len(d)} | {int(d.flagged.sum())} | {100 * d.flagged.sum() / len(us):.1f} | "
              f"{rc['R1']} | {rc['R2']} | {rc['R3']} | {rc['R4']} | {int(d.R0.sum())} |")
tot_units = sum(len(u) for u in units_by_video.values())
md += [f"| ALL | {tot_units} | {len(det)} | {len(flag)} | {100 * len(flag) / tot_units:.1f} | "
       + " | ".join(str(rule_counts.get(k, 0)) for k in ("R1", "R2", "R3", "R4")) + f" | {int(det.R0.sum())} |", "",
       "## A2 review sheets", "",
       f"- intra_review.csv: {len(chosen)} flagged ({'all' if len(flag) <= 150 else 'stratified sample of ' + str(len(flag))}) + {len(unflagged)} unflagged, shuffled (seed {SEED}); "
       "intra_review_blind.csv has the same rows without flag, rule or proposed split, same row_id.", "",
       "## A3 Synthetic recall test (SYNTHETIC, detector recall only; not a claim about real data)", "",
       f"Built from {len(base_units)} unflagged real units with >= 8 tokens; text-only corruptions with contiguous word timing and "
       "confidence 0.9 (so R4 cannot fire and R2's short-gap condition is generous). Corruptions are constructed to contain exactly "
       "the repeats R1/R2 look for, so recall here is an optimistic upper bound.", "",
       "| kind (is_synthetic=1) | n | any rule | R1 | R2 | R3 | R4 |", "|---|---|---|---|---|---|---|"]
for kind, r in syn_tab.iterrows():
    md.append(f"| {kind} | {syn_n[kind]} | {r['any']:.3f} | {r.R1:.3f} | {r.R2:.3f} | {r.R3:.3f} | {r.R4:.3f} |")
md += ["", f"False-positive rate on the clean control: {syn_tab.loc['c_control', 'any']:.3f}.", "",
       "## A4 Impact simulation (real flagged units; discard part vs kept part scored with the existing features)", "",
       f"- Stage 2 score of the (discard, kept) pairs: mean {sim.stage2.mean():.3f}, median {sim.stage2.median():.3f}, "
       f"quartiles {sim.stage2.quantile(.25):.3f} / {sim.stage2.quantile(.75):.3f}, min {sim.stage2.min():.3f}, max {sim.stage2.max():.3f}.",
       f"- Would reach >= 0.65: {int((sim.stage2 >= 0.65).sum())} of {len(sim)}; in [0.45, 0.65): {int(((sim.stage2 >= 0.45) & (sim.stage2 < 0.65)).sum())}; below 0.45: {int((sim.stage2 < 0.45).sum())}.",
       f"- Flagged units already in a labelled-positive pair in pairs_v1.csv: {int(flag.in_positive_pair.sum())}; in some pair (any label): "
       f"{int(flag.in_any_pair.sum())}; in no candidate pair at all (invisible to the current pipeline): {int((~flag.in_any_pair).sum())} of {len(flag)}.",
       "- Note: pairs_v1.csv holds the exported candidate pairs only; units absent from it may still appear in unexported candidates.json pairs.", "",
       "Verdicts come only from your manual review of intra_review.csv; no real-data precision or recall is claimed here.", ""]
(OUT / "intra_detector_report.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
