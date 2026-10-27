"""Build groups_v2: natural groups + the original 60 synthetic groups + ~1,140 new synthetic groups.

Keeper labels come from ONE rule function (rule_keep) that encodes the same rules used for the natural labels:
  1 truncated / stumbling takes lose to a complete take
  2 a take with a mistake loses to the take that corrects it, and a later take with a mistake loses to the earlier correct take
  3 among complete takes the later one wins (identical text -> last)
  4 a later take abandoned mid-sentence loses to the earlier complete take
  5 a sentence split over two units by a pause keeps both units (continuation)
  7 production cue phrases ("take two", "here we go") are not content -> keep_none
Usage: python make_synth_v2.py BASE_WITH_SYNTH.csv OUT_PREFIX
"""
import csv, random, re, sys, collections, json
from openpyxl import Workbook
from openpyxl.styles import Alignment
from bank import CORES, TEMPLATES, MARKERS, LEADS, IDENTICAL_SHORT, CUES, PARAPHRASES, FACTS

R = random.Random(20261026)
INT_COLS = {"n_units", "n_pos_pairs", "n_exempt_pairs"}

TARGET = {"truncated->complete": 265, "stutter->clean": 142, "identical": 118, "paraphrase": 120,
          "mistake->corrected": 96, "continuation": 96, "cue_phrases": 47, "last_take_truncated": 96,
          "later_take_mistake": 60, "later_take_stumble": 50, "truncated->continuation": 50}

# ---------------- text helpers ----------------
def nw(s): return re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("’", "'")).split()
def key(s): return " ".join(nw(s))
def words(s): return s.rstrip(".?!").split()
KEEP_CAP = {"I", "I'm", "I'll", "I'd", "I've"}
def lower_first(s):
    first = s.split(" ", 1)[0]
    return s if first in KEEP_CAP else s[0].lower() + s[1:]

def trunc(s, frac=None):
    w = words(s); frac = frac or R.uniform(0.35, 0.7)
    k = max(3, min(len(w) - 3, round(frac * len(w))))
    return " ".join(w[:k]) + R.choice([",", "...", "", ","])
def head_stutter(s):
    w = words(s); k = min(R.choice([2, 3, 3, 4]), len(w) - 4); h = " ".join(w[:k])
    return f"{h}, {h[0].lower() + h[1:]} {' '.join(w[k:])}."
def mid_stutter(s):
    w = words(s); i = len(w) // 2; pre = w[:i]
    return " ".join(pre) + ", " + " ".join(pre[-2:]) + " " + " ".join(w[i:]) + "."
def filler(s):
    w = words(s); k = max(3, len(w) // 2)
    return f"{' '.join(w[:k])}, {R.choice(['uh', 'um', 'uh, like'])}, {' '.join(w[k:])}."
def dup_word(s):
    w = words(s); i = R.randrange(2, len(w) - 2); w.insert(i, w[i - 1]); return " ".join(w) + "."
def stumble(s):
    f = R.choice(["head", "mid", "filler", "dup", "head+filler"])
    if f == "head": return head_stutter(s)
    if f == "mid": return mid_stutter(s)
    if f == "filler": return filler(s)
    if f == "dup": return dup_word(s)
    return filler(head_stutter(s))
def has_stumble_marker(t):
    w = nw(t)
    if "uh" in w or "um" in w: return True
    for L in range(1, 9):
        for i in range(len(w) - 2 * L + 1):
            if w[i:i + L] == w[i + L:i + 2 * L]: return True
    return False

# ---------------- natural / old-synthetic reference texts ----------------
def unit_texts(row):
    return [re.sub(r"^\[\d+\]\s*", "", x) for x in row["texts"].split(" || ")]

def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))

class Ref:
    def __init__(self, rows):
        self.keys, self.sets = set(), []
        for r in rows:
            for t in unit_texts(r):
                self.keys.add(key(t)); self.sets.append(set(nw(t)))
    def clash(self, s):
        if key(s) in self.keys: return True
        ts = set(nw(s))
        return any(len(ts & o) / max(1, len(ts | o)) > 0.85 for o in self.sets)

# ---------------- base sentence supply ----------------
ALL_CORES = [s for g in CORES.values() for s in g]
TMPL_SENTS = []
for tmpl, tuples in TEMPLATES:
    for tp in tuples:
        names = "abc"[:len(tp)]
        TMPL_SENTS.append(tmpl.format(**dict(zip(names, tp))))
SEEN = set()

def make_base(ref, min_words=8):
    for _ in range(400):
        r = R.random()
        if r < 0.28:
            g = R.choice(list(CORES)); a, b = R.sample(CORES[g], 2)
            s = a[:-1] + ", and " + lower_first(b)
        elif r < 0.55: s = R.choice(TMPL_SENTS)
        else: s = R.choice(ALL_CORES)
        if R.random() < 0.22:
            s = R.choice(MARKERS) + (", " if R.random() < 0.4 else " ") + lower_first(s)
        if len(s.split()) < min_words: continue
        if key(s) in SEEN or ref.clash(s): continue
        SEEN.add(key(s)); return s
    raise RuntimeError("base supply exhausted")

def T(parts, defect): return {"parts": parts, "defect": defect}

# ---------------- rule function (single source of truth for the labels) ----------------
def rule_keep(takes, cue=False):
    """Return indices of the kept takes under the labelling rules; [] means keep_none."""
    if cue: return []
    for i in range(len(takes) - 1, -1, -1):
        if not takes[i]["defect"]: return [i]
    raise ValueError("no clean take")

# ---------------- group specs ----------------
def specs(ref):
    S = []
    def add(typ, takes, note, cue=False): S.append(dict(type=typ, takes=takes, note=note, cue=cue))
    used_para, used_fact = set(), set()

    for _ in range(TARGET["truncated->complete"]):
        b = make_base(ref, 9); nt = R.choices([1, 2, 3, 4, 5], [52, 25, 13, 6, 4])[0]
        add("truncated->complete", [T([trunc(b)], True) for _ in range(nt)] + [T([b], False)],
            "SYNTHETIC truncated->complete; chose the complete take")
    for _ in range(TARGET["stutter->clean"]):
        b = make_base(ref, 9); ns = R.choices([1, 2, 3, 4], [58, 26, 11, 5])[0]
        add("stutter->clean", [T([stumble(b)], True) for _ in range(ns)] + [T([b], False)],
            "SYNTHETIC stutter->clean; chose the clean take")
    for _ in range(TARGET["identical"]):
        txt = R.choice(IDENTICAL_SHORT) if R.random() < 0.45 else make_base(ref, 5)
        k = R.choices([2, 3, 4, 5, 6, 7, 8], [44, 26, 13, 7, 4, 3, 3])[0]
        add("identical", [T([txt], False) for _ in range(k)], "SYNTHETIC identical text; chose last")
    # paraphrase
    order = []
    while len(order) < TARGET["paraphrase"]:
        o = list(range(len(PARAPHRASES))); R.shuffle(o); order += o
    for i in order[:TARGET["paraphrase"]]:
        for _ in range(50):
            v = PARAPHRASES[i]
            idx = R.sample(range(len(v)), 3 if (len(v) == 3 and R.random() < 0.45) else 2)
            texts = [v[j] for j in idx]
            texts = [(R.choice(MARKERS) + " " + lower_first(t)) if R.random() < 0.25 else t for t in texts]
            sig = (i, tuple(texts))
            if sig not in used_para: used_para.add(sig); break
        pre = []
        if R.random() < 0.2:
            pre = [T([trunc(texts[0])], True) for _ in range(1 if R.random() < 0.75 else 2)]
        add("paraphrase", pre + [T([t], False) for t in texts], "SYNTHETIC paraphrase restart; keeper unclear; chose the later take")
    # facts
    def fact_stream(n):
        out = []
        while len(out) < n:
            o = list(range(len(FACTS))); R.shuffle(o); out += o
        return out[:n]
    for i in fact_stream(TARGET["mistake->corrected"]):
        right, wrongs = FACTS[i]
        w = R.choices([[R.choice(wrongs)], list(wrongs), [wrongs[0], wrongs[1], wrongs[0]]], [50, 30, 20])[0]
        add("mistake->corrected", [T([x], True) for x in w] + [T([right], False)],
            "SYNTHETIC earlier take contains a factual error; later take is correct")
    for i in fact_stream(TARGET["later_take_mistake"]):
        right, wrongs = FACTS[i]
        w = R.choices([[R.choice(wrongs)], list(wrongs), [wrongs[0], wrongs[1], wrongs[0]]], [60, 25, 15])[0]
        add("later_take_mistake", [T([right], False)] + [T([x], True) for x in w],
            "SYNTHETIC later take contains a factual error; chose the earlier correct take")
    for _ in range(TARGET["continuation"]):
        b = make_base(ref, 8); lead = R.choice(LEADS)
        pre = [T([trunc(b)], True)] if R.random() < 0.25 else []
        add("continuation", pre + [T([b], False), T([lead, lower_first(b)], False)],
            "SYNTHETIC continuation: last two units form one sentence (pause); keep both")
    for _ in range(TARGET["truncated->continuation"]):
        b = make_base(ref, 9); lead = R.choice(LEADS); nt = R.choices([1, 2, 3], [60, 28, 12])[0]
        full = lead + " " + lower_first(b)
        add("truncated->continuation", [T([trunc(full)], True) for _ in range(nt)] + [T([lead, lower_first(b)], False)],
            "SYNTHETIC truncated takes, then a complete take split over two units by a pause; keep both parts")
    for _ in range(TARGET["cue_phrases"]):
        k = R.choices([2, 3, 4, 5, 6], [40, 30, 15, 10, 5])[0]
        add("cue_phrases", [T([c], False) for c in R.sample(CUES, k)], "SYNTHETIC cue phrases, not script content", cue=True)
    for _ in range(TARGET["last_take_truncated"]):
        b = make_base(ref, 9)
        shape = R.choices(["2", "2b", "3", "4", "5"], [40, 15, 25, 12, 8])[0]
        tr = lambda: T([trunc(b, R.uniform(0.35, 0.6))], True)
        if shape == "2": tk = [T([b], False), tr()]
        elif shape == "2b": tk = [T([b], False), tr(), tr()]
        elif shape == "3": tk = [tr(), T([b], False), tr()]
        elif shape == "4": tk = [tr(), tr(), T([b], False), tr()]
        else: tk = [tr(), tr(), T([b], False), tr(), tr()]
        note = ("SYNTHETIC last take abandoned mid-sentence; kept the earlier complete take" if shape in ("2", "2b")
                else "SYNTHETIC middle take is the only complete one; keeper unclear")
        add("last_take_truncated", tk, note)
    for _ in range(TARGET["later_take_stumble"]):
        b = make_base(ref, 9); ns = R.choices([1, 2, 3, 4], [55, 25, 12, 8])[0]
        add("later_take_stumble", [T([b], False)] + [T([stumble(b)], True) for _ in range(ns)],
            "SYNTHETIC later take stumbles; chose the earlier clean take")
    return S

# ---------------- row construction ----------------
def dur(text): return len(text.split()) / 2.7 + R.uniform(0.25, 0.7)

def make_row(vid, gi, spec, n, t):
    takes = spec["takes"]; keep = rule_keep(takes, spec["cue"])
    ids, texts, starts, ends, take_of = [], [], [], [], []
    prev_single = None
    for ti, tk in enumerate(takes):
        parts = tk["parts"]
        if ti > 0 and prev_single is not None and len(parts) == 1 and R.random() < 0.12:
            t += R.uniform(0.1, 0.4)                                  # two takes inside one ASR segment, like "2:0 | 2:1"
            ids.append(f"{prev_single}:1"); texts.append(parts[0]); take_of.append(ti)
            starts.append(round(t, 2)); t += dur(parts[0]); ends.append(round(t, 2))
            prev_single = None
        else:
            if ti > 0 and R.random() < 0.3: n += 1                    # skipped segment number (speech outside the group)
            for pj, txt in enumerate(parts):
                if pj > 0: t += R.uniform(0.4, 1.0)
                ids.append(f"{n}:{pj}"); texts.append(txt); take_of.append(ti)
                starts.append(round(t, 2)); t += dur(txt); ends.append(round(t, 2))
            prev_single = n if len(parts) == 1 else None
            n += 1
        t += R.uniform(0.25, 0.9) if R.random() < 0.65 else R.uniform(1.6, 6.0)
    pairs = [f"{vid}__{ids[i].replace(':', '-')}__{ids[j].replace(':', '-')}"
             for i in range(len(ids)) for j in range(i + 1, len(ids)) if take_of[i] != take_of[j]]
    keep_ids = [ids[i] for i in range(len(ids)) if take_of[i] in keep]
    row = dict(group_id=f"{vid}__g{gi}", video_id=vid, n_units=len(ids), unit_ids="|".join(ids),
               texts=" || ".join(f"[{k + 1}] {x}" for k, x in enumerate(texts)),
               starts="|".join(map(str, starts)), ends="|".join(map(str, ends)),
               keeper_unit_ids="|".join(keep_ids), keep_none="1" if spec["cue"] else "", notes=spec["note"],
               pipeline_keeper_unit_id="", heuristic_later=ids[-1], heuristic_earlier=ids[0], heuristic_confidence="",
               pair_ids="|".join(pairs), n_pos_pairs=len(pairs), n_exempt_pairs=0,
               feature_version="v1-synthetic", source="synthetic", synth_type=spec["type"])
    return row, n, t

def build(ref, first_video):
    S = specs(ref); R.shuffle(S)
    rows, k, v = [], 0, first_video
    while k < len(S):
        vid = f"synth_{v:03d}"; v += 1
        n, t = R.randint(1, 3), R.uniform(1, 6)
        for gi in range(1, R.choice([3, 4, 5]) + 1):
            if k >= len(S): break
            row, n, t = make_row(vid, gi, S[k], n, t); k += 1
            rows.append(row); n += R.randint(1, 3); t += R.uniform(4, 12)
    return rows

# ---------------- independent validation from the written rows (no generator flags) ----------------
RIGHT = {key(f[0]): f for f in FACTS}
WRONG = {key(w): f for f in FACTS for w in f[1]}
CUEK = {key(c) for c in CUES}

def validate(rows):
    seen_ids = set(); errs = []
    for r in rows:
        gid = r["group_id"]
        if gid in seen_ids: errs.append(f"{gid}: duplicate group id")
        seen_ids.add(gid)
        ids = r["unit_ids"].split("|"); tx = unit_texts(r); keep = [x for x in r["keeper_unit_ids"].split("|") if x]
        st = [float(x) for x in r["starts"].split("|")]
        if len(ids) != len(set(ids)) or len(ids) != len(tx) or len(ids) != int(r["n_units"]): errs.append(f"{gid}: id/text count")
        if st != sorted(st): errs.append(f"{gid}: starts not increasing")
        if not set(keep) <= set(ids): errs.append(f"{gid}: keeper not in units")
        if bool(r["keep_none"]) != (not keep): errs.append(f"{gid}: keep_none mismatch")
        t = r["synth_type"]; W = [nw(x) for x in tx]; K = [key(x) for x in tx]
        pos = {u: i for i, u in enumerate(ids)}; kp = sorted(pos[u] for u in keep)
        last_two = [len(ids) - 2, len(ids) - 1]
        def pref(a, b): return len(a) < len(b) and b[:len(a)] == a
        ok = True
        if t == "truncated->complete": ok = kp == [len(ids) - 1] and all(pref(w, W[-1]) for w in W[:-1])
        elif t == "stutter->clean": ok = kp == [len(ids) - 1] and all(has_stumble_marker(x) for x in tx[:-1]) and not has_stumble_marker(tx[-1])
        elif t == "identical": ok = kp == [len(ids) - 1] and len(set(K)) == 1
        elif t == "paraphrase": ok = kp == [len(ids) - 1] and len(set(K)) >= 2 and all(K[i] != K[-1] for i in range(len(K) - 1))
        elif t == "mistake->corrected": ok = kp == [len(ids) - 1] and K[-1] in RIGHT and all(k in WRONG and WRONG[k] is RIGHT[K[-1]] for k in K[:-1])
        elif t == "later_take_mistake": ok = kp == [0] and K[0] in RIGHT and all(k in WRONG and WRONG[k] is RIGHT[K[0]] for k in K[1:])
        elif t == "later_take_stumble": ok = kp == [0] and not has_stumble_marker(tx[0]) and all(has_stumble_marker(x) for x in tx[1:])
        elif t == "continuation": ok = len(ids) in (3, 4) and kp == last_two and not any(has_stumble_marker(x) for x in tx)
        elif t == "truncated->continuation": ok = kp == last_two and all(pref(w, nw(tx[-2] + " " + tx[-1])) for w in W[:-2])
        elif t == "cue_phrases": ok = (not keep) and all(k in CUEK for k in K)
        elif t == "last_take_truncated":
            m = max(range(len(W)), key=lambda i: len(W[i]))
            ok = kp == [m] and all(pref(W[i], W[m]) for i in range(len(W)) if i != m) and len(ids) >= 2
        if not ok: errs.append(f"{gid} [{t}]: label/structure check failed")
    return errs

# ---------------- main ----------------
if __name__ == "__main__":
    base_path, out = sys.argv[1], sys.argv[2]
    with open(base_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f); hdr = rd.fieldnames; base = list(rd)
    nat = [r for r in base if r["source"] == "natural"]; old = [r for r in base if r["source"] == "synthetic"]
    ref = Ref(nat + old)
    natkeys = {key(x) for n_ in nat for x in unit_texts(n_)}
    CUES = [c for c in CUES if key(c) not in natkeys]          # drop cues that equal a real unit
    CUEK = {key(c) for c in CUES}
    for r in old:                       # old synthetic texts must not be reused by the new ones
        for t in unit_texts(r): SEEN.add(key(t))
    new = build(ref, first_video=21)
    errs = validate(new)
    if errs:
        print("\n".join(errs[:40])); sys.exit(f"{len(errs)} validation errors")
    # new rows must not clash with natural texts
    clashes = [r["group_id"] for r in new for t in unit_texts(r) if key(t) in natkeys]
    assert not clashes, clashes[:5]
    allrows = nat + old + new
    with open(out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(allrows)
    wb = Workbook(); ws = wb.active; ws.title = "groups_v2"; ws.append(hdr)
    for r in allrows:
        ws.append([int(r[c]) if c in INT_COLS and str(r[c]).isdigit() else str(r[c]) for c in hdr])
    for row in ws.iter_rows(min_row=1):
        for c in row:
            if not isinstance(c.value, int): c.number_format = "@"; c.data_type = "s"
            c.alignment = Alignment(vertical="top")
    ws.freeze_panes = "B2"; wb.save(out + ".xlsx")
    # report
    syn = old + new
    kinds = collections.Counter(r["synth_type"] for r in syn)
    def shape(rs):
        single = sum(1 for r in rs if r["keeper_unit_ids"] and "|" not in r["keeper_unit_ids"])
        multi = sum(1 for r in rs if "|" in r["keeper_unit_ids"]); none = sum(1 for r in rs if not r["keeper_unit_ids"])
        last = sum(1 for r in rs if r["keeper_unit_ids"] and "|" not in r["keeper_unit_ids"] and r["keeper_unit_ids"] == r["unit_ids"].split("|")[-1])
        return dict(groups=len(rs), single=single, single_keep_last=last, multi=multi, none=none)
    live_nat = [r for r in nat if "STALE" not in r["notes"].upper()]
    rep = dict(total_rows=len(allrows), natural_rows=len(nat), natural_live=len(live_nat), old_synth=len(old), new_synth=len(new),
               synth_types=dict(kinds), natural_shape=shape(live_nat), synthetic_shape=shape(syn),
               units_per_group=dict(collections.Counter(int(r["n_units"]) for r in syn)),
               synth_videos=len({r["video_id"] for r in syn}))
    print(json.dumps(rep, indent=1))
