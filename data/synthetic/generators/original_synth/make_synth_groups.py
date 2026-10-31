"""Append synthetic keeper-selection groups to the natural groups sheet.

Keepers are assigned by the same rules used for the natural labels:
  1 truncated / stumbling takes lose to a complete take
  2 a take with a mistake loses to the take that corrects it
  3 among complete takes the later one wins (identical text -> last)
  4 a later take that is abandoned mid-sentence loses to the earlier complete take
  5 a sentence split over two units by a pause keeps both units (continuation)
  7 cue phrases ("here we go", "take two") are not content -> keep_none
Usage: python make_synth_groups.py BASE.xlsx OUT_PREFIX
"""
import csv, random, sys
sys.path.insert(0, ".")
from load_base import load
from openpyxl import Workbook
from openpyxl.styles import Alignment

R = random.Random(0)
INT_COLS = {"n_units", "n_pos_pairs", "n_exempt_pairs"}

# ---------- text helpers ----------
def words(s): return s.rstrip(".?!").split()
def trunc(s, frac=None):
    w = words(s); frac = frac or R.uniform(0.35, 0.7)
    k = max(3, min(len(w) - 3, round(frac * len(w))))
    return " ".join(w[:k]) + R.choice([",", "...", "", ","])
def head_stutter(s, k=3):
    w = words(s); k = min(k, len(w) - 4); h = " ".join(w[:k])
    return f"{h}, {h[0].lower() + h[1:]} {' '.join(w[k:])}."
def mid_stutter(s):
    w = words(s); i = len(w) // 2; pre = w[:i]
    return " ".join(pre) + ", " + " ".join(pre[-2:]) + " " + " ".join(w[i:]) + "."
def filler(s, f="uh"):
    w = words(s); k = max(3, len(w) // 2)
    return f"{' '.join(w[:k])}, {f}, {' '.join(w[k:])}."

# ---------- specs ----------
A = [  # truncated -> complete  (index 3 and 10 get two truncated takes)
 "Preheat the oven to 200 degrees and line a baking tray with parchment paper.",
 "The first thing you need to do is open the terminal and install the dependencies.",
 "Compound interest means you earn returns on both your original money and your past returns.",
 "Make sure your elbows stay tucked in as you lower the bar toward your chest.",
 "The Eiffel Tower was built for the 1889 World's Fair and was meant to be temporary.",
 "Water the plants early in the morning so the leaves can dry before the sun gets strong.",
 "If you want sharper photos, switch to a faster shutter speed and raise the ISO a little.",
 "Our new battery lasts about two full days on a single charge with normal use.",
 "Photosynthesis is the process plants use to turn sunlight, water and carbon dioxide into sugar.",
 "In this deck you want to cycle cheap cards quickly and keep your elixir advantage.",
 "Before you book anything, check whether your passport is valid for at least six months.",
 "A good morning routine starts with water, light movement and ten minutes without your phone.",
 "The Roman Empire reached its greatest size under the emperor Trajan.",
 "Always back up your project folder before you try to update the plugin.",
 "Let me show you how to fold the dough so it stays light and airy.",
]
A_DOUBLE = {3, 10}
B = [  # (base, variant)
 ("Today I'm going to compare the two cheapest laptops you can buy right now.", "head"),
 ("The trick is to keep your wrist relaxed and let the brush do the work.", "mid"),
 ("You'll notice the sauce thickens quickly once the cheese is fully melted.", "filler"),
 ("So the main idea behind this algorithm is to split the problem into smaller parts.", "head"),
 ("We tested the camera in low light and the results were honestly impressive.", "mid"),
 ("Think about what you actually want from this job before you accept the offer.", "filler"),
 ("Every measurement matters when you're weighing the ingredients for this recipe.", "head"),
 ("This update fixes the crash on startup and improves the loading time.", "head+filler"),
]
C = [("Let's get started.", 2), ("Thanks for watching, see you next time.", 2), ("Now watch this carefully.", 2),
     ("That's it for today.", 2), ("Pay attention to this part.", 2), ("Here's what you need.", 2),
     ("This is really important.", 3), ("And that's the whole recipe.", 3), ("Now we're ready to begin.", 3),
     ("Let's look at the results.", 4), ("Subscribe for more.", 4), ("Welcome back to the channel.", 6)]
D = [
 ["The app is free to download, but some features cost extra.", "You can download the app for free, though a few features are paid."],
 ["Next, we'll talk about how to choose the right size.", "Now let's go over how you pick the right size."],
 ["This trail is about five miles long and fairly easy.", "The trail runs roughly five miles and it's not too difficult."],
 ["Sleep is the foundation of good health.", "Good health really starts with sleep."],
 ["Mix the flour and sugar before adding the eggs.", "Add the eggs only after the flour and sugar are mixed together."],
 ["Our team spent six months building the first prototype.", "It took the team half a year to build the first prototype."],
 ["Keep your posture straight while you type.", "While typing, make sure you're sitting up straight."],
 ["The museum opens at nine and closes at five.", "You can visit the museum between nine in the morning and five in the afternoon."],
 ["Lower the price and sales will go up.", "If you reduce the price, more people will buy.", "Drop the price a little and you'll see sales rise."],
 ["Start slowly and speed up as you warm up.", "Begin at an easy pace, then build up once you're warm.", "Ease into it and pick up the pace once your body warms up."],
]
E = [  # (earlier takes with the mistake..., correct take, what was wrong)
 (["Mount Everest is about eight hundred meters tall."], "Mount Everest is about eight thousand eight hundred meters tall.", "height"),
 (["The first moon landing took place in 1979."], "The first moon landing took place in 1969.", "year"),
 (["Water boils at ninety degrees Celsius at sea level."], "Water boils at one hundred degrees Celsius at sea level.", "temperature"),
 (["Python was created by Linus Torvalds in 1991.", "Python was created by Linus Torvalds in 1991."], "Python was created by Guido van Rossum in 1991.", "name"),
]
F = [  # (earlier complete take, part A, part B, unclear)
 ("If you leave the dough overnight it will taste much better.", "If you leave the dough in the fridge overnight,", "the flavor develops and it tastes much better.", False),
 ("When pressure drops the pump turns on automatically.", "When the pressure drops below the threshold,", "the pump automatically switches on to restore it.", False),
 ("Even if you have never edited a video, this tutorial walks you through it.", "Even if you've never edited a video before,", "this tutorial will walk you through every step.", True),
 ("Mix it all and let the batter rest twenty minutes.", "After you've mixed everything together,", "let the batter rest for about twenty minutes.", False),
]
G = [["Okay, here we go.", "Alright, from the top.", "Let me start again."],
     ["Take two.", "Okay, rolling.", "Here we go again."],
     ["Let's go.", "Ready, from the beginning.", "One more time.", "Okay, again."]]
H = [  # keeper is NOT the last take
 ("Remember to save your work every few minutes in case something goes wrong.", "2"),
 ("The best time to plant tomatoes is after the last frost has passed.", "2"),
 ("We recommend charging the device fully before you use it for the first time.", "2"),
 ("Slice the onions thinly so they cook evenly and turn golden.", "3"),
]

def build_specs():
    S = []
    for i, b in enumerate(A):
        takes = [[trunc(b)]] + ([[trunc(b)]] if i in A_DOUBLE else []) + [[b]]
        S.append(dict(type="truncated->complete", takes=takes, keep=[len(takes) - 1], note="SYNTHETIC truncated->complete; chose the complete take"))
    for b, v in B:
        t = {"head": [head_stutter(b), b], "mid": [mid_stutter(b), b], "filler": [filler(b), b],
             "head+filler": [head_stutter(b), filler(b), b]}[v]
        S.append(dict(type="stutter->clean", takes=[[x] for x in t], keep=[len(t) - 1], note="SYNTHETIC stutter->clean; chose the clean take"))
    for txt, k in C:
        S.append(dict(type="identical", takes=[[txt]] * k, keep=[k - 1], note="SYNTHETIC identical text; chose last"))
    for v in D:
        S.append(dict(type="paraphrase", takes=[[x] for x in v], keep=[len(v) - 1], note="SYNTHETIC paraphrase restart; keeper unclear; chose the later take"))
    for wrong, right, what in E:
        t = wrong + [right]
        S.append(dict(type="mistake->corrected", takes=[[x] for x in t], keep=[len(t) - 1], note=f"SYNTHETIC earlier take misstates the {what}; later take is correct"))
    for first, a, b, unc in F:
        S.append(dict(type="continuation", takes=[[first], [a, b]], keep=[1],
                      note="SYNTHETIC continuation: last two units form one sentence (pause); keep both" + ("; keeper unclear" if unc else "")))
    for g in G:
        S.append(dict(type="cue_phrases", takes=[[x] for x in g], keep=[], none=True, note="SYNTHETIC cue phrases, not script content"))
    for base, shape in H:
        if shape == "2":
            t = [[base], [trunc(base, 0.45)]]; keep = [0]; note = "SYNTHETIC last take abandoned mid-sentence; kept the earlier complete take"
        else:
            t = [[trunc(base, 0.4)], [base], [trunc(base, 0.55)]]; keep = [1]; note = "SYNTHETIC middle take is the only complete one; keeper unclear"
        S.append(dict(type="last_take_truncated", takes=t, keep=keep, note=note))
    return S

# ---------- row construction ----------
def dur(text): return len(text.split()) / 2.7 + R.uniform(0.25, 0.7)

def make_row(vid, gi, spec, n, t):
    ids, texts, starts, ends, take_of, part_of = [], [], [], [], [], []
    for ti, parts in enumerate(spec["takes"]):
        if ti > 0 and R.random() < 0.3: n += 1                    # skipped unit number (non-group speech)
        for pj, txt in enumerate(parts):
            if pj > 0: t += R.uniform(0.4, 1.0)                   # pause inside a sentence
            ids.append(f"{n}:{pj}"); texts.append(txt); take_of.append(ti); part_of.append(pj)
            starts.append(round(t, 2)); t += dur(txt); ends.append(round(t, 2))
        n += 1
        t += R.uniform(0.25, 0.9) if R.random() < 0.65 else R.uniform(1.6, 6.0)
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if take_of[i] != take_of[j]:
                pairs.append(f"{vid}__{ids[i].replace(':', '-')}__{ids[j].replace(':', '-')}")
    keep_ids = [ids[i] for i in range(len(ids)) if take_of[i] in spec["keep"]]
    row = dict(group_id=f"{vid}__g{gi}", video_id=vid, n_units=len(ids), unit_ids="|".join(ids),
               texts=" || ".join(f"[{k+1}] {x}" for k, x in enumerate(texts)),
               starts="|".join(map(str, starts)), ends="|".join(map(str, ends)),
               keeper_unit_ids="|".join(keep_ids), keep_none="1" if spec.get("none") else "",
               notes=spec["note"], pipeline_keeper_unit_id="", heuristic_later=ids[-1], heuristic_earlier=ids[0],
               heuristic_confidence="", pair_ids="|".join(pairs), n_pos_pairs=len(pairs), n_exempt_pairs=0,
               feature_version="v1-synthetic", source="synthetic", synth_type=spec["type"])
    return row, n, t

def build_synth():
    S = build_specs(); R.shuffle(S)
    rows, k = [], 0
    for v in range(1, 21):
        vid = f"synth_{v:02d}"; n, t = R.randint(1, 3), R.uniform(1, 6)
        for gi in range(1, 4):
            row, n, t = make_row(vid, gi, S[k], n, t); k += 1
            rows.append(row); n += R.randint(1, 3); t += R.uniform(4, 12)
    assert k == len(S) == 60
    return rows

# ---------- main ----------
if __name__ == "__main__":
    base_path, out = sys.argv[1], sys.argv[2]
    hdr, nat = load(base_path)
    for r in nat:
        r["source"] = "natural"; r["synth_type"] = ""
    syn = build_synth()
    cols = hdr + ["source", "synth_type"]
    allrows = nat + syn
    with open(out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(allrows)
    wb = Workbook(); ws = wb.active; ws.title = "groups_v1_with_synth"; ws.append(cols)
    for r in allrows:
        ws.append([int(r[c]) if c in INT_COLS and str(r[c]).isdigit() else str(r[c]) for c in cols])
    for row in ws.iter_rows(min_row=1):
        for c in row:
            if not isinstance(c.value, int): c.number_format = "@"; c.data_type = "s"
            c.alignment = Alignment(vertical="top")
    for k, v in {"A": 20, "B": 12, "C": 8, "D": 26, "E": 60, "F": 22, "G": 22, "H": 16, "I": 10, "J": 60}.items():
        ws.column_dimensions[k].width = v
    ws.freeze_panes = "B2"; wb.save(out + ".xlsx")
    print("wrote", out, len(nat), "natural +", len(syn), "synthetic")
