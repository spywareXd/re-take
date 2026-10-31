# Synthetic data

**This dataset is incomplete.** It is only the synthetic part of the data used to build the scorer and the keeper model. The natural dataset, built from private recordings, is **not published** because it is sensitive data and we do not have permission to release it. Training and evaluation as described in the repository cannot be reproduced from these files alone, and results on synthetic data say little about results on real recordings.

## Files

| File | Rows | What |
|---|---|---|
| `synthetic_pairs_v1.csv` | 501 | sentence pairs: three takes of one line plus hard negatives that share words, topic or meaning-opposite; label 1 = retake, 0 = not |
| `synthetic_pairs_v2.csv` | 203 | the synthetic rows of the second pair set, same columns |
| `synthetic_groups_v2.csv` | 1,200 | groups of takes with the keeper unit(s) chosen by fixed rules (see below) |
| `generators/original_synth/` | 4 scripts | the hand-written sentence banks and the builders of the first 60 groups |

The larger generator of the 1,200 groups is `data/training/make_synth_v2.py` with its word and phrase banks in `data/training/bank.py`.

## How the groups are labelled

One rule function assigns the keeper:

1. truncated or stumbling takes lose to a complete take
2. a take with a mistake loses to the take that corrects it
3. among complete takes the later one wins (identical text: the last)
4. a later take abandoned mid-sentence loses to the earlier complete take
5. a sentence split over two units by a pause keeps both units
6. cue phrases ("here we go", "take two") are not content: keep none

The labels follow the rules, they are not human judgments of real speech.

## Provenance

All sentences are hand-written templates and word banks, not transcripts. They were checked against the private sentences for exact and near-duplicates (character n-gram similarity); the only exact matches are two generic production cues ("Okay, here we go.", "Let's go."), which are in the hand-written cue list.

`make_synth_v2.py` and `make_synth_groups.py` take the private natural groups as an input to avoid clashing with them, so they cannot be run end to end without that data.

## Columns

Pairs: `pair_id, video_id, source_type, speaker, unit_a_id, unit_b_id, text_a, text_b, label, notes, is_hard_negative` and the lexical features `sequence, jaccard, containment, bigram, lexical_score, semantic_score` where present. Groups: `group_id, n_units, unit_ids, texts` (units separated by ` || `, each prefixed `[n]`), `keeper_unit_ids, keep_none, synth_type, notes`.
