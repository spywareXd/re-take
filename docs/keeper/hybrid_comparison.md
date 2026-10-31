# Hybrid keeper vs the model alone vs gpt-oss vs keep-last

Written for the project authors, 2026-10-28. The hybrid keeper is the `finetuned` Step 6 provider now in the pipeline: the fine-tuned Qwen3-4B decision model behind the rule policy (strong cue group, exact gate, empty-set fallback, unclear default; see `pipeline_update.md`). Numbers are rebuilt by `make_comparison_assets.py`; per-group results are in `data/comparison.json`.

## How it was scored

- **Data:** the 131 labelled natural groups. Scoring is exact keeper-set match.
- **Model and hybrid:** each group is scored with the model of its outer fold, a model that never saw that fold, using its stored probabilities. The final all-data model can't be scored here, because it trained on every group.
- **Hybrid cue list:** for each fold, the cue list leaves out that fold's own natural cue phrases.
- **Groq gpt-oss-120b:** one cached pass at temperature 0, with 0 failures.
- **Caveat on clean data:** folds 1 and 2 are clean. Folds 3 and 4 were the set used to choose the training recipe.
- **Caveat on the policy:** the hybrid policy was designed **after** the four-fold results were known. Its rules were chosen while looking at these groups, so its numbers are an optimistic estimate, not a clean test.

## Results

| system | correct of 131 | % | nuanced (of 14) | keep-last groups (of 117) |
|---|---|---|---|---|
| **Hybrid (model + rules)** | **118** | 90.1 | 3 | 115 |
| keep-last | 117 | 89.3 | 0 | 117 |
| Rules only (cue rule + exact gate + keep-last) | 117 | 89.3 | 0 | 117 |
| Fine-tuned model alone | 114 | 87.0 | 5 | 109 |
| Groq gpt-oss-120b | 108 | 82.4 | 5 | 103 |

![overall](figures/h1_overall.png)

| fold | n | hybrid | model | Groq | keep-last | rules only |
|---|---|---|---|---|---|---|
| 1 (clean) | 33 | 30 | 31 | 29 | 29 | 29 |
| 2 (clean) | 39 | 37 | 37 | 35 | 38 | 38 |
| 3 (tuned on) | 25 | 22 | 21 | 21 | 22 | 22 |
| 4 (tuned on) | 34 | 29 | 25 | 23 | 28 | 28 |

![by fold](figures/h2_by_fold.png)

On the clean folds alone (72 groups) the hybrid scores 67, the model alone 68, keep-last 67 and Groq 64.

## Where the systems differ

![nuanced vs keep-last](figures/h3_nuanced_vs_keeplast.png)

The hybrid trades nuanced groups for keep-last groups:
- **On keep-last groups (117):** it gets 115 right, against 109 for the model alone and 103 for Groq. The fallback and exact gate repair the model's misses there.
- **On nuanced groups (14):** it gets 3, against 5 for the model alone and 5 for Groq. Two cases are lost on purpose:
  - speaker_a_4__g1 is a real keep-none cue group. The model said keep none, but the takes are not on the cue list (scored without that group's own phrases), so the fallback keeps the last take.
  - video_02__g1 is a two-keeper group. The unclear rule keeps only the last take, which is the tradeoff you accepted, and the human review step is expected to catch it.

| hybrid layer | groups | correct |
|---|---|---|
| model | 104 | 93 |
| exact gate | 20 | 20 |
| empty-set fallback | 7 | 5 |
| cue rule | 0 | |

Nine groups were flagged unclear for review.

## Are the differences real?

Paired bootstrap over groups (1,000 resamples):

| comparison | difference (points) | 95% interval | groups only the hybrid gets right | groups only the comparator gets right |
|---|---|---|---|---|
| hybrid vs model alone | +3.1 | [-0.8, +7.6] | 6 | 2 |
| hybrid vs Groq gpt-oss-120b | +7.6 | [+1.5, +13.7] | 14 | 4 |
| hybrid vs keep-last | +0.8 | [-2.3, +4.6] | 3 | 2 |

![paired intervals](figures/h4_paired_intervals.png)

- **Against Groq:** the interval excludes zero, which is the only comparison here that does. The hybrid gets 14 groups that Groq misses, 13 keep-last groups and 1 multi-keeper group (video_05__g10). Groq gets 4 that the hybrid misses: video_07__g15, video_04__g5 and video_04__g15, which are earlier-keeper groups, and video_07__g19. The hybrid's rules were fixed after seeing these groups, so treat this as strong evidence that the hybrid is not worse than Groq, and as weaker evidence that it is better.
- **Against keep-last and the model alone:** no reliable difference.

The groups where the hybrid and Groq disagree are listed in `data/comparison_tables.md`.

## Unlabelled end-to-end check (sample_b, cut mode `sentences`)

| | hybrid (default) | gpt-oss |
|---|---|---|
| groups | 9: model 8, empty fallback 1 | 9 resolved |
| keeper stage | 6.80 s | 13.21 s |
| external network calls | none | the Groq API |
| removals / keep spans | 12 / 11 | 12 / 11 |
| rendered vs sum of keep spans | 28.445 s vs 28.230 s | 29.029 s vs 28.800 s |

The two agree on 8 of 9 groups. On one group of three near-identical takes, the hybrid keeps the last take by the fallback and gpt-oss keeps the middle one. sample_b has no labels, so neither answer is scored. Details: `runs/e2e_sample_b_hybrid_sentences/RESULT.md` (git-ignored).

## Reading

- **Against gpt-oss:** on this data the hybrid is at least as accurate. It needs no API key or network, and its keeper stage was faster on sample_b.
- **Against keep-last:** the hybrid ties it. Its value over keep-last is limited to the few nuanced groups where the model is confident, plus the flags it raises for review.
- **What it costs:** it gives up some nuanced groups that the model alone got right, in exchange for fewer wrong answers on ordinary keep-last groups.
- **What would settle it:** a fair test needs new labelled recordings with more nuanced groups, scored with the policy fixed in advance.

## Limits

1. **Small nuanced sample:** of the 131 groups, 14 are nuanced, and only 1 is a natural cue group.
2. **Post-hoc policy:** the policy and its thresholds were chosen after seeing these groups.
3. **Not fully clean:** folds 3 and 4 tuned the training recipe.
4. **One seed per fold:** a second seed on fold 4 changed 7 of 34 model decisions.
5. **Groq measured once:** it was scored on one cached pass.
6. **Final model not scored:** the deployed model (`final_all`) was not itself scored on natural data. The fold models stand in for it.

## Files

- `make_comparison_assets.py`
- `figures/h1_overall`, `h2_by_fold`, `h3_nuanced_vs_keeplast`, `h4_paired_intervals` (PNG and PDF)
- `data/comparison_tables.md`, `data/comparison.json`, `data/policy_scores.json`
