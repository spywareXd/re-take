# Figures

Aggregate results only: counts, accuracies and loss curves. No transcript text, speaker names or video names appear in any figure. They were produced on the private labelled data (see `data/synthetic/README.md`), so they can be inspected but not re-created from this repository. The scripts that drew them are in `scripts/` and need the private result tables.

"Natural groups" are 131 labelled groups of repeated takes from private recordings; folds 3 and 4 were used to tune the recipe, folds 1 and 2 are the clean held-out folds. "Keep-last" is the baseline that always keeps the last take. "Groq gpt-oss-120b" is the hosted comparison model.

| Figure | What it shows |
|---|---|
| `f1_accuracy_by_split` | 4B fine-tune vs Groq vs keep-last on the dev folds and the two held-out folds |
| `f2_nuanced_vs_keeplast` | Accuracy on the 14 groups where the keeper is not the last take vs the 117 where it is |
| `f3_paired_differences` | Paired bootstrap 95% intervals of the model minus each comparator; an interval touching zero is no evidence of a difference |
| `f4_training_loss` | Training and evaluation loss of the dev, fold-1 and fold-2 models |
| `f5_synthetic_by_type` | Held-out synthetic groups (220, 20 per rule type), fold-1 and fold-2 models vs keep-last |
| `f6_probability_histogram` | Held-out p(keep) of the model: nearly all units are at 0 or 1 |
| `f7_four_fold` | Four-fold run, one model per outer fold, pooled over 131 groups |
| `g1_final_loss` | Loss of the final model trained on all 131 natural groups plus 1,200 synthetic ones |
| `g2_synthetic_by_type` | The same synthetic check for the final model and the four fold models |
| `g3_second_seed_fold4` | Fold 4 with two training seeds and per-group agreement between them |
| `g4_keep_last_fallback` | Effect of the "keep the last take if the model keeps nothing" fallback, per fold |
| `g5_pooled_summary` | Pooled accuracy: model alone, model + fallback, keep-last, Groq |
| `h1_overall` | Hybrid (model + rules) vs keep-last vs gpt-oss-120b on the 131 groups |
| `h2_by_fold` | The same by outer fold, marking which folds were used for tuning |
| `h3_nuanced_vs_keeplast` | Where the systems differ: groups whose keeper is not the last take |
| `h4_paired_intervals` | Paired intervals for hybrid vs model alone, vs gpt-oss-120b, vs keep-last |

Read these with the caveats in the main README: the model alone does not beat keep-last, the hybrid rule set was chosen after seeing the held-out results, and fold 3 and 4 numbers are on the tuning set.
