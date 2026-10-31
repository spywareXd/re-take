# Training and evaluation scripts

These scripts built and evaluated the two trained models. **They need the private labelled data, which is not published** (see `data/synthetic/README.md`), so they are here to show exactly what was done, not to be re-run as they are. The trained weights are on the Hugging Face Hub (links in the main README), so you do not need to train anything to use the pipeline.

To reproduce the idea on your own recordings, label your own pairs and groups in the same column layout as `data/synthetic/*.csv` and point the scripts at them.

## Pair scorer (`data/training/`, `models/final/`)

| Script | What it does |
|---|---|
| `make_folds.py` | speaker-grouped cross-validation folds |
| `baselines*.py`, `logreg*.py` | the plain stage-2 score and logistic-regression baselines |
| `train_cross_encoder*.py` | fine-tunes `cross-encoder/ms-marco-MiniLM-L-6-v2` on labelled pairs (BCE with positive weighting, 3 epochs, max length 128) |
| `combine*.py`, `veto_test*.py`, `neighbor_*.py` | stackers and checks that were tried |
| `models/final/train_final_ce.py` | trains the final three seeds on all labelled rows |
| `models/final/fit_final.py`, `fit_t_amb.py`, `fit_unsure_band.py` | fit the stacker and the thresholds on out-of-fold scores |
| `bank.py`, `make_synth_v2.py` | the synthetic sentence banks and the synthetic-group builder |

## Keeper model (`training/`)

| Script | What it does |
|---|---|
| `build_keeper_sft*.py`, `build_decision_data*.py` | build the supervised training records from labelled groups |
| `score_sft_fold.py`, `score_decision_fold.py`, `decision_eval.py` | score and compare models fold by fold |
| `populate_pairs.py` | runs the pipeline over labelled videos to produce candidate pairs |
| `standalone_export/` | merges the adapter and builds the single-file loader published on the Hub, with parity checks against the training-time code |

The keeper is Qwen3-4B-Instruct-2507 fine-tuned with QLoRA (r 16, alpha 16) plus the joint decision head from Cloudflare's Clef (Apache-2.0). Training used Unsloth. Held-out results and caveats are in the main README and [../docs/figures/](../docs/figures/README.md); the keeper does not beat the keep-last baseline on held-out data.

## Evaluation protocol

Folds are split by speaker. Thresholds are chosen on the other folds only (high = lowest threshold with precision at least 0.95, low = highest with recall at least 0.98, on out-of-fold probabilities), and the numbers reported are pooled out-of-fold results. The keeper recipe was tuned on folds 3 and 4, so only folds 1 and 2 are clean held-out folds.
