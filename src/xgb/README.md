# XGBoost Slate Baseline

Tree-based counterpart to the `knn` package. The implementation mirrors the
refactored kNN structure but swaps the index for an XGBoost multi-class
classifier that operates over the same prompt documents.

## Package layout

```
src/xgb/
├── cli.py               # CLI for training, evaluation, and model export
├── data.py              # dataset loading + issue filtering (reuses knn helpers)
├── evaluate.py          # evaluation loop, metrics writer, CLI orchestration
├── features.py          # prompt assembly + slate extraction utilities
├── model.py             # TF-IDF vectoriser + XGBoost training / inference code
├── utils.py             # assorted helpers (video-id canonicalisation, logging)
└── xgboost-baseline.py  # legacy shim that now calls xgb.cli:main
```

## Quick start

Fit a model and evaluate on the default dataset:

```bash
python -m xgb.cli \
  --fit_model \
  --dataset data/cleaned_grail \
  --out_dir models/xgb/run-001
```

You can optionally persist the trained bundle for later reuse:

```bash
python -m xgb.cli \
  --fit_model \
  --save_model models/xgb/checkpoints

python -m xgb.cli \
  --load_model models/xgb/checkpoints \
  --out_dir models/xgb/eval \
  --issues minimum_wage,gun_control
```

CLI arguments cover common experimentation knobs:

- `--extra_text_fields` appends additional columns to the prompt document before
  vectorisation (useful for ablations).
- `--max_train`, `--max_features`, and `--seed` control subsampling and TF-IDF
  vocabulary size.
- `--xgb_*` flags forward hyper-parameters directly to `xgboost.XGBClassifier`.

See `python -m xgb.cli --help` for the full list.

## Implementation notes

- `features.prepare_prompt_documents` builds TF-IDF-friendly text while ensuring
  slate ordering matches the prompt consumed by other baselines.
- The model always trains with `multi:softprob`, enabling per-slate probability
  vectors. `predict_among_slate` extracts the top-ranked option that appears in
  the provided candidate set.
- Evaluation reports accuracy and slate coverage so metrics can be compared to
  the kNN and GPT-4o baselines.

Pull requests that extend the feature pipeline or add richer metrics are welcome
—keep the CLI backwards compatible so existing experiment scripts keep working.
