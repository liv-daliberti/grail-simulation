# XGB CLI Package

`xgb.cli` contains the command-line interface for the XGBoost slate baseline.
The CLI wires together dataset loading, feature preparation, model training,
and evaluation so experiments can be launched with `python -m xgb.cli`.

## Modules

- `main.py` – argument parser and runtime entry point; exposes flags for fitting,
  loading/saving models, selecting text vectorizers, and controlling evaluation.
- `__main__.py` – enables `python -m xgb.cli` execution by forwarding to
  `main.main`.
- `__init__.py` – re-exports the public CLI helpers for scripting.

## Typical usage

```bash
export PYTHONPATH=src
python -m xgb.cli \
  --dataset data/cleaned_grail \
  --out_dir models/xgb/example \
  --fit_model \
  --text_vectorizer sentence_transformer \
  --eval_max 1000
```

Run `python -m xgb.cli --help` to inspect every flag, including the
`--xgb_*` hyper-parameters and opinion-evaluation switches. Pipeline automation
(`python -m xgb.pipeline`) imports the CLI module internally when launching
sweeps.
