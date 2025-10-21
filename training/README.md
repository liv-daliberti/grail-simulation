# Training Launchers

Shell helpers for running the baseline models and reinforcement-learning jobs
described in the project README.

## Scripts

- `training-grpo.sh` – launches the GRPO baseline with the ZeRO-3 accelerate
  profile and optional discriminator disabled.
- `training-grail.sh` – extends `training-grpo.sh` with the discriminator reward
  path used in the full GRAIL experiments.
- `training-knn.sh` – fits and evaluates the kNN slate baseline over the cleaned
  dataset (auto-detects up to 40 Word2Vec worker threads).
- `training-xgb.sh` – runs the XGBoost slate baseline with optional model export
  for reuse across evaluation runs.

## Usage

All scripts assume the cleaned dataset lives at `data/cleaned_grail`. Override
paths or hyper-parameters via environment variables; any additional CLI flags
are forwarded to the underlying Python modules. Example:

```bash
DATASET=/path/to/dataset \
OUT_DIR=models/xgb/run-001 \
bash training/training-xgb.sh --issues minimum_wage
```

### KNN hyperparameter sweeps

`training-knn.sh` exposes the same knobs as `python -m knn.cli`. In addition to
`KNN_*` variables, you can override:

- `WORD2VEC_SIZE`, `WORD2VEC_WINDOW`, `WORD2VEC_MIN_COUNT`, `WORD2VEC_EPOCHS`
- `WORD2VEC_WORKERS` (defaults to `min(nproc, 40)` when unset)

Example TF-IDF sweep using the launcher:

```bash
export DATASET=data/cleaned_grail OUT_DIR=models/knn/sweeps/tfidf
for issue in minimum_wage gun_control; do
  for metric in cosine l2; do
    for fields in "" "viewer_profile,state_text"; do
      label=${fields:-none}
      bash training/training-knn.sh \
        --issues "$issue" \
        --feature-space tfidf \
        --knn-metric "$metric" \
        --knn-k-sweep 1,2,3,4,5,10,15,20,25,50,75,100 \
        --out-dir "$OUT_DIR/${issue}/metric-${metric}_text-${label}" \
        $( [ -n "$fields" ] && printf -- '--knn-text-fields %s' "$fields" )
    done
  done
done
```

and the Word2Vec grid (mirrors the curated sweeps in `reports/knn/hyperparameter_tuning/README.md`):

```bash
export DATASET=data/cleaned_grail OUT_DIR=models/knn/sweeps/word2vec WORD2VEC_WORKERS=40
for issue in minimum_wage gun_control; do
  for metric in cosine l2; do
    for fields in "" "viewer_profile,state_text"; do
      label=${fields:-none}
      for size in 128 256; do
        for window in 5 10; do
          bash training/training-knn.sh \
            --issues "$issue" \
            --feature-space word2vec \
            --knn-metric "$metric" \
            --knn-k-sweep 1,2,3,4,5,10,15,20,25,50,75,100 \
            --word2vec-size "$size" \
            --word2vec-window "$window" \
            --word2vec-min-count 1 \
            --out-dir "$OUT_DIR/${issue}/metric-${metric}_text-${label}_sz${size}_win${window}_min1" \
            $( [ -n "$fields" ] && printf -- '--knn-text-fields %s' "$fields" )
        done
      done
    done
  done
done
```

See `recipes/README.md` for the configuration files consumed by the GRPO and
GRAIL launchers.
