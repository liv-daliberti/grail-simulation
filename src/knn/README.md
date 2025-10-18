# KNN Slate Baseline

This directory contains a light-weight, non-generative baseline that selects the next video from each slate using TF-IDF and k-nearest neighbours. It mirrors the GRPO/GRAIL prompt formatting so we can compare against the reinforcement-learning agents on equal footing.

## What the script does

`knn-baseline.py` transforms the cleaned dataset into the same chat-style prompt used by the GRPO/GRAIL trainers, then:

1. **Prompt construction** – Each row is passed to `build_user_prompt`, giving the KNN model the full PROFILE / HISTORY / CURRENT VIDEO / OPTIONS context. The prompt builder lives in `src/prompt_builder.py` and is shared across baselines and trainers.
2. **TF-IDF index** – The train split is vectorised with scikit-learn’s `TfidfVectorizer`. Each slate size (1, 2, 3, 4, 5+) gets its own sparse matrix so candidate comparisons remain fair.
3. **Candidate-aware ranking** – For a slate item, we combine the prompt text and the candidate’s surface form (title or id) into a query, compute cosine similarity against the rows where that candidate was gold in the train split, and sum the top-k scores. The argmax becomes the predicted option index.
4. **Evaluation outputs** – The script produces per-example JSONL with predictions, along with overall accuracy and option-level metrics. The file layout mirrors the GPT-4o evaluation artifacts.

Because we share the prompt builder and watched-history depth (`GRAIL_MAX_HISTORY` / `KNN_PROMPT_MAX_HISTORY`), this baseline remains aligned with whatever prompt tweaks we make in the main training pipeline.
During evaluation the script sweeps a configurable list of `k` values (default `5,10,25,50` plus `--knn_k`), selects an elbow point automatically, and writes an accuracy-vs-k plot to `reports/knn/` for each issue.

## Quickstart

```bash
python src/knn/knn-baseline.py \
  --dataset data/cleaned_grail \
  --fit_index \
  --knn_k 25 \
  --knn_metric cosine \
  --knn_max_train 200000 \
  --eval_max 500 \
  --out_dir models/knn \
  --overwrite
```

This will automatically:

- Load the cleaned dataset produced by `clean_data/clean_data.py`.
- Discover the available `issue` values (minimum_wage, gun_control).
- Build per-issue TF-IDF indexes (stored under `--save_index/<issue>` if you pass `--save_index`).
- Write predictions/metrics to `--out_dir/<issue>/`.

Key flags:

- `--fit_index` builds a new TF-IDF model from the train split (respecting `--knn_max_train` for subsampling). Use `--load_index path` to reuse a previously saved index.
- `--knn_k` and `--knn_metric` control neighbour search parameters. Supported metrics are `cosine` and `l2`.
- `--knn_k_sweep` evaluates additional k values (default `5,10,25,50`) and selects an elbow point automatically.
- `--knn_text_fields` lets you append additional columns (e.g. `viewer_profile`) onto the prompt text if desired.
- `--eval_max` caps the number of validation rows evaluated, useful for smoke tests.
- `--out_dir` stores the index, predictions, and metrics JSON.
- `--dataset` points to the cleaned dataset root (defaults to `data/cleaned_grail`). You can also supply a Hugging Face dataset id with an `issue` column.
- `--issues` limits evaluation to a subset, e.g. `--issues minimum_wage`.

The script expects the schema emitted by `clean_data/clean_data.py` (train/validation splits with `issue`, prompt columns, slate metadata). When `--save_index` or `--load_index` are set, the script writes/reads from issue-scoped subdirectories so you can reuse models for each policy domain.

## Outputs

For each issue, running the script creates:

- `out_dir/<issue>/index/` – TF-IDF vectoriser, sparse matrix, and label metadata (per slate-size bucket) when `--save_index` is provided.
- `out_dir/<issue>/knn_eval_<issue>_<split>.jsonl` – Row-by-row predictions, including the gold index, predicted index, and the prompt text used.
- `out_dir/<issue>/knn_eval_<issue>_<split>_metrics.json` – Aggregate accuracy, option-level coverage, and timing information.

Use these artifacts to benchmark against GRPO/GRAIL checkpoints or to sanity-check feature drift after modifying the prompt pipeline.
