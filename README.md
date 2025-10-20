# GRAIL Simulation

Grounded-Retrieval Adversarial Imitation Loop (GRAIL) is a framework for grounded human-behaviour simulation that unifies language, agent, and world models. The agent retrieves realistic action slates, reasons about them with a ReAct-style loop, predicts counterfactual outcomes, and aligns to real trajectories through adversarial training.

![GRAIL overview](docs/Simulation.drawio.png)

The interaction logs trace back to the public behavioural dataset introduced in [PNAS (2024)](https://www.pnas.org/doi/epdf/10.1073/pnas.2318127122) and distributed via the companion CodeOcean capsule. Across Studies 1–3 the mean opinion shift per issue remains below 0.05, underscoring how rare substantive changes were in the original experiments. Detailed replication tables and plots live in [reports/research_article_political_sciences/README.md](reports/research_article_political_sciences/README.md).

## Key Components

- **Environment model** – retrieves candidate next actions from behaviour logs to keep the agent grounded.
- **Action model (ReAct)** – reasons over the retrieved slate and emits the next action.
- **Predictor / world model** – estimates outcomes and counterfactuals for the selected action.
- **Sequential discriminator** – provides adversarial rewards that align generated trajectories with human data.

## Repository Layout

```
.
├── clean_data/               # Data cleaning pipeline and prompt analytics helpers
│   └── prompt/               # Plotting + Markdown utilities for prompt diagnostics
├── docs/                     # Sphinx configuration, figures, and rendered reports
├── recipes/                  # Training configuration files organised by model family
├── src/
│   ├── open_r1/              # Supervised fine-tuning & RL trainers (see src/open_r1/README.md)
│   ├── gpt4o/                # GPT-4o slate-prediction baselines
│   ├── knn/                  # Non-generative k-nearest-neighbour baseline
│   └── visualization/        # Graphviz-based session and recommendation-tree renderers
├── training/                 # SLURM launchers for baseline and GRAIL runs
└── setup.py                  # Editable package definition (pip install -e .)
```

## Quickstart

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The editable install pulls in everything required for cleaning, training, and evaluation. For development tasks (linting, tests, docs) install the extras with `pip install -r requirements-dev.txt`.

### 2. Clean the Dataset

Recreate the CodeOcean preprocessing pipeline with:

```bash
python -m clean_data.cli \
  --dataset-name capsule-5416997/data \
  --output-dir data/cleaned_grail \
  --prompt-stats-dir reports/prompt_stats \
  --issue-repo gun_control=my-org/grail-gun \
  --issue-repo minimum_wage=my-org/grail-wage \
  --push-to-hub --hub-token $HF_TOKEN
```

The command:

1. Rebuilds the session dataframe (`clean_data.sessions.build_codeocean_rows`).
2. Converts each interaction into a GRPO-style prompt example (`clean_data.prompting.row_to_example`).
3. Runs prompt analytics (plots + Markdown) if `--prompt-stats-dir` is provided, using a deduplicated `(participant_id, issue)` view for historical comparability.
4. Saves the full cleaned dataset (all interactions) to disk and optionally pushes per-issue splits to the Hugging Face Hub.

Builder notes:

- Rows missing all survey demographics are removed so every prompt has viewer context (~22 % of raw interactions).
- The on-disk dataset preserves the complete interaction history for each participant across Studies 1–3; use `clean_data.clean_data.dedupe_by_participant_issue` if you need the legacy one-row-per-participant layout.
- Study 4 (YouTube Shorts) remains in the allow-list reporting but is excluded from saved prompt rows because the released logs lack recommendation slates.

#### Output artefacts

Running the CLI yields two complementary views:

- `--output-dir`: full cleaned dataset with every promptable interaction.
- `--prompt-stats-dir`: figures + markdown computed from the deduped `(participant_id, issue)` view so coverage charts remain comparable with earlier reports.

### 3. Prompt Analytics

Re-run the prompt coverage report later without rebuilding the dataset:

```bash
python -m clean_data.prompt.cli \
  --dataset data/cleaned_grail \
  --output-dir reports/prompt_stats
```

The package generates histograms, participant summaries, and a Markdown README in the target directory.

### 4. Visualise Recommendation Trees

Render individual sessions or entire recommendation trees:

```bash
python src/visualization/recommendation_tree_viz.py \
  --cleaned-data data/cleaned_grail \
  --session-id 004QUceaM2cVUOrEO1iD \
  --max-steps 5 \
  --output docs/session_example.svg
```

Batch mode (`--batch-output-dir`, `--batch-issues`) produces per-issue session samples ready for documentation or inspection.

### 5. Automated Checks

- `scripts/run-lint.sh` – `pylint` with the repository root on `PYTHONPATH`.
- `scripts/run-tests.sh` – `pytest` for the unit test suite.
- CI (see `.github/workflows/ci.yml`) installs `requirements-dev.txt` and runs both scripts on push/PR.

### 6. Documentation

```bash
pip install -r requirements-dev.txt
make -C docs html
```

The HTML documentation (autodoc + autosummary for `clean_data`) lands in `docs/_build/html`.

## End-to-End Pipeline

The repository stitches together several subsystems to turn raw CodeOcean logs into reproducible training/evaluation runs. The core stages are:

1. **Session ingestion & filtering** – `clean_data.sessions.build_codeocean_rows` loads the capsule exports, enforces participant allow-lists, and retains the full interaction history for every `(participant, issue)` pair.
2. **Prompt construction** – `clean_data.prompting.row_to_example` builds GRPO-style prompts, applying the shared viewer-profile logic used by downstream models.
3. **Feature extraction** – `src/knn/features.py` assembles text documents and optionally trains Word2Vec embeddings (`Word2VecFeatureBuilder`) or TF-IDF vectors.
4. **Index training** – `src/knn/index.py` fits the chosen feature space (`build_tfidf_index` / `build_word2vec_index`) and persists per-issue artefacts.
5. **KNN evaluation & elbow selection** – `src/knn/evaluate.py` scores validation examples, logs running accuracies, generates accuracy-by-`k` curves, and selects the elbow-based `k`.
6. **Reporting** – metrics, per-`k` predictions, elbow plots, and curve diagnostics are written to `models/` and `reports/`.

High-level progression (training + evaluation):

```
        Raw Capsule Exports
                  |
                  v
        clean_data.sessions
                  |
                  v
         Prompt Builder (GRPO)
                  |
                  v
        +-----------------------+
        |  Feature Extraction   |
        |  - TF-IDF (default)   |
        |  - Word2Vec (optional)|
        +-----------+-----------+
                    |
            +-------v-------+
            |  KNN Index   |
            |  Training    |
            +-------+------+
                    |
      +-------------v--------------+
      | KNN Evaluation (metrics,   |
      | elbow curve, acc@k logs)   |
      +-------------+--------------+
                    |
        +-----------+-----------+
        | Reports & Artefacts  |
        |  models/, reports/   |
        +----------------------+
```

Both `bash training/training-knn.sh` and the Python modules follow this path; setting `--feature-space word2vec` switches the feature block while keeping the rest intact.

## Data Sources

Raw data mirrors the CodeOcean capsule at <https://codeocean.com/capsule/5416997/tree/v1>. The commands below reproduce the original download:

```bash
git clone https://git.codeocean.com/capsule-5416997.git
cd capsule-5416997
mkdir results
curl -fL -OJ 'https://codeocean-temp.s3.amazonaws.com/.../results.zip'
unzip results-*.zip
mkdir ../data
cd ../data
curl -fL --retry 5 --retry-all-errors -o capsule-5416997-data.zip 'https://codeocean-temp.s3.amazonaws.com/.../data.zip'
```

The Python builder consumes the intermediate CSV/RDS exports from these folders, reproducing the same attention checks and control-arm drops as the R scripts; the optional `dedupe_by_participant_issue` helper matches the original deduplication when you need that projection.

## Published Artifacts

- Minimum wage split: <https://huggingface.co/datasets/od2961/grail-wage>
- Gun control split: <https://huggingface.co/datasets/od2961/grail-gun>
- Prompt statistics (plots + Markdown): `reports/prompt_stats`

## Training

Launch the reinforcement-learning recipes via SLURM:

```bash
sbatch training/training-grail.sh    # RL + discriminator shaping (src/open_r1/grail.py)
sbatch training/training-grpo.sh     # RL baseline (src/open_r1/grpo.py)
```

Baselines:

- GPT‑4o slate predictor: `python -m gpt4o.cli`
- k-NN slate baseline: `bash training/training-knn.sh`

## Citation

```bibtex
@inproceedings{GRAIL-2025,
  title     = {GRAIL Simulation: Grounded Retrieval for Human Behavior Alignment},
  author    = {Liv G. d'Aliberti and Manoel Horta Ribeiro},
  booktitle = {NeurIPS 2025 (Non-Archival Track)},
  year      = {2025},
  note      = {NeurIPS non-archival},
  url       = {https://openreview.net/pdf?id=MdmszyLpVu}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
