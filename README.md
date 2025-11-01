# GRAIL Simulation

Grounded-Retrieval Adversarial Imitation Loop (GRAIL) is a framework for grounded human-behavior simulation that unifies language, agent, and world models. The agent retrieves realistic action slates, reasons about them with a ReAct-style loop, predicts counterfactual outcomes, and aligns to real trajectories through adversarial training.

![GRAIL overview](docs/Simulation.drawio.png)

The interaction logs trace back to the public behavioral dataset introduced in [PNAS (2024)](https://www.pnas.org/doi/epdf/10.1073/pnas.2318127122) and distributed via the companion CodeOcean capsule. Across Studies 1–3 the mean opinion shift per issue remains below 0.05, underscoring how rare substantive changes were in the original experiments. Detailed replication tables and plots live in [reports/research_article_political_sciences/README.md](reports/research_article_political_sciences/README.md).

## Key Components

- **Environment model** – retrieves candidate next actions from behavior logs to keep the agent grounded.
- **Action model (ReAct)** – reasons over the retrieved slate and emits the next action.
- **Predictor / world model** – estimates outcomes and counterfactuals for the selected action.
- **Sequential discriminator** – provides adversarial rewards that align generated trajectories with human data.
- **Baseline pipelines** – shared CLI and orchestration utilities in `src/common` coordinate the k-NN and XGBoost baselines across plan → sweeps → finalize → reports stages.

## Repository Layout

```
.
├── README.md                 # Project overview, setup, and usage guide
├── LICENSE                   # Project license (MIT-style)
├── pytest.ini                # Root pytest configuration consumed by scripts/run-tests.sh
├── .readthedocs.yaml         # Read the Docs configuration (must live at repo root)
├── .github/                  # GitHub Actions workflows
├── capsule-5416997/          # Snapshot of the CodeOcean capsule inputs/metadata
├── clean_data/               # Data cleaning pipeline and publication replicas
│   ├── sessions/             # Session ingestion helpers (see clean_data/sessions/README.md)
│   ├── prompt/               # Prompt analytics plots and Markdown builders
│   └── research_article_political_sciences/  # Replication figures + summaries
├── data/                     # Local cleaned datasets (gitignored artifacts)
├── development/              # Centralized tooling configs (linting, packaging helpers)
│   ├── .pylintrc             # Pylint configuration
│   ├── pytest.ini            # Pytest configuration for editable installs
│   ├── requirements-dev.txt  # Development-only dependencies
│   └── setup.py              # Editable package definition (`pip install -e development`)
├── docs/                     # Sphinx project that powers the Read the Docs site
├── models/                   # Trained model checkpoints and evaluation curves
├── recipes/                  # Training configuration files organized by model family
├── reports/                  # Markdown reports rendered from analyses
├── scripts/                  # Utility entrypoints (linting, testing, exports)
├── src/                      # Python packages for agents, models, and visualization
│   ├── common/               # Shared CLI, prompt, pipeline, and RL utilities (see common/open_r1/)
│   ├── grail/                # GRPO + discriminator (GAIL) training entry points
│   ├── grpo/                 # GRPO baseline training and evaluation pipelines
│   ├── knn/                  # Reworked k-NN baseline (cli/, core/, pipeline/)
│   ├── xgb/                  # Reworked XGBoost baseline (cli/, core/, pipeline/)
│   ├── gpt4o/                # GPT-4o slate-prediction baseline + pipeline wrapper
│   ├── prompt_builder/       # Prompt templating helpers packaged for reuse
│   └── visualization/        # Recommendation-tree renderers / plotting tools
├── tests/                    # Pytest suite covering data + model components
└── training/                 # SLURM launchers and experiment configs
```

> **Note:** Runtime artifacts (for example `logs/` or `.cache/`) are gitignored and created on demand by the training and evaluation scripts.

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

If you want the exact snapshot used in this repository without re-running the capsule export, pull it directly from Hugging Face: <https://huggingface.co/datasets/od2961/grail-codeocean-raw>.

The Python builder consumes the intermediate CSV/RDS exports from these folders, reproducing the same attention checks and control-arm drops as the R scripts; the optional `dedupe_by_participant_issue` helper matches the original deduplication when you need that projection.

See [reports/research_article_political_sciences/README.md](reports/research_article_political_sciences/README.md) for the replication report that shows the pipeline mirrors the PNAS capsule while tracing the per-participant delta movement we model for individual-level prediction.

## Quickstart

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e development
```

The editable install pulls in everything required for cleaning, training, and evaluation. For development tasks (linting, tests, docs) install the extras with `pip install -r development/requirements-dev.txt`.

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
- Additional prompt coverage methodology is documented in [reports/prompt_stats/README.md](reports/prompt_stats/README.md).

#### Output artifacts

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

The package generates histograms, participant summaries, and a Markdown README in the target directory; browse the curated write-up at [reports/prompt_stats/README.md](reports/prompt_stats/README.md).

### 4. Visualize Recommendation Trees

Render individual sessions or entire recommendation trees:

```bash
python src/visualization/recommendation_tree_viz.py \
  --cleaned-data data/cleaned_grail \
  --session-id 004QUceaM2cVUOrEO1iD \
  --max-steps 5 \
  --output docs/session_example.svg
```

Batch mode (`--batch-output-dir`, `--batch-issues`) produces per-issue session samples ready for documentation or inspection.
See [reports/visualized_recommendation_trees/README.md](reports/visualized_recommendation_trees/README.md) for curated exports and interpretation notes.

### 5. Automated Checks

- `scripts/run-lint.sh` – `pylint` with the repository root on `PYTHONPATH`.
- `scripts/run-tests.sh` – `pytest` for the unit test suite.
- `reports/build-reports.sh` – regenerates the published KNN/XGB reports from existing artifacts by calling the shared pipeline harness (`python -m knn.pipeline --stage reports` and `python -m xgb.pipeline --stage reports`). The pre-commit hook runs this entrypoint so report markdown stays current without retraining.
- `scripts/run-clean-data-suite.sh` – end-to-end dataset cleaning plus prompt/political-science replicas.
- `scripts/update-reports.sh` – aggregates the clean-data suite, KNN/XGB rebuilds, prompt samples, and GPT-4o pipeline so every report is current.
- Sphinx docs build (`sphinx-build -b html -n -W --keep-going docs docs/_build/html`) keeps the documentation green.
- GitHub Actions (see `.github/workflows/`) install `development/requirements-dev.txt`, invoke these scripts, and publish docs/report artifacts on push and PRs. The `Build Reports` workflow calls `reports/build-reports.sh`, which regenerates Markdown from checked-in sweep artifacts and bails out if metrics are missing so training never reruns in CI. Full script descriptions live in `scripts/README.md`.

Pytest markers in `development/pytest.ini` scope the suites that back the workflows above:

- `clean_data` - Dataset ingestion, filtering, and research article statistics.
- `filters` - Filter reporting helpers.
- `prompt_builder` - Prompt assembly utilities and package entry points.
- `prompt_smoke` - End-to-end prompt generation smoke tests.
- `sessions` - Session and slate construction helpers.
- `surveys` - Survey processing and allowlist checks.
- `knn` - k-NN feature extraction, index building, and inference.
- `xgb` - XGBoost baseline training and CLI helpers.
- `gpt4o` - GPT-4o conversation utilities.
- `open_r1` - Shared Open-R1 reinforcement learning helpers under `common.open_r1`.
- `integration` - Cross-package integration flows.
- `visualization` - Visualization and reporting helpers.

Run for example `pytest -m knn` to exercise only the k-NN pipeline.

### 6. Documentation

```bash
pip install -r development/requirements-dev.txt
make -C docs html
```

The HTML documentation (autodoc + autosummary for `clean_data`) lands in `docs/_build/html`; browse the latest build on [Read the Docs](https://grail-simulation.readthedocs.io/en/latest/index.html).

## End-to-End Pipeline

The repository stitches together several subsystems to turn raw CodeOcean logs into reproducible training/evaluation runs. The core stages are:

1. **Session ingestion & filtering** – `clean_data.sessions.build_codeocean_rows` loads the capsule exports, enforces participant allow-lists, and retains the full interaction history for every `(participant, issue)` pair. See `clean_data/sessions/README.md` for the module layout and helper reference.
2. **Prompt construction** – `clean_data.prompting.row_to_example` builds GRPO-style prompts, applying the shared viewer-profile logic used by downstream models.
3. **Model training & evaluation** – consult the package READMEs under `src/` (e.g., `src/knn/README.md`, `src/gpt4o/README.md`, `src/xgb/README.md`) for baseline-specific pipelines, CLI commands, and reporting details.

## Published Artifacts

- Minimum wage split: <https://huggingface.co/datasets/od2961/grail-wage>
- Gun control split: <https://huggingface.co/datasets/od2961/grail-gun>
- Prompt statistics (plots + Markdown): `reports/prompt_stats`

- GRPO trained on gun control data: <https://huggingface.co/od2961/Qwen2.5-1.5B-OpenR1-GRPO-GUN>
- GRPO training on minimum wage data: <https://huggingface.co/od2961/Qwen2.5-1.5B-OpenR1-GRPO>
- GRAIL trained on gun control data: <https://huggingface.co/od2961/Qwen2.5-1.5B-OpenR1-GRAIL-GUN>
- GRAIL trained on minimum wage data: <https://huggingface.co/od2961/Qwen2.5-1.5B-OpenR1-GRAIL-WAGE>

## Training

Launch the reinforcement-learning recipes via SLURM:

```bash
sbatch training/training-grail.sh    # RL + discriminator shaping (src/grail/grail.py)
sbatch training/training-grpo.sh     # RL baseline (src/grpo/grpo.py)
```

Baselines:

- GPT‑4o slate predictor: `python -m gpt4o.cli` for ad-hoc runs or `python -m gpt4o.pipeline` for sweeps + reporting.
- k-NN slate baseline: `python -m knn.pipeline --stage {plan,sweeps,finalize,reports}` (single runs via `python -m knn.cli ...`); `training/training-knn.sh` wraps the same CLI for SLURM.
- XGBoost slate + opinion pipeline: `python -m xgb.pipeline --stage {plan,sweeps,finalize,reports}` (single runs via `python -m xgb.cli ...`); `training/training-xgb.sh` wraps the same CLI for SLURM.

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
