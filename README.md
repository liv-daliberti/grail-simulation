# GRAIL Simulation

Grounded-Retrieval Adversarial Imitation Loop (GRAIL) is a framework for grounded human behavior simulation that unifies language, agent, and world models. The system retrieves realistic action slates, reasons about them with a ReAct-style language agent, predicts counterfactual outcomes, and aligns to real trajectories through adversarial training.

![GRAIL overview](docs/Simulation.drawio.png)

## Key Components

- **Environment Model** – retrieves candidate next actions from behavior logs to keep the agent grounded.
- **Action Model (ReAct)** – a language model selects among retrieved actions while reasoning about the current state.
- **Predictor / World Model** – estimates outcomes and counterfactuals for chosen actions.
- **Sequential Discriminator** – supplies adversarial rewards that align generated trajectories with real data.

## Repository Structure

```
.
├── clean_data/               # Utilities to convert the CodeOcean dump into tidy Hugging Face datasets
├── docs/                     # Figures used in the paper and README
├── recipes/                  # Training configuration files organised by model family
├── src/
│   ├── open_r1/              # Supervised fine-tuning + GRPO reinforcement learning trainers and helpers (see src/open_r1/README.md)
│   ├── gpt-4o/               # GPT-4o evaluation scripts for slate prediction baselines
│   ├── knn/                  # Non-generative k-nearest-neighbour baseline implementation
│   └── visualization/        # Graphviz-based recommendation tree and session visualisation tools
├── training/                 # SLURM launchers for GRPO (baseline) and GRAIL (with discriminator)
├── setup.py                  # Editable package definition (pip install -e .)
└── README.md                 # Project overview and setup instructions (this file)
```

The repository ships with the cleaned datasets referenced in the paper so you can
reproduce the experiments without re-running the CodeOcean extraction workflow.

## Pull the Data

All data comes from ```https://codeocean.com/capsule/5416997/tree/v1``` and is associated with the paper [Short-term exposure to filter-bubble recommendation systems has limited polarization effects: Naturalistic experiments on YouTube](https://www.pnas.org/doi/10.1073/pnas.2318127122). Should you want to manully reconstruct our data pull, you can pull the data using the following commands:
```
git clone https://git.codeocean.com/capsule-5416997.git
cd capsule-5416997
mkdir results
cd results
curl -fL -OJ 'https://codeocean-temp.s3.amazonaws.com/4644338a-0384-44ee-9ca5-7567a7a4afb8/5aa4b399-b9a3-4b37-bec9-f2a606ba4dbd?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJCBIX6WBZE5OXDDQ%2F20251017%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251017T144620Z&X-Amz-Expires=21600&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dresults-8447a699-902e-4f7b-ab6f-26fdb9726670.zip&X-Amz-Signature=47f190731d0d09dbc7218c2b480e5db9cf69d72457e47ba65626eadd185c2e9b'
unzip results-8447a699-902e-4f7b-ab6f-26fdb9726670.zip
cd ..
mkdir data
cd data
curl -fL --retry 5 --retry-all-errors -o capsule-5416997-data.zip 'https://codeocean-temp.s3.amazonaws.com/4644338a-0384-44ee-9ca5-7567a7a4afb8/2a642045-95ce-46a7-971f-c27f7d2ca15a?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJCBIX6WBZE5OXDDQ%2F20251017%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251017T153912Z&X-Amz-Expires=21600&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dcapsule-5416997-data.zip&X-Amz-Signature=008cdfa585998a2ce6c88c79d92c1ba0698bb78b02874a944cc1f05dd69b1e93'
```

This will give you a clean copy of the exact data used for the both the original paper and the data used for this study. We then reformat the data for our study / move it to a huggingface dataset for ease of use.

## Clean the Dataset

The GRPO training loop expects a tidy Hugging Face dataset containing prompts,
gold labels, and metadata for both policy domains (gun control and minimum wage).
The ``clean_data/clean_data.py`` utility produces exactly that schema from the
CodeOcean capsule, and validates that every output split is compatible with
``src/open_r1/grpo.py``.

> Want the full provenance playbook? See [clean_data/README.md](clean_data/README.md) for a line-by-line description of the filters we implement and how they reproduce the CodeOcean pipelines. (The cleaned dataset currently focuses on Studies 1–3; Study 4/Shorts is excluded because the public interaction log lacks recommendation slates.)

```bash
# 1) Create a virtual environment with the repository package
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
mkdir -p .cache/pip .cache/pip/build .tmp
export CONDA_ENVS_PATH="$(pwd)/.conda/envs"
export CONDA_PKGS_DIRS="$(pwd)/.conda/pkgs"
export CONDA_CACHE_DIR="$(pwd)/.cache/conda"
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$CONDA_CACHE_DIR"
export PIP_CACHE_DIR="$(pwd)/.cache/pip"
export PIP_BUILD_DIR="$(pwd)/.cache/pip/build"
export TMPDIR="$(pwd)/.tmp"
pip install -e .
# Hugging Face write access (required if you push checkpoints to the Hub)
export HF_TOKEN="hf_your_personal_token"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# 2) Generate the cleaned dataset
python clean_data/clean_data.py \
    --dataset-name capsule-5416997/data \
    --output-dir data/cleaned_grail \
    --prompt-stats-dir reports/prompt_stats

# (Optional) regenerate the prompt feature report later without re-cleaning
python clean_data/prompt_stats.py \
    --dataset data/cleaned_grail \
    --output-dir reports/prompt_stats

# Optional: push per-issue subsets (split by domain) to the Hub
python clean_data/clean_data.py \
    --dataset-name capsule-5416997/data \
    --output-dir data/cleaned_grail \
    --issue-repo gun_control=my-org/grail-gun \
    --issue-repo minimum_wage=my-org/grail-wage \
    --push-to-hub --hub-token $HF_TOKEN
```

Builder notes:
- Rows missing all survey demographics (age, gender, race, income, etc.) are dropped so every retained interaction includes viewer context (roughly 22% of the raw interactions are removed).
- Each participant is allowed at most one session per issue (gun control vs. minimum wage). Duplicate sessions for the same participant/issue pair are filtered out during cleaning.

The script logs the number of rows kept per split, along with the per-issue
distribution, and raises an error if any required GRPO columns are missing.
When a validation split is requested, the builder assigns entire viewers
(`urlid`/`session_id`) to a single split, so no person appears in both train and
validation.

Public Hugging Face copies of the cleaned splits:
- Minimum wage: https://huggingface.co/datasets/od2961/grail-wage
- Gun control: https://huggingface.co/datasets/od2961/grail-gun

##
Builder note: rows missing all survey demographics (age, gender, race, income, etc.) are dropped during cleaning so every retained interaction has viewer context for the prompt builder. This removes roughly 22% of the ~33k raw interactions.
Builder note: each participant is restricted to a single session per issue (gun control or minimum wage). Duplicate sessions from the same viewer on the same issue are filtered out by combining the URLID or anonymous Firebase ID with the issue label.
# What Each Row Contains

- Viewer survey attributes from the underlying PNAS study (three waves, 300+ columns) merged by `urlid`/`topic_id`.
- Complete watch history and timing context (`watched_detailed_json`, `trajectory_json`) for each decision point.
- Enriched video metadata for the current, recommended, and chosen videos—titles, channel names, IDs, durations, engagement counts—with explicit `*_missing` flags when the capsule lacked a field.
- Provenance markers (`issue_source`, `issue_detail`, `slate_source`) describing how we inferred the issue label and where the slate was sourced.
- A `participant_study` tag aligning each interaction with the original PNAS study cohort (Study 1 gun rights, Studies 2–4 minimum wage).
- A preview of the first two cleaned rows in the build log so you can sanity-check the dataset before training.

## Visualize Recommendation Trees

The repository ships with `src/visualization/recommendation_tree_viz.py`, a small CLI for turning the Gun Rights & Minimum Wage recommendation tree CSVs into Graphviz diagrams. Supply a single tree CSV, optionally merge in per-video metadata for nicer labels, and the tool will emit a `.png`, `.svg`, or any other Graphviz-supported format.

Additional options let you highlight a specific viewing path (`--highlight`), overlay aggregated trajectories to annotate edges with viewer counts (`--trajectories`), limit the depth of the render, or switch the layout direction.

To render a real viewer session from the cleaned dataset (showing the slate at each step and the chosen path):

```bash
python src/visualization/recommendation_tree_viz.py \
  --cleaned-data data/cleaned_grail \
  --session-id 004QUceaM2cVUOrEO1iD \
  --max-steps 5 \
  --output docs/session_example.svg
```

The session visualization highlights the core RL structure:
- **State** (ellipse): the video currently being watched.
- **Actions** (rounded boxes): the recommended videos available for the next choice, with the selected action highlighted in green.

![Minimum wage session example](docs/session_example.svg)

For gun-control sessions the structure is identical:

![Gun session example](docs/batch_sessions/grail_session_gun_control_1.svg)

Each dataset row also carries the viewer profile that accompanies the decision. The `viewer_profile` field distills age, gender, race, media habits, and gun/wage attitudes, while `state_text` expands that information alongside the full watch history. These columns can be surfaced in the visualization template when you want demographic context next to the state nodes—e.g. a row might read “44-year-old, man, Black or African-American (non-Hispanic), democrat liberal, $150,000 or more, college-educated, watches YouTube weekly.”

To export multiple session trajectories at once, point the CLI at the cleaned dataset and a destination directory. The tool will emit distinct session graphs grouped by issue, defaulting to two minimum-wage and two gun-control trajectories:

```bash
python src/visualization/recommendation_tree_viz.py \
  --cleaned-data data/cleaned_grail \
  --max-steps 5 \
  --batch-output-dir docs/batch_sessions \
  --batch-prefix grail_session \
  --format svg
```

The command above produces four `.svg` files such as `grail_session_minimum_wage_1.svg` and `grail_session_gun_control_2.svg`. Override `--batch-issues` (e.g. `minimum_wage=3,gun_control=1`) to customise the mix or file counts.

## Training

1. Ensure the editable package and cache directories are set up (see “Clean the dataset” instructions).
2. Submit training jobs via SLURM:
   ```bash
   sbatch training/training-grail.sh    # GRPO + discriminator shaping (uses src/open_r1/grail.py)
   sbatch training/training-grpo.sh     # Vanilla GRPO baseline (uses src/open_r1/grpo.py)
   ```
   Update the scripts or recipe paths if you place custom configs under `recipes/`.
3. Evaluate baselines locally:
   - GPT‑4o slate baseline: `python src/gpt-4o/gpt-4o-baseline.py`
   - KNN slate baseline: `bash training/training-knn.sh`

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
This repository contains code released under the Apache 2.0 license. See individual files for details.
