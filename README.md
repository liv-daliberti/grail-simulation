# GRAIL Simulation

Grounded-Retrieval Adversarial Imitation Loop (GRAIL) is a framework for grounded human behavior simulation that unifies language, agent, and world models. The system retrieves realistic action slates, reasons about them with a ReAct-style language agent, predicts counterfactual outcomes, and aligns to real trajectories through adversarial training.

![GRAIL overview](docs/Simulation.drawio.png)

## Key Components

- **Environment Model** – retrieves candidate next actions from behavior logs to keep the agent grounded.
- **Action Model (ReAct)** – a language model selects among retrieved actions while reasoning about the current state.
- **Predictor / World Model** – estimates outcomes and counterfactuals for chosen actions.
- **Sequential Discriminator** – supplies adversarial rewards that align generated trajectories with real data.

## Repository Structure

- `src/open_r1/` – training utilities for supervised fine-tuning and GRPO-style reinforcement learning, including a simplified GRPO trainer specialised for GRAIL.
- `src/gpt-4o/` – baseline script to evaluate GPT‑4o on next‑video choice using the same prompts as the GRPO setup.
- `src/knn/` – non‑generative k‑nearest neighbours baseline for slate‑constrained prediction.
- `recipes/` – configuration files for different model sizes and training modes.
- `training-grail.sh` – convenience script to launch GRPO training with preset configs.

## Quick Start

1. Install Python dependencies and system packages (e.g. `graphviz` for diagram generation).
2. Choose a model recipe from `recipes/` and launch training:
   ```bash
   bash training-grail.sh recipes/Qwen2.5-1.5B-Instruct/grpo/config_grail.yaml
   ```
3. Evaluate baselines:
   - GPT‑4o: `python src/gpt-4o/gpt-4o-baseline.py`
   - KNN: `python src/knn/knn-baseline.py`


## Pull the Data

All data comes from ```https://codeocean.com/capsule/5416997/tree/v1``` and is associated with the paper [Short-term exposure to filter-bubble recommendation systems has limited polarization effects: Naturalistic experiments on YouTube](https://www.pnas.org/doi/10.1073/pnas.2318127122).
```
git clone https://git.codeocean.com/capsule-5416997.git
cd capsule-5416997 
curl -fL -OJ 'https://codeocean-temp.s3.amazonaws.com/4644338a-0384-44ee-9ca5-7567a7a4afb8/5aa4b399-b9a3-4b37-bec9-f2a606ba4dbd?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJCBIX6WBZE5OXDDQ%2F20251017%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251017T144620Z&X-Amz-Expires=21600&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dresults-8447a699-902e-4f7b-ab6f-26fdb9726670.zip&X-Amz-Signature=47f190731d0d09dbc7218c2b480e5db9cf69d72457e47ba65626eadd185c2e9b'
unzip results-8447a699-902e-4f7b-ab6f-26fdb9726670.zip
curl -fL --retry 5 --retry-all-errors -o capsule-5416997-data.zip 'https://codeocean-temp.s3.amazonaws.com/4644338a-0384-44ee-9ca5-7567a7a4afb8/2a642045-95ce-46a7-971f-c27f7d2ca15a?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJCBIX6WBZE5OXDDQ%2F20251017%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251017T153912Z&X-Amz-Expires=21600&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dcapsule-5416997-data.zip&X-Amz-Signature=008cdfa585998a2ce6c88c79d92c1ba0698bb78b02874a944cc1f05dd69b1e93'
```

This will give you a clean copy of the exact data used for the both the original paper and the data used for this study. We include a copy of the data within this repo too for full transparency /reproducibility. 

We then reformat the data for our study / move it to a huggingface dataset for ease of use.

## Clean the Dataset for GRPO

The GRPO training loop expects a tidy Hugging Face dataset containing prompts,
gold labels, and metadata for both policy domains (gun control and minimum wage).
The ``clean_data/clean_data.py`` utility produces exactly that schema from the
CodeOcean capsule, and validates that every output split is compatible with
``src/open_r1/grpo.py``.

```bash
# 1) Create a virtual environment with the repository requirements
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Generate the cleaned dataset
python clean_data/clean_data.py \
    --dataset-name capsule-5416997/data \
    --output-dir data/cleaned_grail

# Optional: push per-issue subsets (split by domain) to the Hub
python clean_data/clean_data.py \
    --dataset-name capsule-5416997/data \
    --output-dir data/cleaned_grail \
    --issue-repo gun_control=my-org/grail-gun \
    --issue-repo minimum_wage=my-org/grail-wage \
    --push-to-hub --hub-token $HF_TOKEN
```

The script logs the number of rows kept per split, along with the per-issue
distribution, and raises an error if any required GRPO columns are missing.

## Citation


## License
This repository contains code released under the Apache 2.0 license. See individual files for details.
