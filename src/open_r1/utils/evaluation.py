"""Evaluation utilities for orchestrating LightEval benchmark runs."""

import base64
import logging
import os
import subprocess
from typing import TYPE_CHECKING, Dict, Union

from .hub import get_gpu_count_for_vllm, get_param_count_from_repo_id


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from trl import GRPOConfig, SFTConfig, ModelConfig


# We need a special environment setup to launch vLLM from within Slurm training jobs.
# - Reference code: huggingface/brrr@c55ba35/brrr/lighteval/one_job_runner.py#L105
# - Slack thread:
#   https://huggingface.slack.com/archives/C043JTYE1MJ/p1726566494958269
user_home_directory = os.path.expanduser("~")
VLLM_SLURM_PREFIX = [
    "env",
    "-i",
    "bash",
    "-c",
    (
        "for f in /etc/profile.d/*.sh; do source $f; done; "
        f"export HOME={user_home_directory}; sbatch "
    ),
]


def register_lighteval_task(
    configs: Dict[str, str],
    eval_suite: str,
    task_name: str,
    task_list: str,
    num_fewshot: int = 0,
):
    """Register a LightEval task configuration.

    Notes:
        - Core tasks can be added from this table:
          https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks_table.jsonl
        - Custom tasks requiring bespoke metrics or scripts should live in
          `scripts/evaluation/extended_lighteval_tasks`.

    Args:
        configs: Dictionary used to store the task configuration.
        eval_suite: Evaluation suite to pull tasks from.
        task_name: Display name for the task.
        task_list: Comma-separated task list using the LightEval format:
            ``extended|{task_name}|{num_fewshot}|0`` or
            ``lighteval|{task_name}|{num_fewshot}|0``.
        num_fewshot: Number of few-shot examples.
    """
    # Format task list in lighteval format
    task_list = ",".join(
        f"{eval_suite}|{task}|{num_fewshot}|0" for task in task_list.split(",")
    )
    configs[task_name] = task_list


LIGHTEVAL_TASKS = {}

register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "math_500", "math_500", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "aime24", "aime24", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "aime25", "aime25", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "gpqa", "gpqa:diamond", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb", "lcb:codegeneration", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb_v4", "lcb:codegeneration_v4", 0)


def get_lighteval_tasks() -> list[str]:
    """Return the list of registered LightEval task identifiers."""
    return list(LIGHTEVAL_TASKS.keys())


SUPPORTED_BENCHMARKS = get_lighteval_tasks()


def run_lighteval_job(
    benchmark: str,
    training_args: Union["SFTConfig", "GRPOConfig"],
    model_args: "ModelConfig",
) -> None:
    """Submit a LightEval benchmark job via SLURM."""
    task_list = LIGHTEVAL_TASKS[benchmark]
    model_name = training_args.hub_model_id
    model_revision = training_args.hub_model_revision
    # For large models (>=30B params) or for the MATH benchmark we shard across GPUs
    # to avoid OOM.
    num_gpus = get_gpu_count_for_vllm(model_name, model_revision)
    if get_param_count_from_repo_id(model_name) >= 30_000_000_000:
        tensor_parallel = True
    else:
        num_gpus = 2  # Hack while cluster is full
        tensor_parallel = False

    cmd = VLLM_SLURM_PREFIX.copy()
    cmd_args = [
        f"--gres=gpu:{num_gpus}",
        (
            "--job-name="
            f"or1_{benchmark}_{model_name.split('/')[-1]}_{model_revision}"
        ),
        "slurm/evaluate.slurm",
        benchmark,
        f'"{task_list}"',
        model_name,
        model_revision,
        f"{tensor_parallel}",
        f"{model_args.trust_remote_code}",
    ]
    if training_args.system_prompt is not None:
        # encode to base64 to avoid issues with special characters
        # we decode in the sbatch script
        prompt_encoded = base64.b64encode(training_args.system_prompt.encode()).decode()
        cmd_args.append(prompt_encoded)
    cmd[-1] += " " + " ".join(cmd_args)
    subprocess.run(cmd, check=True)


def run_benchmark_jobs(
    training_args: Union["SFTConfig", "GRPOConfig"],
    model_args: "ModelConfig",
) -> None:
    """Launch benchmark jobs according to the training configuration."""
    benchmarks = training_args.benchmarks
    if len(benchmarks) == 1 and benchmarks[0] == "all":
        benchmarks = get_lighteval_tasks()
        # Evaluate on all supported benchmarks. Later we may want to include a `chat` option
        # that just evaluates on `ifeval` and `mt_bench` etc.

    for benchmark in benchmarks:
        logger.info("Launching benchmark `%s`", benchmark)
        if benchmark in get_lighteval_tasks():
            run_lighteval_job(benchmark, training_args, model_args)
        else:
            raise ValueError(f"Unknown benchmark {benchmark}")
