#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for constructing distilabel pipelines for data generation."""

import argparse
import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

_distilabel_cache: Dict[str, Any] = {"imports": None, "error": None}


def _require_distilabel() -> tuple[Any, Any, Any, Any]:
    """Return distilabel helpers or raise an informative installation error.

    :returns: Tuple containing ``Pipeline``, ``OpenAILLM``, ``StepResources``, and
        ``TextGeneration`` classes.
    :raises ImportError: If the ``distilabel`` dependency cannot be imported.
    """

    cache = _distilabel_cache
    cached_imports = cache.get("imports")
    if cached_imports is not None:
        return cached_imports

    cached_error = cache.get("error")
    if cached_error is not None:
        raise ImportError(
            "distilabel is required to build generation pipelines. "
            "Install it with `pip install distilabel`."
        ) from cached_error

    try:
        pipeline_module = importlib.import_module("distilabel.pipeline")
        llms_module = importlib.import_module("distilabel.llms")
        steps_module = importlib.import_module("distilabel.steps")
        tasks_module = importlib.import_module("distilabel.steps.tasks")
    except ImportError as exc:  # pragma: no cover - handled via helper
        cache["error"] = exc
        raise ImportError(
            "distilabel is required to build generation pipelines. "
            "Install it with `pip install distilabel`."
        ) from exc

    cache["imports"] = (
        pipeline_module.Pipeline,
        llms_module.OpenAILLM,
        steps_module.StepResources,
        tasks_module.TextGeneration,
    )
    return cache["imports"]


def _require_datasets_loader():
    """Return ``datasets.load_dataset`` or raise an informative error.

    :returns: Callable for loading Hugging Face datasets.
    :raises ImportError: If the ``datasets`` package is not installed.
    """

    try:
        return importlib.import_module("datasets").load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'datasets' package is required to run open_r1.generate as a script. "
            "Install it with `pip install datasets`."
        ) from exc


@dataclass
class DistilabelPipelineConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration options used to assemble a distilabel pipeline."""

    base_url: str = "http://localhost:8000/v1"
    prompt_column: Optional[str] = None
    prompt_template: str = "{{ instruction }}"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: int = 8192
    num_generations: int = 1
    input_batch_size: int = 64
    client_replicas: int = 1
    timeout: int = 900
    retries: int = 0

    def build_generation_kwargs(self) -> Dict[str, Any]:
        """
        Return keyword arguments forwarded to the distilabel text generation step.

        The method centralises the logic for handling optional sampling parameters so
        callers keep the outer function signature small enough for linting.

        :returns: Dictionary of keyword arguments passed to ``TextGeneration``.
        """

        kwargs: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        return kwargs


def build_distilabel_pipeline(
    model: str,
    config: Optional[DistilabelPipelineConfig] = None,
) -> "Pipeline":
    """Construct a distilabel pipeline configured for OpenAI-compatible endpoints.

    :param model: Model identifier accessible via an OpenAI-compatible endpoint.
    :param config: Optional pipeline configuration override.
    :returns: Initialised distilabel ``Pipeline`` ready to execute.
    """

    cfg = config or DistilabelPipelineConfig()
    pipeline_cls, openai_llm_cls, step_resources_cls, text_generation_cls = _require_distilabel()

    generation_kwargs = cfg.build_generation_kwargs()
    input_mapping = {"instruction": cfg.prompt_column} if cfg.prompt_column is not None else {}

    with pipeline_cls().ray() as pipeline_obj:
        text_generation_cls(
            llm=openai_llm_cls(
                base_url=cfg.base_url,
                api_key="something",
                model=model,
                timeout=cfg.timeout,
                max_retries=cfg.retries,
                generation_kwargs=generation_kwargs,
            ),
            template=cfg.prompt_template,
            input_mappings=input_mapping,
            input_batch_size=cfg.input_batch_size,
            num_generations=cfg.num_generations,
            group_generations=True,
            resources=step_resources_cls(replicas=cfg.client_replicas),
        )

    return pipeline_obj


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser used when running the module as a script.

    :returns: Configured :class:`argparse.ArgumentParser` for the CLI entrypoint.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run a distilabel pipeline for generating responses with the DeepSeek R1 model"
        )
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
        help="HuggingFace dataset to load",
    )
    parser.add_argument(
        "--hf-dataset-config",
        type=str,
        required=False,
        help="Dataset config to use",
    )
    parser.add_argument(
        "--hf-dataset-split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="{{ instruction }}",
        help="Template string for formatting prompts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for generation",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the vLLM server",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p value for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="Number of generations per problem",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=64,
        help="Batch size for input processing",
    )
    parser.add_argument(
        "--client-replicas",
        type=int,
        default=1,
        help="Number of client replicas for parallel processing",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of retries for failed requests (default: 0)",
    )
    parser.add_argument(
        "--hf-output-dataset",
        type=str,
        required=False,
        help="HuggingFace repo to push results to",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether to make the output dataset private when pushing to HF Hub",
    )
    return parser


def _create_pipeline_config(args: argparse.Namespace) -> DistilabelPipelineConfig:
    """Convert CLI arguments into a :class:`DistilabelPipelineConfig` instance.

    :param args: Parsed CLI arguments.
    :returns: Populated :class:`DistilabelPipelineConfig`.
    """
    return DistilabelPipelineConfig(
        base_url=args.vllm_server_url,
        prompt_column=args.prompt_column,
        prompt_template=args.prompt_template,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations,
        input_batch_size=args.input_batch_size,
        client_replicas=args.client_replicas,
        timeout=args.timeout,
        retries=args.retries,
    )


def _log_cli_arguments(args: argparse.Namespace) -> None:
    """Print the CLI arguments for transparency in CLI usage.

    :param args: Parsed CLI arguments to display.
    :returns: ``None``. Writes human-readable summaries to ``stdout``.
    """
    print("\nRunning with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    print(
        f"Loading '{args.hf_dataset}' "
        f"(config: {args.hf_dataset_config}, split: {args.hf_dataset_split}) dataset..."
    )


def main() -> None:
    """Entry point for running the module as a CLI script.

    :returns: ``None``. Runs the distilabel pipeline for its side effects.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()
    _log_cli_arguments(args)

    dataset = _require_datasets_loader()(
        args.hf_dataset,
        args.hf_dataset_config,
        split=args.hf_dataset_split,
    )
    print("Dataset loaded!")

    pipeline = build_distilabel_pipeline(
        model=args.model,
        config=_create_pipeline_config(args),
    )

    print("Running generation pipeline...")
    distiset = pipeline.run(
        dataset=dataset,
        dataset_batch_size=args.input_batch_size * 1000,
        use_cache=False,
    )
    print("Generation pipeline finished!")

    if args.hf_output_dataset:
        print(f"Pushing resulting dataset to '{args.hf_output_dataset}'...")
        distiset.push_to_hub(args.hf_output_dataset, private=args.private)
        print("Dataset pushed!")


if __name__ == "__main__":
    main()
