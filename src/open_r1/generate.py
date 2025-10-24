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

from dataclasses import dataclass
from typing import Any, Dict, Optional

DISTILABEL_IMPORT_ERROR: Optional[Exception]
try:  # pragma: no cover - optional dependency
    from distilabel.llms import OpenAILLM
    from distilabel.pipeline import Pipeline
    from distilabel.steps import StepResources
    from distilabel.steps.tasks import TextGeneration
except ImportError as exc:  # pragma: no cover - handled via helper
    OpenAILLM = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]
    StepResources = None  # type: ignore[assignment]
    TextGeneration = None  # type: ignore[assignment]
    DISTILABEL_IMPORT_ERROR = exc
else:
    DISTILABEL_IMPORT_ERROR = None


def _require_distilabel() -> tuple[Any, Any, Any, Any]:
    """Return distilabel helpers or raise an informative installation error."""

    if DISTILABEL_IMPORT_ERROR is not None:
        raise ImportError(
            "distilabel is required to build generation pipelines. "
            "Install it with `pip install distilabel`."
        ) from DISTILABEL_IMPORT_ERROR
    assert Pipeline is not None
    assert OpenAILLM is not None
    assert StepResources is not None
    assert TextGeneration is not None
    return Pipeline, OpenAILLM, StepResources, TextGeneration


@dataclass
class DistilabelPipelineConfig:
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
    """Construct a distilabel pipeline configured for OpenAI-compatible endpoints."""

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


if __name__ == "__main__":
    import argparse

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'datasets' package is required to run open_r1.generate as a script. "
            "Install it with `pip install datasets`."
        ) from exc

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

    args = parser.parse_args()

    print("\nRunning with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    print(
        f"Loading '{args.hf_dataset}' "
        f"(config: {args.hf_dataset_config}, split: {args.hf_dataset_split}) dataset..."
    )
    dataset = load_dataset(
        args.hf_dataset,
        args.hf_dataset_config,
        split=args.hf_dataset_split,
    )
    print("Dataset loaded!")

    pipeline_config = DistilabelPipelineConfig(
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
    pipeline = build_distilabel_pipeline(
        model=args.model,
        config=pipeline_config,
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
