"""Sphinx configuration for the GRAIL Simulation project."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
from datetime import datetime

# Make the project root and src directory importable so autodoc can find modules.
ROOT_DIR = os.path.abspath("..")
SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _ensure_trl_stub() -> None:
    """Provide a minimal ``trl`` stub so autodoc can import open_r1 modules."""
    if os.environ.get("SPHINX_USE_REAL_TRL"):
        return

    sys.modules.pop("trl", None)
    trl_module = types.ModuleType("trl")

    class _Stub:
        """Placeholder object used for optional TRL types."""

        def __init__(self, *args, **kwargs):
            pass

    def _make_stub_class(name: str, namespace: dict[str, object] | None = None):
        namespace = namespace or {}
        namespace.setdefault("__module__", "trl")
        return type(name, (_Stub,), namespace)

    class GRPOTrainer(_Stub):
        __module__ = "trl"

        def add_callback(self, *args, **kwargs):
            return None

    trl_module.GRPOTrainer = GRPOTrainer
    trl_module.ScriptArguments = _make_stub_class("ScriptArguments")
    trl_module.GRPOConfig = _make_stub_class("GRPOConfig")
    trl_module.SFTConfig = _make_stub_class("SFTConfig")
    trl_module.ModelConfig = _make_stub_class("ModelConfig")

    class TrlParser(_Stub):
        __module__ = "trl"

        @staticmethod
        def parse_args(*_args, **_kwargs):
            return {}

    trl_module.TrlParser = TrlParser

    def _noop(*_args, **_kwargs):
        return None

    trl_module.get_peft_config = _noop
    trl_module.get_kbit_device_map = _noop
    trl_module.get_quantization_config = _noop

    extras_module = types.ModuleType("trl.extras")
    profiling_module = types.ModuleType("trl.extras.profiling")
    profiling_module.profiling_context = contextlib.nullcontext
    extras_module.profiling = profiling_module

    trainer_module = types.ModuleType("trl.trainer")
    grpo_trainer_module = types.ModuleType("trl.trainer.grpo_trainer")
    grpo_trainer_module.unwrap_model_for_generation = _noop
    grpo_trainer_module.pad = _noop
    grpo_trainer_module.GRPOTrainer = GRPOTrainer
    trainer_module.grpo_trainer = grpo_trainer_module

    data_utils_module = types.ModuleType("trl.data_utils")
    data_utils_module.maybe_apply_chat_template = _noop
    data_utils_module.is_conversational = lambda *_args, **_kwargs: False

    sys.modules["trl"] = trl_module
    sys.modules["trl.extras"] = extras_module
    sys.modules["trl.extras.profiling"] = profiling_module
    sys.modules["trl.trainer"] = trainer_module
    sys.modules["trl.trainer.grpo_trainer"] = grpo_trainer_module
    sys.modules["trl.data_utils"] = data_utils_module


_ensure_trl_stub()

project = "GRAIL Simulation"
author = "GRAIL Simulation Team"
year = datetime.utcnow().year
copyright = f"{year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_heading_anchors = 3

autodoc_mock_imports = [
    "accelerate",
    "aiofiles",
    "async_lru",
    "bitsandbytes",
    "datasets",
    "deepspeed",
    "distilabel",
    "dotenv",
    "einops",
    "graphviz",
    "hf_transfer",
    "huggingface_hub",
    "jieba",
    "langdetect",
    "gensim",
    "gensim.models",
    "gensim.models.word2vec",
    "latex2sympy2_extended",
    "liger_kernel",
    "lighteval",
    "math_verify",
    "matplotlib",
    "numpy",
    "openai",
    "pandas",
    "peft",
    "sentence_transformers",
    "pyarrow",
    "pydantic",
    "safetensors",
    "scipy",
    "scikit_learn",
    "sklearn",
    "sentencepiece",
    "torch",
    "transformers",
    "vllm",
    "wandb",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_type_aliases = {
    # stdlib/concurrency helpers
    "Future": "concurrent.futures.Future",
    # Prefer canonical common.pipeline.* anchors for core generics
    "StudySpec": "common.pipeline.types.StudySpec",
    "StudySelection": "common.pipeline.types.StudySelection",
    "OpinionStudySelection": "common.pipeline.types.OpinionStudySelection",
    # Do not alias Sweep* names globally to avoid conflicts with GPT-4o
    # pipeline types; let Sphinx resolve by actual module.
    "BasePipelineSweepOutcome": "common.pipeline.types.BasePipelineSweepOutcome",
    "BaseOpinionSweepOutcome": "common.opinion.sweep_types.BaseOpinionSweepOutcome",
    # Report bundles exposed via knn.pipeline.context
    "ReportSelections": "knn.pipeline.context.ReportSelections",
    "ReportOutcomes": "knn.pipeline.context.ReportOutcomes",
    "ReportMetrics": "knn.pipeline.context.ReportMetrics",
    "ReportPresentation": "knn.pipeline.context.ReportPresentation",
    # XGBoost config dataclasses (canonical path under core.model_config)
    "XGBoostBoosterParams": "xgb.core.model_config.XGBoostBoosterParams",
    "XGBoostTrainConfig": "xgb.core.model_config.XGBoostTrainConfig",
    # KNN pipeline private bundles referenced in type hints
    "_PipelinePaths": "knn.pipeline.context_pipeline._PipelinePaths",
    "_ModelDefaults": "knn.pipeline.context_pipeline._ModelDefaults",
    "_Workflow": "knn.pipeline.context_pipeline._Workflow",
}
autodoc_typehints_format = "fully-qualified"
python_use_unqualified_type_names = False
napoleon_include_init_with_doc = True
napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

html_static_path = ["_static"]
# Copy project report artifacts into the built site so that included Markdown
# under `reports/` can reference images directly (e.g., ../reports/xgb/opinion/*).
html_extra_path = ["../reports"]
if (
    os.environ.get("SPHINX_ENABLE_FURO", "").lower() in {"1", "true", "yes"}
    or importlib.util.find_spec("furo") is not None
):
    html_theme = "furo"
    html_theme_options = {
        "light_css_variables": {
            "color-brand-primary": "#1f4b8e",
            "color-brand-content": "#1f4b8e",
        },
        "dark_css_variables": {
            "color-brand-primary": "#8bbcef",
            "color-brand-content": "#8bbcef",
        },
    }
else:
    html_theme = "alabaster"
    html_theme_options = {}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable/", None),
    "datasets": ("https://huggingface.co/docs/datasets/main/en", None),
}

if os.environ.get("SPHINX_ENABLE_INTERSPHINX", "").lower() not in {"1", "true", "yes"}:
    # Avoid noisy warnings when building docs in restricted or offline environments.
    intersphinx_mapping = {}

_nitpick_targets = {
    "py:class": [
        # Typing artifacts introduced by forward-referenced type aliases in 3.12
        "TypeAliasForwardRef",
        "DatasetDict",
        "Execution",
        "ModelConfig",
        "Path",
        "argparse.ArgumentParser",
        "argparse.Namespace",
        "Namespace",
        "CliRunner",
        "click.testing.CliRunner",
        "abc.ABC",
        "collections.Counter",
        "common.text.title_index.TitleResolver",
        "common.evaluation.slate_eval.Observation",
        "common.evaluation.slate_eval.SlateMetricsRequest",
        "common.evaluation.slate_eval.EvaluationAccumulator",
        "common.pipeline.utils.StudyT",
        "common.pipeline.utils.OutcomeT",
        "common.opinion.results.OpinionEvaluationResult",
        "gpt4o.core.opinion.OpinionEvaluationResult",
        "knn.pipeline.context.OpinionSweepOutcome",
        "knn.pipeline.context.SweepOutcome",
        "OpinionSweepOutcome",
        "SweepOutcome",
        "knn.pipeline.utils.TaskCachePartition",
        "grpo.next_video.NextVideoEvaluationResult",
        "grpo.opinion.OpinionEvaluationResult",
        "OpinionSpec",
        "OpinionExampleInputs",
        "ExampleT",
        "T",
        "TitleLookup",
        "concurrent.futures._base.Future",
        "concurrent.futures.Future",
        "datasets.Dataset",
        "datasets.DatasetDict",
        "distilabel.pipeline.Pipeline",
        "graphviz.Digraph",
        "logging.Logger",
        "knn.index.SlateQueryConfig",
        "numpy.ndarray",
        "openai.AzureOpenAI",
        "optional",
        "pandas.DataFrame",
        "pandas.Series",
        "pd.DataFrame",
        "pathlib.Path",
        "sequence-like",
        "sklearn.feature_extraction.text.TfidfVectorizer",
        "sklearn.preprocessing.LabelEncoder",
        "SentenceTransformer",
        "torch.device",
        "torch.utils.data.Dataset",
        "transformers.AutoModelForCausalLM",
        "transformers.PreTrainedModel",
        "transformers.PreTrainedTokenizer",
        "transformers.PreTrainedTokenizerBase",
        "transformers.TrainerCallback",
        "transformers.TrainerControl",
        "transformers.TrainerState",
        "transformers.TrainingArguments",
        "transformers.generation.utils.GenerationMixin",
        "trl.GRPOConfig",
        "trl.ModelConfig",
        "trl.SFTConfig",
        "trl.ScriptArguments",
        "trl._ensure_trl_stub.<locals>.GRPOTrainer",
        "xgboost.XGBClassifier",
        "rewards where",
        "Word2VecConfig",
        "Future",
        "gensim.models.Word2Vec",
        "knn.core.index.SlateQueryConfig",
        "knn.core.evaluate.pipeline.IssueEvaluationRequest",
        "NearestNeighbors",
        "sklearn.neighbors.NearestNeighbors",
        "common.prompts.selection.CandidateMetadata",
        "knn.features.Word2VecConfig",
        "Pipeline",
        "AutoModelForCausalLM",
        "PreTrainedTokenizer",
        "VideoStats",
        "TreeData",
        "common.open_r1.torch_stub_utils.build_torch_stubs.<locals>._ModuleStub",
        "common.open_r1.torch_stub_utils.build_torch_stubs.<locals>.<lambda>",
        "common.pipeline.types.StudySelection",
        "common.pipeline.types.StudySpec",
        "knn.pipeline.context.StudySpec",
        "common.pipeline.types.BasePipelineSweepOutcome",
        "common.opinion.sweep_types.BaseOpinionSweepTask",
        "common.opinion.sweep_types.BaseOpinionSweepOutcome",
        "common.opinion.sweep_types.ConfigT",
        "common.opinion.sweep_types.MetricsArtifact",
        "common.opinion.sweep_types.AccuracySummary",
        "common.opinion.models.OpinionCalibrationMetrics",
        # TypeVar references surfaced in autodoc across modules
        "common.opinion.models.ExampleT",
        "common.pipeline.types.ConfigT",
        "common.pipeline.types.OutcomeT",
        "common.opinion.sweep_types.ConfigT",
        "ConfigT",
        "ExampleT",
        # Generic type variables often rendered as classes by Sphinx
        "knn.pipeline.utils.TaskT",
        "knn.pipeline.utils.OutcomeT",
        "common.pipeline.utils.OutcomeT",
        "TaskT",
        "OutcomeT",
        # Private helper classes referenced in type hints/docstrings
        "xgb.core.model._BoosterCore",
        "xgb.core.model._BoosterSampling",
        "xgb.core.model._BoosterRegularization",
        "xgb.core.model._TrainVectorizers",
        # Private helpers moved to model_config
        "xgb.core.model_config._BoosterCore",
        "xgb.core.model_config._BoosterSampling",
        "xgb.core.model_config._BoosterRegularization",
        "xgb.core.model_config._TrainVectorizers",
        "xgb.pipeline.context._NextVideoCore",
        "xgb.pipeline.context._NextVideoMeta",
        "xgb.pipeline.context.metrics_next_video._NextVideoCore",
        "xgb.pipeline.context.metrics_next_video._NextVideoMeta",
        "xgb.pipeline.context._OpinionAfter",
        "xgb.pipeline.context._OpinionBaseline",
        "xgb.pipeline.context._OpinionCalibration",
        "xgb.pipeline.context._OpinionDeltas",
        "xgb.pipeline.context._OpinionMeta",
        "xgb.pipeline.context.opinion_summary._OpinionAfter",
        "xgb.pipeline.context.opinion_summary._OpinionBaseline",
        "xgb.pipeline.context.opinion_summary._OpinionCalibration",
        "xgb.pipeline.context.opinion_summary._OpinionDeltas",
        "xgb.pipeline.context.opinion_summary._OpinionMeta",
        # KNN pipeline private internals referenced from public docs
        "knn.pipeline.context_pipeline._PipelinePaths",
        "knn.pipeline.context_pipeline._ModelDefaults",
        "knn.pipeline.context_pipeline._Workflow",
        "knn.pipeline.context_reports._PredictionRoots",
        "knn.pipeline.context_reports._PresentationFlags",
        "knn.pipeline.context_reports._ReportMetrics",
        "knn.pipeline.context_reports._ReportOutcomes",
        "knn.pipeline.context_reports._ReportPresentation",
        "knn.pipeline.context_reports._ReportSelections",
        # Re-exported private bundles also show up under the aggregator module
        "knn.pipeline.context._PredictionRoots",
        "knn.pipeline.context._PresentationFlags",
        "knn.pipeline.context._ReportMetrics",
        "knn.pipeline.context._ReportOutcomes",
        "knn.pipeline.context._ReportPresentation",
        "knn.pipeline.context._ReportSelections",
        "knn.pipeline.context_sweeps._KnnSweepStats",
        "knn.pipeline.context_sweeps._KnnOpinionExtras",
    ],
    "py:func": [
        "load_participant_allowlists",
        "load_dataset_any",
        "datasets.load_dataset",
        "generate_research_article_report",
        "argparse.ArgumentParser.add_argument",
        "argparse.ArgumentParser.set_defaults",
        "build_user_prompt",
        "get_logger",
        "knn.cli.build_parser",
        "collect_selected_examples",
        "write_issue_outputs",
    ],
    "py:mod": [
        "xgboost",
        "datasets",
        "gensim",
        "grpo.pipeline",
        "xgb.pipeline.reports",
    ],
    "py:data": [
        "sys.argv",
        "BUCKET_LABELS",
    ],
    "py:meth": [
        "datasets.Dataset.filter",
        "filter",
    ],
    
    "py:obj": [
        "common.pipeline.utils.StudyT",
        "common.pipeline.utils.OutcomeT",
        "knn.pipeline.utils.TaskT",
        "knn.pipeline.utils.OutcomeT",
        "common.pipeline.types.ConfigT",
        "common.pipeline.types.OutcomeT",
        "common.opinion.sweep_types.ConfigT",
        "common.opinion.models.ExampleT",
        "ConfigT",
        "TaskT",
        "OutcomeT",
    ],
    # Attribute cross-references emitted by autodoc for grouped configs
    "py:attr": [
        # XGBoostBoosterParams grouped attributes
        "sampling.colsample_bytree",
        "sampling.subsample",
        "regularization.reg_alpha",
        "regularization.reg_lambda",
        "core.learning_rate",
        "core.max_depth",
        "core.n_estimators",
        "core.tree_method",
        # XGBoostTrainConfig vectorizer bundle attributes
        "vectorizers.sentence_transformer",
        "vectorizers.tfidf",
        "vectorizers.word2vec",
    ],
}

nitpick_ignore: list[tuple[str, str]] = []
for domain, targets in _nitpick_targets.items():
    nitpick_ignore.extend((domain, target) for target in targets)

# Enable Sphinx "nitpicky" mode when requested (used in CI to fail on warnings).
nitpicky = os.environ.get("SPHINX_NITPICKY", "").lower() in {"1", "true", "yes"}

# Suppress MyST cross-reference warnings for optional report assets.
# The Markdown reports under `reports/` include example image paths that may not
# exist in fresh checkouts, and MyST treats some literals as cross-references.
# These are informational and should not fail the docs build.
suppress_warnings = [
    "myst.xref_missing",
    # Many report images are generated artifacts and may be absent in clean
    # checkouts; avoid noisy warnings when they are not present.
    "image.not_readable",
]
