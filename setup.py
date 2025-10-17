import os
from pathlib import Path
from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "pip"
TMP_DIR = PROJECT_ROOT / ".tmp"
PYCACHE_DIR = PROJECT_ROOT / ".cache" / "pyc"

for directory in (CACHE_DIR, TMP_DIR, PYCACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PIP_CACHE_DIR", str(CACHE_DIR))
os.environ.setdefault("TMPDIR", str(TMP_DIR))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(PYCACHE_DIR))


def read_long_description() -> str:
    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


setup(
    name="grail-simulation",
    version="0.1.0",
    description="Grounded-Retrieval Adversarial Imitation Loop (GRAIL) simulation toolkit",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="GRAIL Simulation Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "accelerate==1.4.0",
        "aiofiles>=24.1.0",
        "async-lru>=2.0.5",
        "bitsandbytes>=0.43.0",
        "datasets>=3.2.0",
        "deepspeed==0.16.8",
        "distilabel[vllm,ray,openai]>=1.5.2",
        "e2b-code-interpreter>=1.0.5",
        "einops>=0.8.0",
        "flake8>=6.0.0",
        "hf_transfer>=0.1.4",
        "huggingface-hub[cli,hf_xet]>=0.30.2,<1.0",
        "isort>=5.12.0",
        # Needed for Chinese language support
        "jieba",
        # Needed for LightEval's extended tasks
        "langdetect",
        "latex2sympy2_extended>=1.0.6",
        "liger-kernel>=0.5.10",
        # Critical bug fix for tokenizer revisions
        "lighteval @ git+https://github.com/huggingface/lighteval.git@d3da6b9bbf38104c8b5e1acc86f83541f9a502d1",
        "math-verify==0.5.2",
        "morphcloud==0.1.67",
        "numpy>=1.24.0",
        "packaging>=23.0",
        "pandas>=2.2.3",
        "parameterized>=0.9.0",
        "peft>=0.14.0",
        "pyarrow>=14.0.0",
        "pytest",
        "python-dotenv",
        "ruff>=0.9.0",
        "safetensors>=0.3.3",
        "scikit-learn>=1.3.0",
        "sentencepiece>=0.1.99",
        "torch==2.6.0",
        "transformers==4.52.3",
        "trl[vllm]==0.18.0",
        "tqdm>=4.66.0",
        "wandb>=0.19.1",
    ],
    include_package_data=True,
)
