import os
from pathlib import Path
from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
CACHE_DIR = REPO_ROOT / ".cache" / "pip"
TMP_DIR = REPO_ROOT / ".tmp"
PYCACHE_DIR = REPO_ROOT / ".cache" / "pyc"

for directory in (CACHE_DIR, TMP_DIR, PYCACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PIP_CACHE_DIR", str(CACHE_DIR))
os.environ.setdefault("TMPDIR", str(TMP_DIR))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(PYCACHE_DIR))


def read_long_description() -> str:
    readme_path = REPO_ROOT / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


def read_requirements(relative_path: str) -> list[str]:
    """Return requirement specifiers from a requirements-style text file."""
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return []
    requirements: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            requirements.append(stripped)
    return requirements


SRC_DIR = REPO_ROOT / "src"
CLEAN_DATA_DIR = REPO_ROOT / "clean_data"
SRC_DIR_REL = os.path.relpath(SRC_DIR, PROJECT_ROOT)
CLEAN_DATA_DIR_REL = os.path.relpath(CLEAN_DATA_DIR, PROJECT_ROOT)

src_packages = find_packages(where=str(SRC_DIR))
clean_data_packages = find_packages(where=str(REPO_ROOT), include=["clean_data", "clean_data.*"])
dev_requirements = read_requirements("requirements-dev.txt")

setup(
    name="grail-simulation",
    version="0.1.0",
    description="Grounded-Retrieval Adversarial Imitation Loop (GRAIL) simulation toolkit",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="GRAIL Simulation Team",
    packages=sorted(set(src_packages + clean_data_packages)),
    package_dir={"": SRC_DIR_REL, "clean_data": CLEAN_DATA_DIR_REL},
    python_requires=">=3.10",
    install_requires=read_requirements("../requirements.txt"),
    extras_require={
        "dev": sorted(set(dev_requirements)),
    },
    include_package_data=True,
)
