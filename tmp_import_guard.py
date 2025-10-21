
try:
    from datasets import load_dataset  # type: ignore
except ImportError:
    load_dataset = None
