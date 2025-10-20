"""Lightweight stub for the Hugging Face ``datasets`` library used in tests.

This stub is deliberately minimal—only the behaviours exercised by the unit
tests are implemented.  When the real library is available it is left
undisturbed; otherwise we register stand-in classes that mimic the APIs needed
by the test suite (``Dataset``, ``DatasetDict``, ``Value``, ``Sequence``,
``Features`` and a handful of helper functions).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional


def _infer_feature(values: List[Any]) -> "Value | Sequence":
    for value in values:
        if value is not None:
            if isinstance(value, list):
                return Sequence(Value("string"))
            return Value(type(value).__name__)
    return Value("null")


def ensure_datasets_stub() -> None:
    """Install the datasets stub if the real library is unavailable."""

    try:  # pragma: no cover - exercised implicitly when dependency exists
        import datasets  # type: ignore
    except ModuleNotFoundError:
        module = type(sys)("datasets")

        class Value:
            def __init__(self, dtype: str = "string") -> None:
                self.dtype = dtype

        @dataclass
        class Sequence:
            feature: "Value | Sequence"

        class Features(dict):  # type: ignore[type-arg]
            pass

        class Dataset:
            def __init__(self, data: Dict[str, List[Any]], features: Optional[Features] = None) -> None:
                lengths = {len(values) for values in data.values()} if data else {0}
                if len(lengths) > 1:
                    raise ValueError("All columns must share the same length")
                self._data = {key: list(values) for key, values in data.items()}
                if features is None:
                    inferred = {key: _infer_feature(values) for key, values in self._data.items()}
                    features = Features(inferred)
                self._features = features

            @classmethod
            def from_dict(cls, data: Dict[str, List[Any]]) -> "Dataset":
                return cls(data)

            @property
            def column_names(self) -> List[str]:
                return list(self._data.keys())

            @property
            def features(self) -> Features:
                return Features(self._features)

            def __len__(self) -> int:
                if not self._data:
                    return 0
                first_key = next(iter(self._data))
                return len(self._data[first_key])

            def _row(self, index: int) -> Dict[str, Any]:
                return {key: values[index] for key, values in self._data.items()}

            def __getitem__(self, item: Any) -> Any:
                if isinstance(item, str):
                    return self._data[item]
                if isinstance(item, int):
                    return self._row(item)
                raise TypeError(f"Unsupported index type: {type(item)!r}")

            def map(
                self,
                function: Callable[[Dict[str, Any]], Dict[str, Any]],
                *,
                remove_columns: Optional[Iterable[str]] = None,
                load_from_cache_file: Optional[bool] = None,  # noqa: ARG002 - kept for compatibility
            ) -> "Dataset":
                del remove_columns  # Our simplified implementation always starts from fresh outputs.
                outputs: List[Dict[str, Any]] = []
                for index in range(len(self)):
                    row = self._row(index)
                    result = function(row)
                    if not isinstance(result, dict):
                        raise TypeError("map function must return a dictionary")
                    outputs.append(result)
                if not outputs:
                    return Dataset({})
                new_data: Dict[str, List[Any]] = {key: [] for key in outputs[0].keys()}
                for output in outputs:
                    for key in new_data:
                        new_data[key].append(output.get(key))
                return Dataset.from_dict(new_data)

            def add_column(self, name: str, values: List[Any]) -> "Dataset":
                if len(values) != len(self):
                    raise ValueError("New column must match dataset length")
                data = {**self._data, name: list(values)}
                features = Features(self._features)
                features[name] = _infer_feature(values)
                return Dataset(data, features)

            def cast(self, features: Features) -> "Dataset":
                # Casting is a no-op for the stub; we simply record the requested features.
                return Dataset(self._data, Features(features))

            def filter(self, predicate: Callable[[Dict[str, Any]], bool]) -> "Dataset":
                data = {key: [] for key in self._data}
                for index in range(len(self)):
                    row = self._row(index)
                    if predicate(row):
                        for key in data:
                            data[key].append(self._data[key][index])
                return Dataset(data, Features(self._features))

            def unique(self, column_name: str) -> List[Any]:
                return list(dict.fromkeys(self._data[column_name]))

            def select_columns(self, column_names: Iterable[str]) -> "Dataset":
                data = {name: self._data[name] for name in column_names if name in self._data}
                features = Features({name: self._features[name] for name in data})
                return Dataset(data, features)

            def select(self, indices: Iterable[int]) -> "Dataset":
                idx_list = list(indices)
                data = {key: [self._data[key][i] for i in idx_list] for key in self._data}
                return Dataset(data, Features(self._features))

            def shuffle(self, seed: Optional[int] = None) -> "Dataset":  # noqa: ARG002 - parity with HF API
                return Dataset(self._data, Features(self._features))

            def save_to_disk(self, path: str | Path) -> None:
                folder = Path(path)
                folder.mkdir(parents=True, exist_ok=True)
                rows = [self._row(i) for i in range(len(self))]
                payload = {
                    "data": rows,
                    "features": {
                        key: {"type": ("sequence" if isinstance(feature, Sequence) else "value")}
                        for key, feature in self._features.items()
                    },
                }
                with (folder / "dataset.json").open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle)

            @classmethod
            def _load(cls, path: Path) -> "Dataset":
                with (path / "dataset.json").open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                data = {key: [row.get(key) for row in payload.get("data", [])] for key in payload.get("data", [{}])[0].keys()} if payload.get("data") else {}
                rows = payload.get("data", [])
                if rows:
                    columns = {key: [row.get(key) for row in rows] for key in rows[0].keys()}
                else:
                    columns = {}
                features = Features(
                    {
                        key: Sequence(Value("string")) if meta.get("type") == "sequence" else Value("string")
                        for key, meta in payload.get("features", {}).items()
                    }
                )
                return Dataset(columns, features)

        class DatasetDict:
            def __init__(self, mapping: Optional[Dict[str, Dataset]] = None) -> None:
                self._splits: Dict[str, Dataset] = mapping.copy() if mapping else {}

            def __getitem__(self, item: str) -> Dataset:
                return self._splits[item]

            def __setitem__(self, key: str, value: Dataset) -> None:
                self._splits[key] = value

            def __iter__(self) -> Iterator[str]:
                return iter(self._splits)

            def __len__(self) -> int:
                return len(self._splits)

            def items(self) -> Iterator[tuple[str, Dataset]]:
                return iter(self._splits.items())

            def keys(self) -> Iterable[str]:
                return self._splits.keys()

            def values(self) -> Iterable[Dataset]:
                return self._splits.values()

            def get(self, key: str, default: Optional[Dataset] = None) -> Optional[Dataset]:
                return self._splits.get(key, default)

            def save_to_disk(self, path: str | Path) -> None:
                folder = Path(path)
                folder.mkdir(parents=True, exist_ok=True)
                manifest = {"splits": list(self._splits.keys())}
                with (folder / "splits.json").open("w", encoding="utf-8") as handle:
                    json.dump(manifest, handle)
                for name, dataset in self._splits.items():
                    dataset.save_to_disk(folder / name)

            def push_to_hub(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401, ARG002
                raise RuntimeError("datasets stub does not support push_to_hub")

        def load_from_disk(path: str | Path) -> Dataset | DatasetDict:
            base = Path(path)
            if (base / "splits.json").exists():
                with (base / "splits.json").open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                mapping = {
                    name: Dataset._load(base / name) for name in manifest.get("splits", [])
                }
                return DatasetDict(mapping)
            if (base / "dataset.json").exists():
                return Dataset._load(base)
            raise FileNotFoundError(path)

        def concatenate_datasets(datasets_list: Iterable[Dataset]) -> Dataset:
            iterator = iter(datasets_list)
            try:
                first = next(iterator)
            except StopIteration:
                return Dataset({})
            columns = {name: list(values) for name, values in first._data.items()}
            for dataset in iterator:
                for name, values in dataset._data.items():
                    columns.setdefault(name, [])
                    columns[name].extend(values)
            return Dataset(columns)

        def load_dataset(*_args: Any, **_kwargs: Any) -> Dataset | DatasetDict:
            raise RuntimeError("datasets stub cannot load remote datasets")

        module.Dataset = Dataset
        module.DatasetDict = DatasetDict
        module.Features = Features
        module.Sequence = Sequence
        module.Value = Value
        module.concatenate_datasets = concatenate_datasets
        module.load_from_disk = load_from_disk
        module.load_dataset = load_dataset

        sys.modules["datasets"] = module
