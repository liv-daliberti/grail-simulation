"""Lightweight pandas stub implementing the limited API surface needed in tests."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence


def _is_nan(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def ensure_pandas_stub() -> None:
    """Install the pandas stub when the real dependency is unavailable."""

    try:  # pragma: no cover - executed only when pandas exists
        import pandas  # type: ignore
    except ModuleNotFoundError:
        module = type(sys)("pandas")

        class ParserError(Exception):
            """Placeholder for pandas.errors.ParserError."""

        module.errors = type("errors", (), {"ParserError": ParserError})

        class Series:
            def __init__(self, data: Iterable[Any], name: str | None = None) -> None:
                self._data = list(data)
                self.name = name

            def __iter__(self) -> Iterator[Any]:
                return iter(self._data)

            def __len__(self) -> int:
                return len(self._data)

            def __getitem__(self, index: Any) -> Any:
                if isinstance(index, Series):
                    mask = index._coerce_bool()
                    return Series(
                        [value for value, keep in zip(self._data, mask) if keep],
                        name=self.name,
                    )
                if isinstance(index, list):
                    return Series([self._data[i] for i in index], name=self.name)
                return self._data[index]

            def _coerce_bool(self) -> List[bool]:
                return [bool(value) for value in self._data]

            def fillna(self, value: Any) -> "Series":
                return Series(
                    [value if _is_nan(item) else item for item in self._data],
                    name=self.name,
                )

            def astype(self, dtype: Any) -> "Series":
                if dtype in (str, "str", "string"):
                    return Series(
                        ["" if item is None else str(item) for item in self._data],
                        name=self.name,
                    )
                raise TypeError(f"Unsupported dtype conversion: {dtype}")

            def eq(self, other: Any) -> "Series":
                return Series([item == other for item in self._data])

            def __eq__(self, other: Any) -> "Series":
                return self.eq(other)

            def __ne__(self, other: Any) -> "Series":
                return Series([item != other for item in self._data])

            def isin(self, values: Iterable[Any]) -> "Series":
                lookup = set(values)
                return Series([item in lookup for item in self._data])

            def between(
                self,
                left: float,
                right: float,
                inclusive: str = "both",
            ) -> "Series":
                left_inclusive = inclusive in {"both", "left"}
                right_inclusive = inclusive in {"both", "right"}
                result: List[bool] = []
                for item in self._data:
                    try:
                        numeric = float(item)
                    except (TypeError, ValueError):
                        result.append(False)
                        continue
                    left_ok = numeric >= left if left_inclusive else numeric > left
                    right_ok = numeric <= right if right_inclusive else numeric < right
                    result.append(left_ok and right_ok)
                return Series(result)

            def tolist(self) -> List[Any]:
                return list(self._data)

            def copy(self) -> "Series":
                return Series(list(self._data), name=self.name)

            def _compare(self, other: Any, op) -> "Series":
                result: List[bool] = []
                for item in self._data:
                    try:
                        lhs = float(item)
                        rhs = float(other)
                        result.append(op(lhs, rhs))
                    except (TypeError, ValueError):
                        result.append(False)
                return Series(result)

            def __ge__(self, other: Any) -> "Series":
                return self._compare(other, lambda a, b: a >= b)

            def __gt__(self, other: Any) -> "Series":
                return self._compare(other, lambda a, b: a > b)

            def __le__(self, other: Any) -> "Series":
                return self._compare(other, lambda a, b: a <= b)

            def __lt__(self, other: Any) -> "Series":
                return self._compare(other, lambda a, b: a < b)

            def __and__(self, other: "Series") -> "Series":
                other_bool = other._coerce_bool()
                return Series([a and b for a, b in zip(self._coerce_bool(), other_bool)])

            def __or__(self, other: "Series") -> "Series":
                other_bool = other._coerce_bool()
                return Series([a or b for a, b in zip(self._coerce_bool(), other_bool)])

            def __invert__(self) -> "Series":
                return Series([not value for value in self._coerce_bool()])

            def __iand__(self, other: "Series") -> "Series":
                combined = self.__and__(other)
                self._data = combined._data
                return self

            def __ior__(self, other: "Series") -> "Series":
                combined = self.__or__(other)
                self._data = combined._data
                return self

            class _StringAccessor:
                def __init__(self, series: "Series") -> None:
                    self._series = series

                def _apply(self, func) -> "Series":
                    return Series([func("" if value is None else str(value)) for value in self._series._data])

                def strip(self) -> "Series":
                    return self._apply(lambda s: s.strip())

                def lower(self) -> "Series":
                    return self._apply(lambda s: s.lower())

                def eq(self, other: str) -> "Series":
                    return Series([(("" if value is None else str(value)) == other) for value in self._series._data])

                def startswith(self, prefix: str) -> "Series":
                    return Series([str(value).startswith(prefix) for value in self._series._data])

            @property
            def str(self) -> "Series._StringAccessor":
                return Series._StringAccessor(self)

        class DataFrame:
            def __init__(self, rows: Optional[Iterable[dict]] = None, columns: Optional[Sequence[str]] = None) -> None:
                if rows is None:
                    rows = []
                rows_list = [dict(row) for row in rows]
                if columns is None and rows_list:
                    ordered: List[str] = []
                    for row in rows_list:
                        for key in row.keys():
                            if key not in ordered:
                                ordered.append(key)
                    columns = ordered
                self._columns = list(columns or [])
                self._rows: List[dict] = []
                for row in rows_list:
                    normalized = {col: row.get(col) for col in self._columns}
                    self._rows.append(normalized)

            @property
            def empty(self) -> bool:
                return not self._rows

            @property
            def columns(self) -> List[str]:
                return list(self._columns)

            def __len__(self) -> int:
                return len(self._rows)

            def __iter__(self) -> Iterator[str]:
                return iter(self._columns)

            def copy(self) -> "DataFrame":
                return DataFrame([dict(row) for row in self._rows], columns=self._columns)

            def iterrows(self) -> Iterator[tuple[int, dict]]:
                for index, row in enumerate(self._rows):
                    yield index, dict(row)

            def __getitem__(self, key: Any) -> Any:
                if isinstance(key, str):
                    return Series([row.get(key) for row in self._rows], name=key)
                if isinstance(key, list):
                    subset_rows = [{col: row.get(col) for col in key} for row in self._rows]
                    return DataFrame(subset_rows, columns=key)
                if isinstance(key, Series):
                    mask = key._coerce_bool()
                    filtered = [row for row, keep in zip(self._rows, mask) if keep]
                    return DataFrame(filtered, columns=self._columns)
                if isinstance(key, slice):
                    sliced = self._rows[key]
                    return DataFrame(sliced, columns=self._columns)
                raise TypeError(f"Unsupported index type: {type(key)!r}")

            def __setitem__(self, key: str, value: Series) -> None:
                values = list(value)
                if len(values) != len(self._rows):
                    raise ValueError("Column length must match number of rows")
                if key not in self._columns:
                    self._columns.append(key)
                for row, val in zip(self._rows, values):
                    row[key] = val

            def get(self, key: str, default: Any = None) -> Any:
                if key in self._columns:
                    return self[key]
                return default

            def to_csv(self, path: Path | str, index: bool = False) -> None:  # noqa: ARG002
                with Path(path).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(self._columns)
                    for row in self._rows:
                        writer.writerow([row.get(col, "") for col in self._columns])

            class _LocIndexer:
                def __init__(self, frame: "DataFrame") -> None:
                    self._frame = frame

                def __getitem__(self, key: Any) -> Any:
                    if isinstance(key, tuple):
                        row_key, col_key = key
                    else:
                        row_key, col_key = key, None

                    if isinstance(row_key, Series):
                        mask = row_key._coerce_bool()
                    elif isinstance(row_key, list):
                        mask = row_key
                    else:
                        raise TypeError("Row selector must be a boolean Series or list")

                    filtered_rows = [
                        row for row, keep in zip(self._frame._rows, mask) if keep
                    ]

                    if col_key is None:
                        return DataFrame(filtered_rows, columns=self._frame._columns)

                    if isinstance(col_key, list):
                        return DataFrame(
                            [{col: row.get(col) for col in col_key} for row in filtered_rows],
                            columns=col_key,
                        )

                    return Series([row.get(col_key) for row in filtered_rows], name=col_key)

            @property
            def loc(self) -> "_LocIndexer":
                return DataFrame._LocIndexer(self)

            def drop_duplicates(self, subset: List[str], keep: str = "first") -> "DataFrame":  # noqa: ARG002
                seen = set()
                unique_rows = []
                for row in self._rows:
                    key = tuple(row.get(col) for col in subset)
                    if key in seen:
                        continue
                    seen.add(key)
                    unique_rows.append(dict(row))
                return DataFrame(unique_rows, columns=self._columns)

            def drop(self, columns: List[str], errors: str = "raise") -> "DataFrame":
                for column in columns:
                    if column not in self._columns:
                        if errors != "ignore":
                            raise KeyError(column)
                        continue
                    index = self._columns.index(column)
                    self._columns.pop(index)
                    for row in self._rows:
                        row.pop(column, None)
                return self

            def sort_values(self, by: List[str], kind: str | None = None) -> "DataFrame":  # noqa: ARG002
                def _key(row: dict) -> tuple:
                    values = []
                    for col in by:
                        val = row.get(col)
                        if _is_nan(val):
                            values.append((1, None))
                        else:
                            values.append((0, val))
                    return tuple(values)

                sorted_rows = sorted(self._rows, key=_key)
                return DataFrame(sorted_rows, columns=self._columns)

            def where(self, condition: Any, other: Any) -> "DataFrame":
                if isinstance(condition, DataFrame):
                    cond_rows = condition._rows
                elif isinstance(condition, Series):
                    cond_rows = [{col: flag for col in self._columns} for flag in condition._coerce_bool()]
                else:
                    raise TypeError("Unsupported condition type for where()")
                result_rows: List[dict] = []
                for row, mask in zip(self._rows, cond_rows):
                    new_row: dict = {}
                    for col in self._columns:
                        keep = mask[col] if isinstance(mask, dict) else mask
                        new_row[col] = row[col] if keep else other
                    result_rows.append(new_row)
                return DataFrame(result_rows, columns=self._columns)

        def DataFrame_constructor(data: Optional[Any] = None, dtype: Any = None) -> DataFrame:  # noqa: ARG002
            if data is None:
                return DataFrame()
            if isinstance(data, list):
                return DataFrame(data)
            if isinstance(data, dict):
                columns = list(data.keys())
                rows = []
                length = max((len(values) for values in data.values()), default=0)
                for i in range(length):
                    rows.append({col: data[col][i] if i < len(data[col]) else None for col in columns})
                return DataFrame(rows, columns=columns)
            raise TypeError("Unsupported data type for DataFrame stub")

        def read_csv(path: Path | str, dtype: Any = None) -> DataFrame:  # noqa: ARG002
            with Path(path).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                try:
                    header = next(reader)
                except StopIteration:
                    return DataFrame()
                rows = []
                for row in reader:
                    rows.append({col: row[i] if i < len(row) else "" for i, col in enumerate(header)})
                return DataFrame(rows, columns=header)

        def isna(value: Any) -> bool:
            if isinstance(value, Series):
                return Series([_is_nan(item) for item in value])
            if isinstance(value, DataFrame):
                return DataFrame(
                    [{col: _is_nan(row.get(col)) for col in value.columns} for row in value._rows],
                    columns=value.columns,
                )
            return _is_nan(value)

        def notna(value: Any) -> Any:
            if isinstance(value, Series):
                return Series([not _is_nan(item) for item in value])
            if isinstance(value, DataFrame):
                return DataFrame(
                    [{col: not _is_nan(row.get(col)) for col in value.columns} for row in value._rows],
                    columns=value.columns,
                )
            return not _is_nan(value)

        def to_numeric(series: Series, errors: str = "raise") -> Series:
            result = []
            for item in series:
                try:
                    result.append(float(item))
                except (TypeError, ValueError):
                    if errors == "coerce":
                        result.append(float("nan"))
                    else:
                        raise
            return Series(result)

        def to_datetime(series: Series, unit: str | None = None, errors: str = "raise", utc: bool = False) -> Series:  # noqa: ARG002
            result = []
            for item in series:
                try:
                    value = float(item)
                    if unit == "ms":
                        value /= 1000.0
                    result.append(value)
                except (TypeError, ValueError):
                    if errors == "coerce":
                        result.append(float("nan"))
                    else:
                        raise
            return Series(result)

        module.DataFrame = DataFrame_constructor
        module.Series = Series
        module.read_csv = read_csv
        module.isna = isna
        module.notna = notna
        module.to_numeric = to_numeric
        module.to_datetime = to_datetime

        sys.modules["pandas"] = module
