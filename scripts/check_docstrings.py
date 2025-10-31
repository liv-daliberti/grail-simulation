#!/usr/bin/env python
"""AST-based docstring consistency checker.

Checks Python files for:
- module/class/function docstrings
- presence of ``:param:`` fields for function/method arguments
- presence of ``:returns:``/``:return:`` (or "Returns:" for Napoleon style)

Usage:
    python scripts/check_docstrings.py [PATH ...]

If no PATH is supplied, defaults to checking ``src/visualization``.
Exits with a non-zero status if any gap is detected.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Iterable


PARAM_RE = re.compile(r":param\s+(\*\*?\w+|\w+)(?:\s*:[^\n]*)?")
RETURNS_PRESENT = re.compile(r":returns?:|Returns:\s*", re.IGNORECASE)


def iter_py_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from (q for q in p.rglob("*.py"))
        elif p.suffix == ".py":
            yield p


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Files or directories to scan")
    args = parser.parse_args(argv)

    targets = args.paths or [Path("src/visualization")]
    py_files = list(iter_py_files(targets))
    if not py_files:
        print("No Python files found to check.", file=sys.stderr)
        return 0

    report = {
        "modules_missing_doc": [],
        "classes_missing_doc": [],
        "dataclasses_missing_attributes": [],
        "functions_missing_doc": [],
        "functions_missing_params": [],
        "functions_missing_returns": [],
    }

    def _record(key: str, msg: str) -> None:
        report[key].append(msg)

    for path in py_files:
        try:
            src = path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"WARN: cannot read {path}: {exc}", file=sys.stderr)
            continue
        try:
            tree = ast.parse(src)
        except Exception as exc:
            print(f"WARN: cannot parse {path}: {exc}", file=sys.stderr)
            continue

        mod_doc = ast.get_docstring(tree)
        if not mod_doc:
            _record("modules_missing_doc", str(path))

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                cls_doc = ast.get_docstring(node)
                if not cls_doc:
                    _record("classes_missing_doc", f"{path}:{node.lineno}:{node.name}")
                # dataclass detection
                is_dataclass = False
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name) and dec.id == "dataclass":
                        is_dataclass = True
                    elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "dataclass":
                        is_dataclass = True
                if is_dataclass:
                    if not cls_doc or ("Attributes:" not in cls_doc and ":ivar" not in cls_doc):
                        _record("dataclasses_missing_attributes", f"{path}:{node.lineno}:{node.name}")
                # methods
                for n2 in node.body:
                    if isinstance(n2, ast.FunctionDef):
                        doc = ast.get_docstring(n2)
                        if not doc:
                            _record("functions_missing_doc", f"{path}:{n2.lineno}:{node.name}.{n2.name}")
                            continue
                        arg_names = (
                            [a.arg for a in getattr(n2.args, "posonlyargs", [])]
                            + [a.arg for a in n2.args.args]
                            + [a.arg for a in n2.args.kwonlyargs]
                        )
                        arg_names = [a for a in arg_names if a not in {"self", "cls"}]
                        found = {m.group(1).lstrip("*") for m in PARAM_RE.finditer(doc)}
                        missing = [a for a in arg_names if a not in found]
                        if missing:
                            _record(
                                "functions_missing_params",
                                f"{path}:{n2.lineno}:{node.name}.{n2.name} missing params: {missing}",
                            )
                        if not RETURNS_PRESENT.search(doc):
                            _record(
                                "functions_missing_returns",
                                f"{path}:{n2.lineno}:{node.name}.{n2.name} missing :returns:",
                            )
            elif isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node)
                if not doc:
                    _record("functions_missing_doc", f"{path}:{node.lineno}:{node.name}")
                    continue
                arg_names = (
                    [a.arg for a in getattr(node.args, "posonlyargs", [])]
                    + [a.arg for a in node.args.args]
                    + [a.arg for a in node.args.kwonlyargs]
                )
                arg_names = [a for a in arg_names if a not in {"self", "cls"}]
                found = {m.group(1).lstrip("*") for m in PARAM_RE.finditer(doc)}
                missing = [a for a in arg_names if a not in found]
                if missing:
                    _record("functions_missing_params", f"{path}:{node.lineno}:{node.name} missing params: {missing}")
                if not RETURNS_PRESENT.search(doc):
                    _record("functions_missing_returns", f"{path}:{node.lineno}:{node.name} missing :returns:")

    print("Docstring consistency report:\n")
    failures = 0
    for key, items in report.items():
        print(f"{key} ({len(items)}):")
        for item in items:
            print(f"  - {item}")
        print()
        failures += len(items)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
