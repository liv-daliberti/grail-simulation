"""Minimal stub for :mod:`graphviz` used in unit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional


def ensure_graphviz_stub() -> None:
    """Register a lightweight Graphviz substitute if the real package is missing."""

    try:  # pragma: no cover - executed only when dependency exists
        import graphviz  # type: ignore
    except ModuleNotFoundError:
        module = type(sys)("graphviz")

        class Digraph:
            def __init__(
                self,
                name: str = "G",
                engine: str = "dot",
                format: str = "pdf",
                comment: str | None = None,
            ) -> None:
                self.name = name
                self.engine = engine
                self.format = format
                self.comment = comment
                self._graph_attr: Dict[str, str] = {}
                self._node_attr: Dict[str, str] = {}
                self._edge_attr: Dict[str, str] = {}
                self._nodes: List[tuple[str, Dict[str, str]]] = []
                self._edges: List[tuple[str, str, Dict[str, str]]] = []

            @staticmethod
            def _format_value(value: str) -> str:
                if value.replace(".", "", 1).isdigit():
                    return value
                if any(ch in value for ch in [' ', ',', '"']) or value.startswith("#"):
                    return f'"{value}"'
                return value

            def _format_attrs(self, attrs: Dict[str, str]) -> str:
                if not attrs:
                    return ""
                parts = [f"{key}={self._format_value(value)}" for key, value in attrs.items()]
                return " [" + ", ".join(parts) + "]"

            def attr(self, name: Optional[str] = None, **kwargs: str) -> None:
                target = name or "graph"
                if target == "graph":
                    self._graph_attr.update({k: str(v) for k, v in kwargs.items()})
                elif target == "node":
                    self._node_attr.update({k: str(v) for k, v in kwargs.items()})
                elif target == "edge":
                    self._edge_attr.update({k: str(v) for k, v in kwargs.items()})
                else:
                    raise ValueError(f"Unsupported attr target: {target}")

            def node(self, name: str, label: Optional[str] = None, **attrs: str) -> None:
                merged = {**self._node_attr, **{k: str(v) for k, v in attrs.items()}}
                if label is not None:
                    merged.setdefault("label", label)
                self._nodes.append((name, merged))

            def edge(self, tail_name: str, head_name: str, label: Optional[str] = None, **attrs: str) -> None:
                merged = {**self._edge_attr, **{k: str(v) for k, v in attrs.items()}}
                if label is not None:
                    merged.setdefault("label", label)
                self._edges.append((tail_name, head_name, merged))

            @property
            def source(self) -> str:
                lines = [f"digraph {self.name} {{"]
                if self._graph_attr:
                    lines.append(
                        "    " + " ".join(f'{k}="{v}"' for k, v in self._graph_attr.items())
                    )
                if self._node_attr:
                    lines.append(
                        "    node" + self._format_attrs(self._node_attr) + ";"
                    )
                if self._edge_attr:
                    lines.append(
                        "    edge" + self._format_attrs(self._edge_attr) + ";"
                    )
                for name, attrs in self._nodes:
                    lines.append(f'    "{name}"' + self._format_attrs(attrs) + ";")
                for tail, head, attrs in self._edges:
                    lines.append(
                        f'    "{tail}" -> "{head}"' + self._format_attrs(attrs) + ";"
                    )
                lines.append("}")
                return "\n".join(lines)

            def render(self, filename: str, cleanup: bool = False) -> str:
                path = Path(filename)
                output = path.with_suffix(f".{self.format}")
                output.write_text(self.source, encoding="utf-8")
                if cleanup:
                    try:
                        path.with_suffix(".gv").unlink()
                    except FileNotFoundError:
                        pass
                return str(output)

        module.Digraph = Digraph
        sys.modules["graphviz"] = module
