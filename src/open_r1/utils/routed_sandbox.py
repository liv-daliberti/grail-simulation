#!/usr/bin/env python
"""Compatibility layer that routes batch jobs to an E2B sandbox router."""

from __future__ import annotations

from typing import List, Optional, Tuple, Type, TYPE_CHECKING

import requests

from .router_common import build_router_payload

if TYPE_CHECKING:  # pragma: no cover
    from e2b_code_interpreter.models import Execution, ExecutionError, Result


class RoutedSandbox:
    """
    A sandbox environment that routes code execution requests to the E2B Router.
    This class is designed for batched execution of scripts, primarily for Python
    code. It mimics the usage of ``Sandbox`` from ``e2b_code_interpreter`` but
    adds support for batch processing.

    Attributes:
        router_url (str): The URL of the E2B Router receiving code execution
            requests.
    """

    def __init__(self, router_url: str):
        """
        Initializes the RoutedSandbox with the specified router URL.

        :param router_url: URL of the E2B Router.
        :type router_url: str
        """
        self.router_url = router_url
        self.timeout = 300
        self.request_timeout = 30
        self._model_classes: Tuple[
            Type["Execution"],
            Type["ExecutionError"],
            Type["Result"],
        ] | None = None

    @staticmethod
    def _load_model_classes() -> Tuple[
        Type["Execution"],
        Type["ExecutionError"],
        Type["Result"],
    ]:
        """Import execution-related models from ``e2b_code_interpreter`` on demand."""

        try:
            from e2b_code_interpreter.models import (
                Execution,
                ExecutionError,
                Result,
            )
        except ImportError as exc:  # pragma: no cover - handled via error propagation
            raise ImportError(
                "e2b-code-interpreter is required to use the routed sandbox. "
                "Install it with `pip install e2b-code-interpreter`."
            ) from exc

        return Execution, ExecutionError, Result

    def _get_model_classes(self) -> Tuple[
        Type["Execution"],
        Type["ExecutionError"],
        Type["Result"],
    ]:
        """Return cached model classes, importing them lazily when necessary."""

        if self._model_classes is None:
            self._model_classes = self._load_model_classes()
        return self._model_classes

    def configure_timeouts(
        self,
        *,
        timeout: Optional[int] = None,
        request_timeout: Optional[int] = None,
    ) -> None:
        """Update default execution or request timeouts."""
        if timeout is not None:
            self.timeout = timeout
        if request_timeout is not None:
            self.request_timeout = request_timeout

    def run_code(
        self,
        scripts: List[str],
        languages: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        request_timeout: Optional[int] = None,
    ) -> List["Execution"]:
        """
        Executes a batch of scripts in the sandbox environment.

        :param scripts: Code scripts to execute.
        :type scripts: list[str]
        :param languages: Programming languages for each script; defaults to Python when omitted.
        :type languages: list[str] | None
        :param timeout: Maximum execution time for each script in seconds.
        :type timeout: int | None
        :param request_timeout: HTTP request timeout in seconds.
        :type request_timeout: int | None
        :return: Execution objects containing results, logs, and errors (if any) per script.
        :rtype: list[Execution]
        """
        timeout = timeout if timeout is not None else self.timeout
        request_timeout = (
            request_timeout if request_timeout is not None else self.request_timeout
        )

        execution_cls, execution_error_cls, result_cls = self._get_model_classes()

        languages, payload = build_router_payload(
            scripts,
            languages,
            timeout=timeout,
            request_timeout=request_timeout,
        )

        try:
            response = requests.post(
                f"http://{self.router_url}/execute_batch",
                json=payload,
                timeout=request_timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Request to E2B router failed: {exc}")
            return [execution_cls() for _ in scripts]

        results = response.json()
        output = []
        for result in results:
            if result["execution"] is None:
                execution = execution_cls()
            else:
                exec_payload = result["execution"]
                execution = execution_cls(
                    results=[result_cls(**res) for res in exec_payload["results"]],
                    logs=exec_payload["logs"],
                    error=(
                        execution_error_cls(**exec_payload["error"])
                        if exec_payload["error"]
                        else None
                    ),
                    execution_count=exec_payload["execution_count"],
                )
            output.append(execution)

        return output


if __name__ == "__main__":
    # for local testing launch an E2B router with: python scripts/e2b_router.py
    sbx = RoutedSandbox(router_url="0.0.0.0:8000")
    codes = ["print('hello world')", "print('hello world)"]
    executions = sbx.run_code(codes)  # Execute Python inside the sandbox

    print(executions)
