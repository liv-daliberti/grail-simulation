#!/usr/bin/env python
"""Compatibility layer that routes batch jobs to an E2B sandbox router."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

import requests

from .router_common import build_router_payload

if TYPE_CHECKING:  # pragma: no cover
    from e2b_code_interpreter.models import Execution, ExecutionError, Result


def _hydrate_execution(
    model_classes: Tuple[
        Type["Execution"],
        Type["ExecutionError"],
        Type["Result"],
    ],
    result: Dict[str, Any],
) -> "Execution":
    """
    Convert an E2B router response payload into an ``Execution`` instance.

    Separating the construction logic keeps ``run_code`` smaller and within the
    pylint local-variable threshold while keeping the conversion code testable.
    """

    execution_cls, execution_error_cls, result_cls = model_classes
    execution_payload = result.get("execution")
    if execution_payload is None:
        return execution_cls()

    error_payload = execution_payload.get("error")
    error_obj = (
        execution_error_cls(**error_payload) if error_payload else None
    )
    results = [
        result_cls(**res_payload) for res_payload in execution_payload.get("results", [])
    ]
    return execution_cls(
        results=results,
        logs=execution_payload.get("logs"),
        error=error_obj,
        execution_count=execution_payload.get("execution_count"),
    )


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
            )  # pylint: disable=import-error,import-outside-toplevel
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
        effective_timeout = timeout if timeout is not None else self.timeout
        effective_request_timeout = (
            request_timeout if request_timeout is not None else self.request_timeout
        )
        model_classes = self._get_model_classes()
        _, payload = build_router_payload(
            scripts,
            languages,
            timeout=effective_timeout,
            request_timeout=effective_request_timeout,
        )

        try:
            response = requests.post(
                f"http://{self.router_url}/execute_batch",
                json=payload,
                timeout=effective_request_timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Request to E2B router failed: {exc}")
            return [model_classes[0]() for _ in scripts]

        return [
            _hydrate_execution(model_classes, result)
            for result in response.json()
        ]


if __name__ == "__main__":
    # for local testing launch an E2B router with: python scripts/e2b_router.py
    sbx = RoutedSandbox(router_url="0.0.0.0:8000")
    codes = ["print('hello world')", "print('hello world)"]
    executions = sbx.run_code(codes)  # Execute Python inside the sandbox

    print(executions)
