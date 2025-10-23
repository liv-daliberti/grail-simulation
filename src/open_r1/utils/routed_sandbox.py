# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility layer that routes batch jobs to an E2B sandbox router."""

from typing import List, Optional

import requests
from e2b_code_interpreter.models import Execution, ExecutionError, Result

from .router_common import build_router_payload


class RoutedSandbox:
    """
    A sandbox environment that routes code execution requests to the E2B Router.
    This class is designed for batched execution of scripts, primarily for Python code.
    It mimics the usage of 'Sandbox' from 'e2b_code_interpreter', but adds support for batch processing.

    Attributes:
        router_url (str): The URL of the E2B Router to which code execution requests are sent.
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
    ) -> List[Execution]:
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
            return [Execution() for _ in scripts]

        results = response.json()
        output = []
        for result in results:
            if result["execution"] is None:
                execution = Execution()
            else:
                exec_payload = result["execution"]
                execution = Execution(
                    results=[Result(**res) for res in exec_payload["results"]],
                    logs=exec_payload["logs"],
                    error=(
                        ExecutionError(**exec_payload["error"])
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
