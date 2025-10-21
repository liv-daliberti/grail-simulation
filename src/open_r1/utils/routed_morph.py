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

"""Thin wrapper around the MorphCloud router batch execution API."""

from typing import List, Optional

import requests


class RoutedMorphSandbox:
    """
    Client for the MorphCloud router service that mimics the API of MorphCloud's Sandbox.

    This class provides a simple interface to execute code via a central MorphCloud router,
    which manages sandbox creation and cleanup. It allows batch processing of multiple scripts
    in a single request for improved efficiency.

    Attributes:
        router_url (str): The URL of the MorphCloud router service.
        timeout (int): Execution timeout in seconds.
        request_timeout (int): HTTP request timeout in seconds.
    """

    def __init__(self, router_url: str, timeout: int = 300, request_timeout: int = 60):
        """
        Initialize the routed MorphCloud sandbox client.

        :param router_url: URL of the MorphCloud router, including host and port.
        :type router_url: str
        :param timeout: Default execution timeout in seconds.
        :type timeout: int
        :param request_timeout: Default HTTP request timeout in seconds.
        :type request_timeout: int
        """
        self.router_url = router_url
        self.timeout = timeout
        self.request_timeout = request_timeout

    def run_code(
        self,
        scripts: List[str],
        languages: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        request_timeout: Optional[int] = None,
    ) -> List:
        """
        Execute multiple scripts using MorphCloud via the router.

        :param scripts: Code scripts to execute.
        :type scripts: list[str]
        :param languages: Programming languages for each script; defaults to Python when omitted.
        :type languages: list[str] | None
        :param timeout: Execution timeout in seconds; defaults to the instance timeout.
        :type timeout: int | None
        :param request_timeout: HTTP request timeout in seconds; defaults to the instance setting.
        :type request_timeout: int | None
        :return: Execution results with ``text`` and ``exception_str`` attributes.
        :rtype: list
        """

        actual_timeout = timeout if timeout is not None else self.timeout
        actual_request_timeout = request_timeout if request_timeout is not None else self.request_timeout

        # Default to Python for all scripts if languages is not provided
        if languages is None:
            languages = ["python"] * len(scripts)

        payload = {
            "scripts": scripts,
            "languages": languages,
            "timeout": actual_timeout,
            "request_timeout": actual_request_timeout,
        }

        try:
            endpoint = f"http://{self.router_url}/execute_batch"
            response = requests.post(endpoint, json=payload, timeout=actual_request_timeout)

            if response.status_code != 200:
                error = f"Request to MorphCloud router failed with status code: {response.status_code}"
                print(error)

                results = []
                for _ in scripts:
                    results.append(type("obj", (object,), {"text": None, "exception_str": error}))
                return results

            response_data = response.json()
            results = []

            for item in response_data:
                # Log the response data to see what we're getting
                # print(f"RoutedMorphSandbox: Got response item: {item}")
                result = type(
                    "obj",
                    (object,),
                    {
                        "text": item.get("text"),
                        "exception_str": item.get("exception_str"),
                    },
                )
                results.append(result)

            return results

        except Exception as e:
            error = f"Error communicating with MorphCloud router: {str(e)}"
            print(error)

            results = []
            for _ in scripts:
                results.append(type("obj", (object,), {"text": None, "exception_str": error}))
            return results
