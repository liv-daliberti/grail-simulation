#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
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

"""Asynchronous load-balanced client for IOI Piston execution workers."""

# pylint: disable=line-too-long

import asyncio
import os
import random
import re
import subprocess
from contextlib import suppress
from functools import lru_cache
from typing import Union

import aiohttp


class PistonError(Exception):
    """Raised when all available Piston endpoints fail to satisfy a request."""


@lru_cache(maxsize=1)
def get_piston_client_from_env(session=None):
    """Initialise a :class:`PistonClient` using configuration from the environment.

    :param session: Optional shared :class:`aiohttp.ClientSession`.
    :returns: Configured :class:`PistonClient` instance.
    :raises ValueError: When the ``PISTON_ENDPOINTS`` variable is missing.
    """
    piston_endpoints = os.getenv("PISTON_ENDPOINTS")
    if piston_endpoints is None:
        raise ValueError(
            "For IOI/CF problems Piston endpoints running our IOI package are required. Please add a list of valid Piston endpoints to a PISTON_ENDPOINTS variable in a `.env` file."
        )
    piston_endpoints = sorted(
        piston_endpoints.split(",") if piston_endpoints != "slurm" else get_slurm_piston_endpoints()
    )
    gpu_nb = int(os.getenv("LOCAL_RANK", "0"))  # per‑GPU index
    world = int(os.getenv("WORLD_SIZE", "1"))  # total GPUs
    if world > 1:
        print(f"Using a subset of piston endpoints for GPU#{gpu_nb}")
        piston_endpoints = piston_endpoints[gpu_nb::world]
    random.shuffle(piston_endpoints)
    max_requests_per_endpoint = os.getenv("PISTON_MAX_REQUESTS_PER_ENDPOINT", "1")
    return PistonClient(piston_endpoints, session, max_requests_per_endpoint=int(max_requests_per_endpoint))


class PistonClient:
    r"""
    A client that will automatically load balance across multiple Piston (https://github.com/engineer-man/piston) workers.
    This assumes piston is running our custom cms_ioi package: https://github.com/guipenedo/piston/releases/
    We recommend starting the instances with the following script as otherwise some IOI problems will hit default limits:
    ```
    export PISTON_COMPILE_TIMEOUT=60000
    export PISTON_RUN_TIMEOUT=60000
    export PISTON_OUTPUT_MAX_SIZE=1000000000
    export PISTON_MAX_FILE_SIZE=1000000000
    export PISTON_DISABLE_NETWORKING=true
    export PISTON_REPO_URL=https://github.com/guipenedo/piston/releases/download/pkgs/index
    mkdir /piston

    sed -i '/app.use(body_parser.urlencoded/c\    app.use(body_parser.urlencoded({ extended: true, limit: \"512mb\" }));' src/index.js
    sed -i '/app.use(body_parser.json/c\    app.use(body_parser.json({ limit: \"512mb\" }));' src/index.js

    # Start server in background
    node src```

    Piston docs for API usage: https://piston.readthedocs.io/en/latest/api-v2/
    """

    def __init__(
        self,
        base_endpoint: Union[str, list[str]] = "http://ip-10-53-80-65:3223/api/v2",
        session=None,
        max_requests_per_endpoint=1,
    ):
        """Create a load-balanced client over one or more Piston endpoints.

        :param base_endpoint: Single endpoint or list of endpoints.
        :param session: Optional shared :class:`aiohttp.ClientSession`.
        :param max_requests_per_endpoint: Parallel requests allowed per endpoint.
        :raises ValueError: If no endpoints are provided.
        """
        self.max_requests_per_endpoint = max_requests_per_endpoint
        self.base_endpoints = [base_endpoint] if isinstance(base_endpoint, str) else base_endpoint
        if len(self.base_endpoints) == 0:
            raise ValueError("No Piston endpoints provided. Please check your PISTON_ENDPOINTS environment variable.")
        self.endpoint_ids = {endpoint: i for i, endpoint in enumerate(self.base_endpoints)}

        self._session = session
        self.endpoint_tokens = asyncio.Queue(maxsize=max_requests_per_endpoint * len(self.base_endpoints))

        for _ in range(max_requests_per_endpoint):
            for endpoint in self.base_endpoints:
                self.endpoint_tokens.put_nowait(endpoint)

        self._endpoint_state = {
            "unhealthy": set(),
            "lock": asyncio.Lock(),
        }

    @property
    def session(self):
        """Return a lazily-initialised :class:`aiohttp.ClientSession`.

        :returns: Shared client session for HTTP requests.
        """
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(sock_read=30),
                connector=aiohttp.TCPConnector(
                    limit=self.max_requests_per_endpoint * len(self.base_endpoints),
                    ttl_dns_cache=300,
                    keepalive_timeout=5 * 60,
                ),
            )
        return self._session

    async def _wait_for_endpoint(self):
        """Acquire the next available endpoint token for issuing a request.

        :returns: Endpoint base URL reserved for the request.
        """
        endpoint = await self.endpoint_tokens.get()
        return endpoint

    async def _release_endpoint(self, endpoint):
        """Release a previously acquired endpoint token back to the pool.

        :param endpoint: Endpoint URL returned by :meth:`_wait_for_endpoint`.
        """
        await self.endpoint_tokens.put(endpoint)

    async def _send_request(self, endpoint, route, data=None, method="post"):
        """Send an HTTP request to a specific endpoint.

        :param endpoint: Endpoint base URL.
        :param route: API route to call (without leading slash).
        :param data: Optional JSON payload.
        :param method: HTTP method to use.
        :returns: JSON-decoded response payload.
        """
        async with self.session.request(
            method, f"{endpoint.rstrip('/')}/{route}", json=data, headers={"Content-Type": "application/json"}
        ) as response:
            return await response.json(content_type=None)

    async def _send_to_all(self, route, data=None, method="post"):
        """Send a request to all endpoints concurrently and gather responses.

        :param route: API route to call.
        :param data: Optional JSON payload.
        :param method: HTTP method to use.
        :returns: List of JSON responses from each endpoint.
        """
        return await asyncio.gather(
            *[self._send_request(endpoint, route, data, method) for endpoint in self.base_endpoints]
        )

    async def _send_to_one(self, endpoint, route, data=None, method="post"):
        """Send a request to a single endpoint.

        :param endpoint: Endpoint base URL.
        :param route: API route to call.
        :param data: Optional JSON payload.
        :param method: HTTP method to use.
        :returns: JSON-decoded response payload.
        """
        return await self._send_request(endpoint, route, data, method)

    async def install_package(self, language, version):
        """Install a runtime package on all endpoints.

        :param language: Runtime language identifier.
        :param version: Package version to install.
        :returns: Responses returned by each endpoint.
        """
        return await self._send_to_all("packages", {"language": language, "version": version}, method="post")

    async def uninstall_package(self, language, version):
        """Uninstall a runtime package from all endpoints.

        :param language: Runtime language identifier.
        :param version: Package version to remove.
        :returns: Responses returned by each endpoint.
        """
        return await self._send_to_all("packages", {"language": language, "version": version}, method="delete")

    async def get_supported_runtimes(self):
        """Return the runtimes supported by each configured endpoint.

        :returns: List of runtime metadata responses.
        """
        return await self._send_to_all("runtimes", method="get")

    async def _check_failed_endpoint(self, endpoint):
        """Probe and mark endpoints that repeatedly fail requests.

        :param endpoint: Endpoint URL to check.
        :raises PistonError: When all endpoints are marked unhealthy.
        """
        state = self._endpoint_state
        async with state["lock"]:
            if endpoint in state["unhealthy"]:
                return
            try:
                await asyncio.sleep(5)
                await self.get_supported_runtimes()
            except (aiohttp.ClientError, asyncio.TimeoutError, PistonError) as error:
                print(f"Error checking endpoint {endpoint}, dropping it ({error})")
                state["unhealthy"].add(endpoint)
                if len(state["unhealthy"]) >= len(self.base_endpoints):
                    raise PistonError(
                        "All endpoints are unhealthy. Please check your Piston workers."
                    ) from error

    async def send_execute(self, data, language="cms_ioi", max_retries=5):
        # pylint: disable=too-many-branches
        """Execute code on a managed endpoint with retry and backoff.

        :param data: Execution payload forwarded to Piston.
        :param language: Runtime language identifier.
        :param max_retries: Maximum number of retry attempts.
        :returns: JSON response returned by the successful endpoint.
        :raises PistonError: If all retries fail or endpoints are unhealthy.
        """
        payload = {
            "language": language,
            "version": "*",
        }
        data = {**(data or {}), **payload}

        base_delay = 1.0

        status = None
        endpoint = None

        for attempt in range(max_retries + 1):
            try:
                endpoint = await self._wait_for_endpoint()
                if attempt > 0:
                    await asyncio.sleep(1)
                async with self.session.post(
                    f"{endpoint.rstrip('/')}/execute", json=data, headers={"Content-Type": "application/json"}
                ) as response:
                    status = response.status
                    res_json = await response.json(content_type=None)

                    if status != 200:
                        raise PistonError(f"Server error. status={status}. {res_json}")
                    if res_json is None:
                        raise PistonError(f"Empty response. status={status}")
                    # piston overloaded
                    if "run" in res_json and "Resource temporarily unavailable" in res_json["run"].get("stderr", ""):
                        raise PistonError(f"Piston overloaded: {res_json['run']['stderr']}")
                    return res_json

            except (PistonError, asyncio.TimeoutError, aiohttp.ClientConnectionError, RuntimeError) as error:
                # Only retry if we haven't reached max retries yet
                if attempt < max_retries:
                    # Calculate backoff with jitter
                    delay = min(base_delay * (2**attempt), 10)  # Exponential backoff, capped at 10 seconds
                    jitter = delay * 0.2 * (2 * asyncio.get_event_loop().time() % 1 - 0.5)  # Add ±10% jitter
                    retry_delay = delay + jitter
                    print(
                        f"Retrying in {retry_delay:.2f} seconds [{self.endpoint_ids[endpoint]}] "
                        f"{endpoint} - {error}"
                    )

                    # special case: worker died
                    if isinstance(error, aiohttp.ClientConnectionError) and "Connect call failed" in str(error):
                        await self._check_failed_endpoint(endpoint)
                    else:
                        # hopefully we won't get this one again
                        await self._release_endpoint(endpoint)
                    endpoint = None

                    await asyncio.sleep(retry_delay)
                else:
                    await self._check_failed_endpoint(endpoint)
            finally:
                # Ensure endpoint is always released, even if an exception occurs
                if endpoint is not None:
                    with suppress(Exception):
                        await self._release_endpoint(endpoint)
                    endpoint = None


def get_slurm_piston_endpoints():
    """Return active Piston worker endpoints extracted from ``squeue`` output.

    :returns: List of endpoint URLs discovered via SLURM.
    """
    # Run squeue command to get job name, hostname and status, filtering for RUNNING state
    result = subprocess.run(
        ["squeue", '--format="%j %N %T"', "--noheader", "--states=RUNNING"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Split output into lines and skip header
    lines = result.stdout.strip().split("\n")

    endpoints = []
    for line in lines:
        # Parse job name from squeue output
        fields = line.split()
        job_name = fields[0].strip('"')  # Remove quotes
        hostname = fields[1]

        # Extract port if job name matches pattern
        match = re.match(r"piston-worker-(\d+)", job_name)
        if match:
            port = match.group(1)
            endpoints.append(f"http://{hostname}:{port}/api/v2")

    return endpoints
