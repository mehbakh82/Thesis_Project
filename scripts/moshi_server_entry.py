#!/usr/bin/env python3
"""Project-owned secure entry point for the pinned official Moshi server."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def requested_host(argv: list[str]) -> str:
    host = "127.0.0.1"
    for index, argument in enumerate(argv):
        if argument == "--host":
            if index + 1 >= len(argv):
                raise ValueError("--host requires a value")
            host = argv[index + 1]
        elif argument.startswith("--host="):
            host = argument.split("=", 1)[1]
    if not host:
        raise ValueError("server host must be nonempty")
    return host


def configure_bound_host(web_module, host: str) -> None:
    """Make the upstream server honor its parsed --host when binding."""

    original_run_app = web_module.run_app

    def run_app(app, *args, **kwargs):
        requested = kwargs.get("host")
        if requested not in (None, host):
            raise RuntimeError(f"conflicting server bind host: {requested}")
        kwargs["host"] = host
        return original_run_app(app, *args, **kwargs)

    web_module.run_app = run_app
    os.environ["MOSHI_SERVER_BOUND_HOST"] = host


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checkout = root / "third_party/checkouts/moshi/moshi"
    server_module = checkout / "moshi/server.py"
    if not server_module.is_file():
        raise RuntimeError(f"pinned Moshi server is missing: {server_module}")
    sys.path.insert(0, str(checkout))

    from aiohttp import web

    host = requested_host(sys.argv[1:])
    configure_bound_host(web, host)
    runpy.run_module("moshi.server", run_name="__main__")


if __name__ == "__main__":
    main()
