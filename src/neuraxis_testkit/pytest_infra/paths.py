#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.pytest_infra.paths - NeuraxisPaths + get_paths (pure data + accessor)
"""
from dataclasses import dataclass
from pathlib import Path
import pytest

@dataclass(frozen=True)
class NeuraxisPaths:
    """Immutable container for framework-wide paths."""
    root: Path
    output: Path
    results: Path
    logs: Path
    reports: Path
    report_csv: Path
    log_file: Path

paths_key: pytest.StashKey[NeuraxisPaths] = pytest.StashKey()
# In-process fallback (each xdist worker holds its own).
_process_paths: NeuraxisPaths | None = None

def init_paths(config: pytest.Config, paths: NeuraxisPaths) -> None:
    """Called from hooks.pytest_configure; writes into stash and the in-process fallback."""

    global _process_paths
    config.stash[paths_key] = paths
    _process_paths = paths

def get_paths(config: pytest.Config | None = None) -> NeuraxisPaths:
    """Unified path-access entry point within the framework.

    Resolution order:
      1. config.stash   (explicit, testable, xdist-safe) — preferred.
      2. In-process cache (legacy compatibility / utility functions without
         a pytest context).

    Raises:
        RuntimeError: when neither source is available.
    """
    if config is not None:
        if paths_key not in config.stash:
            raise RuntimeError(
                "NeuraxisPaths has not been written into config.stash. "
                "Ensure pytest_configure has executed init_paths(config, paths)."
            )
        return config.stash[paths_key]
    if _process_paths is None:
        raise RuntimeError(
            "NeuraxisPaths has not been initialized. Ensure pytest_configure "
            "has executed, or pass config explicitly: get_paths(config)."
        )
    return _process_paths
