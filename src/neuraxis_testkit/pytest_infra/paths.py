#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit/pytest_infra/paths.py - NeuraxisPaths + get_paths (pure data + accessor)
"""
from dataclasses import dataclass
from pathlib import Path
import pytest

@dataclass(frozen=True)
class NeuraxisPaths:
    root: Path
    output: Path
    results: Path
    logs: Path
    reports: Path
    report_csv: Path
    log_file: Path

paths_key: pytest.StashKey[NeuraxisPaths] = pytest.StashKey()

def init_paths(config: pytest.Config, paths: NeuraxisPaths) -> None:
    """Called by conftest.py during pytest_configure phase."""
    config.stash[paths_key] = paths

def get_paths(config: pytest.Config) -> NeuraxisPaths:
    """Unified path access entry point within the framework."""
    return config.stash[paths_key]