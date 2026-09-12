#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.pytest_infra.manifest_loader - Load config/test_manifests/*.yaml and generate pytest parametrized data

YAML format convention:
    scenario: dirty_data              # Scenario name
    defaults:                         # Optional, field default values (inherited if case does not specify)
      future_covs: 24
    cases:
      - id: dirty_s0                  # Required, used for pytest.ids
        data: data/test_dirty_s0.csv  # Required, data file relative to testcases directory
        future_covs: 24               # Other fields are passed through to the test fixture/parameters
        tags: [dirty]                 # Optional, used to generate markers

Usage (inside test file):
    pytestmark = pytest.mark.parametrize(
        "case", load_manifest(__file__, "dirty_data"), ids=lambda c: c["id"]
    )
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any
from neuraxis_testkit.log import get_logger

logger = get_logger(__name__)

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "config" / "test_manifests"

REQUIRED_FIELDS = ("id", "data")


def load_manifest_file(manifest_path: Path) -> dict[str, Any]:
    """Load a single YAML manifest file and validate its structure."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        # raise FileNotFoundError(f"清单不存在: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError(f"Manifest format error (missing cases): {manifest_path}")

    defaults = data.get("defaults") or {}
    cases = []
    for i, raw in enumerate(data["cases"] or []):
        case = {**defaults, **raw}
        for field in REQUIRED_FIELDS:
            if field not in case:
                raise ValueError(f"Manifest {manifest_path.name} case {i} missing required field '{field}'")
        case["_scenario"] = data.get("scenario", manifest_path.stem)
        cases.append(case)
    return {"scenario": data.get("scenario", manifest_path.stem), "cases": cases}


def load_manifest(test_file: str, scenario: str) -> list[dict[str, Any]]:
    """
    Load manifest by scenario name and return a list of parameters directly usable for parametrize.

    Args:
        test_file: __file__ of the calling test file (for cross‑platform location)
        scenario:  Scenario name, e.g. "dirty_data"
    """
    manifest_path = MANIFEST_DIR / f"{scenario}.yaml"
    manifest = load_manifest_file(manifest_path)
    cases = manifest["cases"]

    # Resolve data file paths to absolute paths (relative to test file directory), cross‑platform safe.
    base = Path(test_file).resolve().parent
    for case in cases:
        data_path = Path(case["data"])
        if not data_path.is_absolute():
            candidate = base / data_path
            if not candidate.exists():
                candidate = Path(__file__).resolve().parents[1] / data_path
            data_path = candidate
        # Keep absolute path string, works on Windows/Linux
        case["data"] = str(data_path)

    logger.info(f"Manifest {manifest_path.name}: loaded {len(cases)} cases")
    return cases


def load_all_manifests() -> dict[str, list[dict[str, Any]]]:
    """Load all manifests for dynamic collection by collection.py."""
    result: dict[str, list[dict[str, Any]]] = {}
    if not MANIFEST_DIR.exists():
        logger.warning(f"Manifest directory does not exist: {MANIFEST_DIR}")
        return result
    for f in sorted(MANIFEST_DIR.glob("*.yaml")):
        manifest = load_manifest_file(f)
        result[manifest["scenario"]] = manifest["cases"]
    return result


def case_id_func(case: dict[str, Any]) -> str:
    """parametrize ids function: use case['id'] as the test case ID."""
    return str(case["id"])
