#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.pytest_infra.collection - Dynamic test case collection driven by YAML manifests

Two working modes:
  A. Static test cases + manifest parameterization (recommended, write test_*.py under testcases/ normally)
     -> This file is not needed; tests directly use manifest_loader.load_manifest()
  B. Fully manifest-driven (a single entry file auto-generates all test cases)
     -> Called by pytest_collection_modifyitems / pytest_generate_tests

Usage (in testcases/conftest.py or test file):
    def pytest_generate_tests(metafunc):
        from neuraxis_testkit.pytest_infra.collection import parametrize_from_manifest
        parametrize_from_manifest(metafunc, "case", "dirty_data")
"""

from __future__ import annotations
from neuraxis_testkit.log import get_logger

logger = get_logger(__name__)


def parametrize_from_manifest(metafunc, fixture_name: str, scenario: str) -> None:
    """
    pytest_generate_tests hook helper: parameterize the specified fixture from a manifest.

    Args:
        metafunc:     pytest_generate_tests callback parameter
        fixture_name: Name of the parameter in the test function that receives the case dict (e.g., "case")
        scenario:     Scenario name (corresponds to config/test_manifests/<scenario>.yaml)
    """
    if fixture_name not in metafunc.fixturenames:
        return

    from neuraxis_testkit.pytest_infra.manifest_loader import load_manifest, case_id_func

    test_file = str(metafunc.module.__file__)
    cases = load_manifest(test_file, scenario)

    # Support command-line filtering: --scenario dirty_data is registered by conftest
    scenario_filter = metafunc.config.getoption("--scenario", default=None)
    if scenario_filter and scenario_filter != scenario:
        cases = []

    tags_filter = metafunc.config.getoption("--tags", default=None)
    if tags_filter:
        wanted = {t.strip() for t in tags_filter.split(",")}
        cases = [c for c in cases if wanted & set(c.get("tags", []))]

    logger.info(f"Parameterizing [{scenario}]: {len(cases)} cases -> fixture '{fixture_name}'")
    metafunc.parametrize(fixture_name, cases, ids=[case_id_func(c) for c in cases])


def get_scenarios_from_items(items) -> set[str]:
    """Extract scenario set from collected items (for report aggregation)."""
    scenarios: set[str] = set()
    for item in items:
        for marker in item.iter_markers(name="scenario"):
            if marker.args:
                scenarios.add(str(marker.args[0]))
    return scenarios
