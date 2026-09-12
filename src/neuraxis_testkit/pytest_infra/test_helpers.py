#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.pytest_infra.test_helpers - pytest test-process helpers (glue layer)

Usage Examples:
    from neuraxis_testkit.pytest_infra.test_helpers import load_test_data, load_and_validate

    data = load_test_data("tests/fixtures/sample.json")

    prediction, response = load_and_validate(
        "tests/fixtures/sample.json",
        forecast_func=my_forecast_func,
        extract_func=my_extract_func,
        validate_func=assert_prediction_valid,
        min_length=1,
    )

Responsibilities:
  - Only handles test-process concerns: loading fixture data, orchestrating call -> extract -> validate flow.
  - Does not implement concrete assertion logic. Generic assertions live in utils/assertions.py; business assertions live in the business layer.
  - May use pytest APIs (e.g. pytest.fail). This is the essential difference from the utils layer.
"""

import json, pytest
from pathlib import Path
from neuraxis_testkit.log import get_logger

logger = get_logger("test_helpers")


def load_test_data(filepath: str | Path) -> dict:
    """
    Load a JSON test data file; if the file is missing, fail the current test case directly (pytest-context behavior).
    """
    fp = Path(filepath)
    if not fp.exists():
        pytest.fail(f"Test data file does not exist: {fp}")

    with open(fp, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as jsonErr:
            # A truncated JSON file is itself one of the test scenarios; allow partial parsing.
            logger.warning(f"JSON parsing failed ({fp}): {json_err}")
            return {"_parse_error": str(jsonErr), "_raw": ""}


def load_and_validate(
    filepath: str | Path,
    forecast_func,
    extract_func,
    validate_func,
    min_length: int = 1,
) -> tuple:
    """
    Complete in one step: load data -> call forecast -> extract result -> delegate validation.
    validate_func is injected by the caller (the business layer passes assert_prediction_valid).
    Returns (prediction, response).
    """
    data = load_test_data(filepath)
    resp = forecast_func(data)
    pred = extract_func(resp)

    if pred is not None:
        validate_func(pred, min_length=min_length)

    return pred, resp
