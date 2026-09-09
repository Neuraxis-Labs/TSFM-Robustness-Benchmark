#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit/utils/files.py -- File operation utility module

Provides unified CSV file read/write interface, supporting:
  - Multiple data format saving
  - Resume from breakpoint data appending
  - Unified error handling

Author: Janesong
Create Date: 2026/07/12.
"""

import json
from pathlib import Path
from typing import Any
import pandas as pd


class CSVFileError(Exception):
    """Custom exception for CSV file operations."""
    pass


def save_to_csv(
    result_csv_path_file: str | Path,
    data: list[dict[str, Any]] | list[list[Any]] | dict[str, Any] | pd.DataFrame,
    index: bool = False,
    mode: str = "w",
    encoding: str = "utf-8",
    columns: list[str] | None = None,
    strict_suffix: bool = True,
    **kwargs
) -> Path:
    """
    Unified save results to CSV file

    Args:
        result_csv_path_file: CSV file path (including filename), required parameter
        data: Data to save, supports the following formats:
            - List[Dict]: List of dictionaries, each dictionary represents one row of data
            - Dict: Single dictionary, converted to single row of data
            - pd.DataFrame: Directly save DataFrame
        index: Whether to save index, default False
        mode: Write mode, 'w'=overwrite write, 'a'=append write, default 'w'
        encoding: File encoding, default 'utf-8'
        columns: Column order to enforce. If provided, output columns will follow this order
        **kwargs: Additional pandas to_csv parameters

    Returns:
        Path: Saved file path object

    Raises:
        CSVFileError: Raised when file path or data is invalid

    Example:
        >>> # Save list of dictionaries
        >>> data = [
        ...     {"model": "model_a", "mae": 0.5, "rmse": 0.8},
        ...     {"model": "model_b", "mae": 0.6, "rmse": 0.9}
        ... ]
        >>> save_to_csv("./results/test.csv", data)

        >>> # Save single dictionary
        >>> result = {"model": "model_a", "mae": 0.5, "rmse": 0.8}
        >>> save_to_csv("./results/test.csv", result)

        >>> # Save DataFrame
        >>> df = pd.DataFrame({"model": ["a", "b"], "mae": [0.5, 0.6]})
        >>> save_to_csv("./results/test.csv", df)
    """
    # Validate required parameters
    if not result_csv_path_file:
        raise CSVFileError("result_csv_path_file parameter cannot be empty; must provide a filename including path")

    # Convert to Path object
    file_path = Path(result_csv_path_file)

    # Validate file extension
    # Suffix validation: strict mode requires ".csv"; non-strict only checks when a suffix exists
    suffix = file_path.suffix.lower()
    if strict_suffix and suffix != ".csv":
        raise CSVFileError(f"File extension must be .csv; current is: {file_path.suffix}")

    # Create parent directory (if not exists)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Data format conversion
    if isinstance(data, pd.DataFrame):
        df = data
        if columns is not None:
            df = df.copy()
            df.columns = columns
    elif isinstance(data, dict):
        # Single dictionary converted to single-row DataFrame, with optional column ordering
        df = pd.DataFrame([data], columns=columns)
    elif isinstance(data, list):
        if len(data) == 0:
            # Empty list, create empty DataFrame
            df = pd.DataFrame(columns=columns) if columns else pd.DataFrame()
        else:
            # List of dictionaries converted to DataFrame
            df = pd.DataFrame(data, columns=columns)
    else:
        raise CSVFileError(f"Unsupported data type: {type(data).__name__}")

    # Handle append mode
    if mode == "a" and file_path.exists() and file_path.stat().st_size > 0:
        # Append mode: do not write header
        header = False
    else:
        # Write mode: write header
        header = True

    # Save CSV
    try:
        df.to_csv(
            file_path,
            index=index,
            mode=mode,
            encoding=encoding,
            header=header,
            **kwargs
        )
    except Exception as exp:
        raise CSVFileError(f"Failed to save CSV file: {file_path}\nError: {exp}")

    return file_path

def append_to_csv(
    result_csv_path_file: str | Path,
    data: dict[str, Any] | list[dict[str, Any]],
    encoding: str = "utf-8",
    columns: list[str] | None = None,
    strict_suffix: bool = True,
) -> Path:
    """
    Append results to CSV file (resume from breakpoint scenario)

    If file does not exist, will create new file; if exists, will append data (without writing header)

    Args:
        result_csv_path_file: CSV file path (including filename), required parameter
        data: Data to append
            - Dict: Single row of data
            - List[Dict]: Multiple rows of data
        encoding: File encoding, default 'utf-8'
        columns: Column order to enforce when appending. If provided, output columns will follow this order.

    Returns:
        Path: Appended file path object

    Example:
        >>> result = {"model": "model_a", "mae": 0.5}
        >>> append_to_csv("./results/test.csv", result, columns=["model", "mae"])
    """
    return save_to_csv(
        result_csv_path_file=result_csv_path_file,
        data=data,
        mode="a",
        encoding=encoding,
        columns=columns
    )

def save_with_json_backup(
    result_csv_path_file: str | Path,
    data: list[dict[str, Any]] | pd.DataFrame,
    save_json: bool = True,
    index: bool = False,
    encoding: str = "utf-8",
    **kwargs
) -> tuple[Path, Path | None]:
    """
    Save results to CSV, with optional JSON backup

    Args:
        result_csv_path_file: CSV file path (including filename), required parameter
        data: Data to save
        save_json: Whether to also save JSON format, default True
        index: Whether to save index, default False
        encoding: File encoding, default 'utf-8'
        **kwargs: Additional parameters

    Returns:
        tuple[Path, Optional[Path]]: (CSV path, JSON path or None)

    Example:
        >>> data = [{"model": "a", "mae": 0.5}]
        >>> csv_path, json_path = save_with_json_backup("./results/test.csv", data)
    """
    # Save CSV
    csv_path = save_to_csv(
        result_csv_path_file=result_csv_path_file,
        data=data,
        index=index,
        encoding=encoding,
        **kwargs
    )

    json_path = None
    if save_json:
        # Generate JSON file path
        csv_path_obj = Path(result_csv_path_file)
        json_path = csv_path_obj.with_suffix(".json")

        # Convert data format
        if isinstance(data, pd.DataFrame):
            json_data = data.to_dict(orient="records")
        else:
            json_data = data

        # Save JSON
        with open(json_path, "w", encoding=encoding) as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    return csv_path, json_path


def read_csv_to_dataframe(
    result_csv_path_file: str | Path,
    encoding: str = "utf-8",
    **kwargs
) -> pd.DataFrame:
    """
    Read CSV file as DataFrame

    Args:
        result_csv_path_file: CSV file path (including filename), required parameter
        encoding: File encoding, default 'utf-8'
        **kwargs: Additional pandas read_csv parameters

    Returns:
        pd.DataFrame: Read data

    Raises:
        CSVFileError: Raised when file does not exist or read fails

    Example:
        >>> df = read_csv_to_dataframe("./results/test.csv")
    """
    if not result_csv_path_file:
        raise CSVFileError("result_csv_path_file parameter cannot be empty")

    file_path = Path(result_csv_path_file)

    if not file_path.exists():
        raise CSVFileError(f"File does not exist: {file_path}")

    try:
        return pd.read_csv(file_path, encoding=encoding, **kwargs)
    except Exception as exp:
        raise CSVFileError(f"Failed to read CSV file: {file_path}\nError: {exp}")

def read_csv_to_list(
    result_csv_path_file: str | Path,
    encoding: str = "utf-8"
) -> list[dict[str, Any]]:
    """
    Read CSV file as a list of dictionaries. (Generic CSV parser)

    Args:
        result_csv_path_file: CSV file path (including filename), required parameter
        encoding: File encoding, default 'utf-8'

    Returns:
        list[dict[str, Any]]: List of dictionaries, each dictionary represents one row of data

    Example:
        >>> data = read_csv_to_list("./results/test.csv")
        >>> # Returns: [{"model": "a", "mae": 0.5}, {"model": "b", "mae": 0.6}]
    """
    df = read_csv_to_dataframe(result_csv_path_file, encoding=encoding)
    return df.to_dict(orient="records")


def csv_exists_and_not_empty(result_csv_path_file: str | Path) -> bool:
    """
    Check if CSV file exists and is not empty (has at least one data row)

    Args:
        result_csv_path_file: CSV file path

    Returns:
        bool: True=file exists and has data; False=file does not exist or is empty

    Example:
        >>> if csv_exists_and_not_empty("./results/test.csv"):
        ...     df = read_csv_to_dataframe("./results/test.csv")
    """
    if not result_csv_path_file:
        return False

    file_path = Path(result_csv_path_file)
    if not file_path.exists():
        return False
    if file_path.stat().st_size == 0:
        return False

    # Check if file is empty (only header also counts as non-empty)
    try:
        df = pd.read_csv(file_path, nrows=1)
        return len(df) > 0
    except Exception:
        return False


def ensure_dir(path) -> Path:
    """
    Ensure a directory exists, creating it and any necessary parent directories.

    If the directory already exists, no error is raised. Returns the corresponding Path object.
    Accepts either a string or a Path object as input.

    Example:
        ensure_dir(RESULTS_DIR / "futureCovs" / "dirtyData")
    """
    p = Path(path)
    # If a file path is provided, create its parent directory instead.
    if p.is_file():
        p = p.parent
    p.mkdir(parents=True, exist_ok=True)
    return p

