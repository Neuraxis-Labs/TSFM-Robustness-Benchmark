#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.models
"""
from dataclasses import dataclass, field

@dataclass
class ForecastResult:
    """Time series forecast result"""
    timestamp: str
    predicted: float
    actual: float
    error: float
    metrics: dict[str, float] = field(default_factory=dict)

@dataclass
class BatchForecastReport:
    """Batch forecast report"""
    model_name: str
    dataset_name: str
    results: list[ForecastResult]
    summary_metrics: dict[str, float]
    created_at: str
