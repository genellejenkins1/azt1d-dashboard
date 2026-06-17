"""Counterfactual metric utilities for the AZT1D dashboard.

This module provides small, well-defined transformations of CGM time series
and associated clinical metrics. The focus here is deliberately modest:

1. Allow the user to redefine the *target glucose range* and recompute
   time-in-range metrics without modifying the underlying glucose trace.
2. Allow the user to apply a *uniform CGM bias* (e.g., ±10 mg/dL) to model
   calibration error and recompute the same metrics.

These are "metric-level" counterfactuals: we do *not* attempt to simulate
physiology or treatment decisions. Instead, we change the definition of what
counts as "in range" or what the CGM would have reported under a constant
offset and then recompute summary statistics.

Mathematically, for a glucose trajectory g_1, ..., g_N (mg/dL), and a target
range [L, U], the standard metrics are defined as:

    TIR(L, U) = 100 * (1/N) * sum_{t=1}^N I(L <= g_t <= U)
    TBR(L)    = 100 * (1/N) * sum_{t=1}^N I(g_t <  L)
    TAR(U)    = 100 * (1/N) * sum_{t=1}^N I(g_t >  U),

where I(·) is the indicator function. These are consistent with the
International Consensus on Time in Range (TIR) for CGM data.

References
----------
- Battelino T. et al., "Clinical Targets for Continuous Glucose Monitoring
  Data Interpretation: Recommendations From the International Consensus on
  Time in Range", Diabetes Care, 2019.
- Hernán MA, Robins JM, *Causal Inference: What If*, Chapman & Hall/CRC, 2020.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .data_loader import SubjectDataLoader


#: Default lower bound of target range (mg/dL), aligned with SubjectDataLoader.
DEFAULT_LOWER: float = float(SubjectDataLoader.TARGET_RANGE_LOWER)

#: Default upper bound of target range (mg/dL), aligned with SubjectDataLoader.
DEFAULT_UPPER: float = float(SubjectDataLoader.TARGET_RANGE_UPPER)


@dataclass
class RangeMetrics:
    """Container for time-in-range style metrics.

    All values are expressed as percentages in [0, 100], except for the
    mean glucose which is in mg/dL.
    """

    mean_glucose: float
    tir: float  # Time in range
    tbr: float  # Time below range
    tar: float  # Time above range

    def as_dict(self) -> Dict[str, float]:
        """Return metrics as a plain dictionary (useful for DataFrame construction)."""

        return {
            "mean_glucose": float(self.mean_glucose),
            "tir": float(self.tir),
            "tbr": float(self.tbr),
            "tar": float(self.tar),
        }


def _validate_bounds(lower: float, upper: float) -> None:
    """Basic sanity checks for target range bounds.

    We require
    - lower < upper
    - both within a physiologically reasonable range [20, 600] mg/dL.
    """

    if not (20.0 <= lower < upper <= 600.0):
        raise ValueError(
            f"Invalid target range [{lower}, {upper}]. "
            "Expected 20.0 <= lower < upper <= 600.0."
        )


def compute_range_metrics(
    glucose: pd.Series,
    lower: float = DEFAULT_LOWER,
    upper: float = DEFAULT_UPPER,
) -> RangeMetrics:
    """Compute TIR/TBR/TAR-style metrics for a glucose series.

    Parameters
    ----------
    glucose:
        One-dimensional pandas Series of CGM values (mg/dL). NaNs are
        allowed; they are removed before computing proportions.
    lower:
        Lower bound of the target range (mg/dL).
    upper:
        Upper bound of the target range (mg/dL).

    Returns
    -------
    RangeMetrics
        Dataclass with mean glucose and the three standard proportions.

    Raises
    ------
    ValueError
        If all values are NaN or if the target range bounds are invalid.
    """

    _validate_bounds(lower, upper)

    g = glucose.astype("float64")
    mask = g.notna()
    if not mask.any():
        raise ValueError("Glucose series contains no finite values.")

    g_valid = g[mask]
    n = float(len(g_valid))

    mean_glucose = float(g_valid.mean())

    tir = 100.0 * float(((g_valid >= lower) & (g_valid <= upper)).sum()) / n
    tbr = 100.0 * float((g_valid < lower).sum()) / n
    tar = 100.0 * float((g_valid > upper).sum()) / n

    # Numerical robustness: clamp very small negatives due to float error.
    tir = max(0.0, min(100.0, tir))
    tbr = max(0.0, min(100.0, tbr))
    tar = max(0.0, min(100.0, tar))

    return RangeMetrics(
        mean_glucose=mean_glucose,
        tir=tir,
        tbr=tbr,
        tar=tar,
    )


def apply_uniform_offset(glucose: pd.Series, offset_mgdl: float) -> pd.Series:
    """Apply a uniform additive offset to a glucose series.

    This models a simple sensor calibration bias: every CGM value is shifted by
    the same number of mg/dL. The transformation is purely algebraic:

        g'_t = g_t + b,

    where ``b`` is ``offset_mgdl``.

    Parameters
    ----------
    glucose:
        One-dimensional pandas Series of CGM values (mg/dL).
    offset_mgdl:
        Additive bias (mg/dL) applied to all values (can be positive or
        negative).

    Returns
    -------
    pd.Series
        Series of the same shape, with the offset applied (NaNs preserved).
    """

    g = glucose.astype("float64")
    return g + float(offset_mgdl)


def compute_offset_counterfactual_metrics(
    glucose: pd.Series,
    offset_mgdl: float,
    lower: float = DEFAULT_LOWER,
    upper: float = DEFAULT_UPPER,
) -> RangeMetrics:
    """Compute range metrics after applying a uniform CGM bias.

    This is a composition of :func:`apply_uniform_offset` and
    :func:`compute_range_metrics`.

    Parameters
    ----------
    glucose:
        CGM time series (mg/dL).
    offset_mgdl:
        Additive bias (mg/dL) applied to all values before metric computation.
    lower, upper:
        Target range bounds (mg/dL).
    """

    shifted = apply_uniform_offset(glucose, offset_mgdl)
    return compute_range_metrics(shifted, lower=lower, upper=upper)