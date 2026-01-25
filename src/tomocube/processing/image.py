"""
TCF Image Processing - Image normalization and analysis utilities.

This module provides functions for normalizing and analyzing
microscopy image data.
"""

from __future__ import annotations

import logging

import numpy as np

from tomocube.core.constants import DEFAULT_PERCENTILE_HIGH, DEFAULT_PERCENTILE_LOW

logger = logging.getLogger(__name__)


def normalize_image(
    img: np.ndarray,
    percentile_low: float = DEFAULT_PERCENTILE_LOW,
    percentile_high: float = DEFAULT_PERCENTILE_HIGH,
    use_nonzero: bool = True,
) -> np.ndarray:
    """
    Normalize image to [0, 1] range using percentile clipping.

    Args:
        img: Input image
        percentile_low: Lower percentile for clipping (0-100)
        percentile_high: Upper percentile for clipping (0-100)
        use_nonzero: If True, compute percentiles from non-zero pixels only

    Returns:
        Normalized image in [0, 1] range

    Raises:
        ValueError: If percentile values are invalid
    """
    # Input validation
    if img.size == 0:
        return np.array([], dtype=float)

    if not (0 <= percentile_low <= 100):
        raise ValueError(f"percentile_low must be in [0, 100], got {percentile_low}")
    if not (0 <= percentile_high <= 100):
        raise ValueError(f"percentile_high must be in [0, 100], got {percentile_high}")
    if percentile_low >= percentile_high:
        raise ValueError(
            f"percentile_low ({percentile_low}) must be less than "
            f"percentile_high ({percentile_high})"
        )

    # Handle NaN values
    if np.any(np.isnan(img)):
        logger.warning("Image contains NaN values, replacing with 0")
        img = np.nan_to_num(img, nan=0.0)

    # Handle all-zeros case
    if not np.any(img):
        return np.zeros_like(img, dtype=float)

    if use_nonzero and np.any(img > 0):
        values = img[img > 0]
    else:
        values = img.ravel()

    p_low, p_high = np.percentile(values, [percentile_low, percentile_high])

    if p_high - p_low < 1e-10:
        return np.zeros_like(img, dtype=float)

    return np.clip((img - p_low) / (p_high - p_low), 0, 1)


def normalize_with_bounds(
    img: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """
    Normalize image to [0, 1] range using explicit bounds.

    Args:
        img: Input image
        vmin: Minimum value (maps to 0)
        vmax: Maximum value (maps to 1)

    Returns:
        Normalized image in [0, 1] range
    """
    if vmax - vmin < 1e-10:
        return np.zeros_like(img, dtype=float)

    return np.clip((img - vmin) / (vmax - vmin), 0, 1)


def compute_overlap_score(
    ht_image: np.ndarray,
    fl_image: np.ndarray,
    ht_percentile: float = 75.0,
    fl_percentile: float = 85.0,
) -> float:
    """
    Compute spatial overlap score between HT and FL images.

    Higher scores indicate better co-localization of bright regions.

    Args:
        ht_image: HT image (any shape)
        fl_image: FL image (same shape as ht_image)
        ht_percentile: Threshold percentile for HT
        fl_percentile: Threshold percentile for FL

    Returns:
        Overlap score in [0, 1] range
    """
    ht_thresh = np.percentile(ht_image, ht_percentile)
    ht_mask = ht_image > ht_thresh

    fl_nonzero = fl_image[fl_image > 0]
    if len(fl_nonzero) < 100:
        return 0.0

    fl_thresh = np.percentile(fl_nonzero, fl_percentile)
    fl_mask = fl_image > fl_thresh

    if np.sum(fl_mask) == 0:
        return 0.0

    return float(np.sum(fl_mask & ht_mask) / np.sum(fl_mask))
