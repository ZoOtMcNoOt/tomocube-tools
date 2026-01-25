"""
TCF Registration - FL to HT registration functions.

This module provides functions for registering fluorescence data
to holotomography coordinate space.
"""

from __future__ import annotations

import logging

import numpy as np

from tomocube.core.types import RegistrationParams

logger = logging.getLogger(__name__)


def register_fl_to_ht(
    fl_data: np.ndarray,
    ht_shape: tuple[int, ...],
    params: RegistrationParams | None = None,
) -> np.ndarray:
    """
    Register FL data to HT coordinate space.

    Both modalities cover the same physical field of view (~230 um).
    Registration is simply resampling FL to match HT pixel dimensions.

    Args:
        fl_data: 2D slice (Y, X) or 3D volume (Z, Y, X)
        ht_shape: Target shape for output
        params: Registration parameters (for 3D Z-axis mapping)

    Returns:
        Registered FL data matching ht_shape dimensions

    Raises:
        ValueError: If fl_data is not 2D or 3D, or if arrays are empty
    """
    from scipy import ndimage

    # Input validation
    if fl_data.size == 0:
        raise ValueError("fl_data array is empty")
    if len(ht_shape) < 2:
        raise ValueError(f"ht_shape must have at least 2 dimensions, got {len(ht_shape)}")
    if any(dim <= 0 for dim in ht_shape):
        raise ValueError(f"ht_shape dimensions must be positive, got {ht_shape}")
    if np.any(np.isnan(fl_data)):
        logger.warning("fl_data contains NaN values, replacing with 0")
        fl_data = np.nan_to_num(fl_data, nan=0.0)

    if fl_data.ndim == 2:
        # 2D: simple resize
        ht_h, ht_w = ht_shape[-2], ht_shape[-1]
        fl_h, fl_w = fl_data.shape
        zoom_factors = (ht_h / fl_h, ht_w / fl_w)
        result: np.ndarray = np.asarray(ndimage.zoom(fl_data.astype(float), zoom_factors, order=1))
        return result

    elif fl_data.ndim == 3:
        # 3D: resize XY and interpolate Z
        if params is None:
            params = RegistrationParams()

        ht_z, ht_h, ht_w = ht_shape
        fl_z, fl_h, fl_w = fl_data.shape

        output = np.zeros((ht_z, ht_h, ht_w), dtype=np.float32)
        zoom_xy = (ht_h / fl_h, ht_w / fl_w)

        for ht_slice_idx in range(ht_z):
            # Physical Z position of this HT slice
            ht_z_um = ht_slice_idx * params.ht_res_z
            fl_z_um = ht_z_um - params.fl_offset_z
            fl_slice_idx = fl_z_um / params.fl_res_z

            if fl_slice_idx < 0 or fl_slice_idx >= fl_z - 1:
                continue

            # Interpolate between FL slices
            fl_z0 = int(np.floor(fl_slice_idx))
            fl_z1 = min(fl_z0 + 1, fl_z - 1)
            fz = fl_slice_idx - fl_z0

            fl_interp = (1 - fz) * fl_data[fl_z0] + fz * fl_data[fl_z1]
            output[ht_slice_idx] = ndimage.zoom(fl_interp.astype(float), zoom_xy, order=1)

        return output

    else:
        raise ValueError(f"Expected 2D or 3D array, got {fl_data.ndim}D")
