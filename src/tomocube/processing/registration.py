"""
TCF Registration - FL to HT registration functions.

This module provides functions for registering fluorescence data
to holotomography coordinate space using affine transforms.
"""

from __future__ import annotations

import logging

import numpy as np

from tomocube.core.types import RegistrationParams
from tomocube.core.config import vprint

logger = logging.getLogger(__name__)


def _apply_affine_transform_2d(
    fl_slice: np.ndarray,
    ht_shape: tuple[int, int],
    params: RegistrationParams,
) -> np.ndarray:
    """
    Apply affine transform (rotation, scale, translation) to a 2D FL slice.

    The transform maps FL pixel coordinates to HT pixel coordinates:
    1. Convert FL pixels to physical coordinates (um)
    2. Apply rotation around FL center
    3. Apply scale correction
    4. Apply translation offset
    5. Convert back to HT pixel coordinates

    Args:
        fl_slice: 2D FL data (Y, X)
        ht_shape: Target (H, W) in HT pixels
        params: Registration parameters

    Returns:
        Transformed FL slice matching ht_shape
    """
    from scipy import ndimage

    fl_h, fl_w = fl_slice.shape
    ht_h, ht_w = ht_shape

    # Centers in pixels
    fl_center_y, fl_center_x = fl_h / 2, fl_w / 2
    ht_center_y, ht_center_x = ht_h / 2, ht_w / 2

    # Resolution ratios (FL pixels per HT pixel)
    # Both HT and FL cover the same physical FOV, so scaling is purely based on resolution
    scale_y = params.fl_res_y / params.ht_res_y  # FL pixels per HT pixel in Y
    scale_x = params.fl_res_x / params.ht_res_x  # FL pixels per HT pixel in X

    # Build affine transform matrix that maps HT coords -> FL coords
    # We need the inverse transform for ndimage.affine_transform
    #
    # Forward: FL -> HT
    #   1. Translate FL to origin (center)
    #   2. Scale by resolution ratio
    #   3. Rotate
    #   4. Translate by offset
    #   5. Scale to HT pixel coords
    #   6. Translate to HT center
    #
    # We need inverse: HT -> FL

    cos_r = np.cos(params.rotation)
    sin_r = np.sin(params.rotation)

    # Translation in FL pixels
    trans_y = params.translation_y / params.fl_res_y
    trans_x = params.translation_x / params.fl_res_x

    # Affine matrix for HT -> FL mapping
    # For each HT pixel (y, x), find the corresponding FL pixel
    #
    # Steps (working backwards from HT to FL):
    # 1. Offset from HT center
    # 2. Apply inverse scale (HT -> physical -> FL scale)
    # 3. Apply inverse rotation
    # 4. Apply inverse translation
    # 5. Add FL center

    # Build 3x3 affine matrix [y', x', 1] = M @ [y, x, 1]
    # where (y, x) is HT coord and (y', x') is FL coord

    # Rotation matrix (inverse = transpose for rotation)
    R_inv = np.array([
        [cos_r, sin_r],
        [-sin_r, cos_r]
    ])

    # Scale matrix (inverse) - just resolution ratios
    S_inv = np.array([
        [scale_y, 0],
        [0, scale_x]
    ])

    # Combined rotation and scale
    RS = R_inv @ S_inv

    # Full transform:
    # fl_coord = RS @ (ht_coord - ht_center) - trans + fl_center
    # fl_coord = RS @ ht_coord - RS @ ht_center - trans + fl_center

    # Offset term
    offset = -RS @ np.array([ht_center_y, ht_center_x]) - np.array([trans_y, trans_x]) + np.array([fl_center_y, fl_center_x])

    # Apply affine transform
    # ndimage.affine_transform uses: output[o] = input[matrix @ o + offset]
    result = ndimage.affine_transform(
        fl_slice.astype(np.float32),
        RS,
        offset=offset,
        output_shape=(ht_h, ht_w),
        order=1,  # bilinear interpolation
        mode='constant',
        cval=0.0
    )

    return result


def register_fl_to_ht(
    fl_data: np.ndarray,
    ht_shape: tuple[int, ...],
    params: RegistrationParams | None = None,
    channel: str | None = None,
    z_offset_mode: str = "start",
) -> np.ndarray:
    """
    Register FL data to HT coordinate space.

    Applies the full affine transformation including:
    - Rotation (from params.rotation)
    - XY translation (from params.translation_x, translation_y)
    - Resolution resampling (FL and HT have different pixel sizes but same FOV)
    - Z-axis interpolation (for 3D data)

    Note: Scaling is determined purely by resolution ratios since both HT and FL
    cover the same physical field of view (typically 230×230 µm).

    Args:
        fl_data: 2D slice (Y, X) or 3D volume (Z, Y, X)
        ht_shape: Target shape for output
        params: Registration parameters from TCF file
        channel: FL channel name (e.g., "CH0") for per-channel Z offset
        z_offset_mode: How to interpret fl_offset_z:
            - "start": OffsetZ is HT Z position where FL slice 0 starts (default)
            - "center": OffsetZ is HT Z position of FL volume center
            - "auto": Center FL on HT volume, ignoring OffsetZ

    Returns:
        Registered FL data matching ht_shape dimensions

    Raises:
        ValueError: If fl_data is not 2D or 3D, or if arrays are empty
    """
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

    if params is None:
        params = RegistrationParams()

    # Get channel-specific Z offset from file
    file_offset_z = params.get_offset_z(channel)

    # Verbose output
    vprint(f"[registration] FL → HT registration")
    vprint(f"  FL shape:       {fl_data.shape}")
    vprint(f"  HT shape:       {ht_shape}")
    vprint(f"  FL resolution:  {params.fl_res_x:.4f} × {params.fl_res_y:.4f} × {params.fl_res_z:.4f} µm/px")
    vprint(f"  HT resolution:  {params.ht_res_x:.4f} × {params.ht_res_y:.4f} × {params.ht_res_z:.4f} µm/px")
    vprint(f"  Rotation:       {np.degrees(params.rotation):.3f}°")
    vprint(f"  Translation:    ({params.translation_x:.2f}, {params.translation_y:.2f}) µm")
    vprint(f"  Scale factor:   {params.scale:.4f}")
    vprint(f"  Z offset mode:  {z_offset_mode}")
    vprint(f"  File offset Z:  {file_offset_z:.2f} µm" + (f" (channel {channel})" if channel else ""))

    if fl_data.ndim == 2:
        # 2D: apply full affine transform
        ht_h, ht_w = ht_shape[-2], ht_shape[-1]
        return _apply_affine_transform_2d(fl_data, (ht_h, ht_w), params)

    elif fl_data.ndim == 3:
        # 3D: apply affine transform to each XY slice with Z interpolation
        ht_z, ht_h, ht_w = ht_shape
        fl_z, fl_h, fl_w = fl_data.shape

        # Calculate the effective Z offset based on mode
        ht_total_z_um = ht_z * params.ht_res_z
        fl_total_z_um = fl_z * params.fl_res_z
        ht_center_z_um = ht_total_z_um / 2
        fl_center_z_um = fl_total_z_um / 2

        if z_offset_mode == "auto":
            # Smart alignment: find where signal actually is in each volume
            # and align those regions (center-of-mass based)

            # Find FL signal center (intensity-weighted Z position)
            fl_z_profile = np.sum(fl_data, axis=(1, 2))  # Sum each Z slice
            fl_z_profile = np.maximum(fl_z_profile, 0)  # Ensure non-negative
            fl_total_signal = np.sum(fl_z_profile)

            if fl_total_signal > 0:
                # Weighted average Z position (center of mass)
                fl_signal_center_slice = np.sum(np.arange(fl_z) * fl_z_profile) / fl_total_signal
                fl_signal_center_um = fl_signal_center_slice * params.fl_res_z
            else:
                # Fallback to geometric center if no signal
                fl_signal_center_um = fl_center_z_um

            # Align FL signal center to HT volume center
            effective_offset_z = ht_center_z_um - fl_signal_center_um
            logger.info(f"Z offset mode 'auto': aligning FL signal center ({fl_signal_center_um:.2f} µm) "
                       f"to HT center ({ht_center_z_um:.2f} µm), offset={effective_offset_z:.2f} µm")
            vprint(f"  AUTO mode: FL signal center at Z={fl_signal_center_um:.2f} µm")
            vprint(f"  AUTO mode: HT volume center at Z={ht_center_z_um:.2f} µm")
            vprint(f"  AUTO mode: Effective offset = {effective_offset_z:.2f} µm")
        elif z_offset_mode == "center":
            # OffsetZ is the HT Z position of FL center
            effective_offset_z = file_offset_z - fl_center_z_um
            logger.info(f"Z offset mode 'center': file_offset={file_offset_z:.2f}, effective={effective_offset_z:.2f} µm")
            vprint(f"  CENTER mode: FL center at Z={fl_center_z_um:.2f} µm")
            vprint(f"  CENTER mode: File offset = {file_offset_z:.2f} µm")
            vprint(f"  CENTER mode: Effective offset = {effective_offset_z:.2f} µm")
        else:  # "start" (default)
            # OffsetZ is the HT Z position where FL slice 0 starts
            effective_offset_z = file_offset_z
            logger.debug(f"Z offset mode 'start': offset={effective_offset_z:.2f} µm")
            vprint(f"  START mode: Effective offset = {effective_offset_z:.2f} µm")

        # Log Z coverage info
        fl_start_z_um = effective_offset_z
        fl_end_z_um = effective_offset_z + fl_total_z_um
        vprint(f"  FL covers HT Z range: {fl_start_z_um:.2f} - {fl_end_z_um:.2f} µm")
        vprint(f"  HT Z range: 0.00 - {ht_total_z_um:.2f} µm")

        output = np.zeros((ht_z, ht_h, ht_w), dtype=np.float32)

        for ht_slice_idx in range(ht_z):
            # Physical Z position of this HT slice
            ht_z_um = ht_slice_idx * params.ht_res_z
            # Convert to FL coordinate space
            fl_z_um = ht_z_um - effective_offset_z
            fl_slice_idx = fl_z_um / params.fl_res_z

            if fl_slice_idx < 0 or fl_slice_idx >= fl_z - 1:
                continue

            # Interpolate between FL slices in Z
            fl_z0 = int(np.floor(fl_slice_idx))
            fl_z1 = min(fl_z0 + 1, fl_z - 1)
            fz = fl_slice_idx - fl_z0

            fl_interp = (1 - fz) * fl_data[fl_z0] + fz * fl_data[fl_z1]

            # Apply XY affine transform
            output[ht_slice_idx] = _apply_affine_transform_2d(
                fl_interp, (ht_h, ht_w), params
            )

        return output

    else:
        raise ValueError(f"Expected 2D or 3D array, got {fl_data.ndim}D")
