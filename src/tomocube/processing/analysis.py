"""Header inventory and bounded, calibrated descriptive volume measurements.

Measurements use each channel's native grid. An ROI or threshold is not a cell
segmentation, and the spatial RI integral is not a dry-mass estimate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tomocube.core.constants import (
    ATTR_OFFSET_Z,
    ATTR_RESOLUTION_X,
    ATTR_RESOLUTION_Y,
    ATTR_RESOLUTION_Z,
    PATH_DATA_3D,
    PATH_DATA_3D_FL,
)
from tomocube.core.exceptions import TCFFileError
from tomocube.core.file import (
    TCFFile,
    _block_depth,
    _dataset_ri_divisor,
    _read_volume_blocks,
    _timepoint_key,
    _volume_dataset,
    _volume_roi,
)


def _describe_volume(
    file: h5py.File, info: TCFFile, key: str, index: int | None, channel: str
) -> dict[str, Any]:
    is_ht = channel == "HT"
    path = f"{PATH_DATA_3D}/{key}" if is_ht else f"{PATH_DATA_3D_FL}/{channel}/{key}"
    spacing = info.ht_resolution if is_ht else info.fl_resolution
    calibration_group = file[PATH_DATA_3D if is_ht else PATH_DATA_3D_FL]
    dataset = _volume_dataset(file[path]) if path in file else None
    count = math.prod(dataset.shape) if dataset is not None else None
    return {
        "timepoint_index": index,
        "acquisition_key": key,
        "channel": channel,
        "status": "available" if dataset is not None else "missing",
        "dataset_path": "/" + path.lstrip("/"),
        "shape_zyx": list(dataset.shape) if dataset is not None else None,
        "dtype": str(dataset.dtype) if dataset is not None else None,
        "spacing_zyx_um": list(spacing),
        "spacing_zyx_source": [
            "metadata" if attribute in calibration_group.attrs else "instrument_default"
            for attribute in (ATTR_RESOLUTION_Z, ATTR_RESOLUTION_Y, ATTR_RESOLUTION_X)
        ],
        "voxel_count": count,
        "size_bytes": count * dataset.dtype.itemsize if dataset is not None else None,
        "value_unit": "RI" if is_ht else "count",
        "offset_z_um": None if is_ht else info.registration.get_offset_z(channel),
        "offset_z_source": None if is_ht else (
            "metadata" if ATTR_OFFSET_Z in file[f"{PATH_DATA_3D_FL}/{channel}"].attrs else "fallback"
        ),
    }


def _read_info(file: h5py.File) -> TCFFile:
    info = TCFFile.from_hdf5(file)
    if "HT" in info.fl_channels:
        raise TCFFileError("Fluorescence channel name 'HT' conflicts with the reserved HT selector")
    return info


def inspect_acquisition(path: str | Path) -> dict[str, Any]:
    """Return JSON-compatible inventory without reading any volume pixels.

    Every HT acquisition and FL channel is represented, including missing
    channel/acquisition pairs. FL acquisition keys without an HT volume have a
    null ``timepoint_index`` and are inspectable but not analysis timepoints.
    ``size_bytes`` is the uncompressed native array size, not HDF5 storage size.
    Spacing is in native ZYX order; ``spacing_zyx_source`` distinguishes measured
    attributes from instrument defaults. Registration is retained as metadata.
    """
    source = Path(path).resolve()
    with h5py.File(source, "r") as file:
        info = _read_info(file)
        indices = {key: index for index, key in enumerate(info.timepoints)}
        keys = set(info.timepoints)
        for channel in info.fl_channels:
            keys.update(file[f"{PATH_DATA_3D_FL}/{channel}"].keys())
        volumes = [
            _describe_volume(file, info, key, indices.get(key), channel)
            for key in sorted(keys, key=_timepoint_key)
            for channel in ["HT", *info.fl_channels]
        ]
        metadata = {
            name: getattr(info, name)
            for name in (
                "device_model", "device_serial", "software_version", "magnification",
                "numerical_aperture", "medium_ri", "ri_min", "ri_max",
            )
        }
        metadata["registration"] = asdict(info.registration)
        return {
            "source": str(source),
            "axis_order": "ZYX",
            "length_unit": "um",
            "timepoints": list(info.timepoints),
            "fl_channels": list(info.fl_channels),
            "metadata": metadata,
            "volumes": volumes,
        }


def _select_timepoints(values: Sequence[int] | None, count: int) -> list[int]:
    if values is None:
        return list(range(count))
    if isinstance(values, (str, bytes)):
        raise ValueError("timepoints must be a non-empty sequence of integer indices")
    try:
        selected = list(values)
    except TypeError as error:
        raise ValueError("timepoints must be a non-empty sequence of integer indices") from error
    if not selected or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in selected
    ):
        raise ValueError("timepoints must be a non-empty sequence of integer indices")
    if any(value < 0 or value >= count for value in selected):
        raise IndexError(f"Timepoint indices must be in [0, {count})")
    if len(set(selected)) != len(selected):
        raise ValueError("timepoints must not contain duplicates")
    return sorted(int(value) for value in selected)


def _select_channels(values: Sequence[str] | None, available: list[str]) -> list[str]:
    if values is None:
        return ["HT"]
    if isinstance(values, (str, bytes)):
        raise ValueError("channels must be a non-empty sequence of channel names")
    try:
        selected = list(values)
    except TypeError as error:
        raise ValueError("channels must be a non-empty sequence of channel names") from error
    if not selected or any(not isinstance(value, str) for value in selected):
        raise ValueError("channels must be a non-empty sequence of channel names")
    unknown = [value for value in selected if value not in available]
    if unknown:
        raise ValueError(f"Unknown channels: {', '.join(unknown)}; available: {', '.join(available)}")
    if len(set(selected)) != len(selected):
        raise ValueError("channels must not contain duplicates")
    return selected


def _measure_volume(
    dataset: h5py.Dataset,
    row: dict[str, Any],
    roi: tuple[tuple[int, int], ...],
    threshold: float | None,
    block_depth: int,
) -> dict[str, Any]:
    divisor = _dataset_ri_divisor(dataset, block_depth=block_depth) if row["channel"] == "HT" else None
    count = 0
    mean = m2 = 0.0
    origin = None
    minimum = maximum = None
    selected_count = 0
    # Merge block means/second moments after shifting by the first value. This
    # avoids the catastrophic cancellation in E[x*x] - E[x]**2 for high offsets.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for block in _read_volume_blocks(dataset, roi=roi, block_depth=block_depth, ri_divisor=divisor):
                low, high = block.min().item(), block.max().item()
                minimum = low if minimum is None else min(minimum, low)
                maximum = high if maximum is None else max(maximum, high)
                if threshold is not None:
                    selected_count += int(np.count_nonzero(block >= threshold))
                if origin is None:
                    origin = float(block.flat[0])
                shifted = block.astype(np.float64) - origin
                block_count = int(block.size)
                block_mean = float(shifted.mean())
                centered = shifted - block_mean
                block_m2 = float(np.sum(centered * centered, dtype=np.float64))
                total_count = count + block_count
                delta = block_mean - mean
                m2 += block_m2 + delta * delta * (count / total_count) * block_count
                mean += delta * (block_count / total_count)
                count = total_count
    except (FloatingPointError, OverflowError) as error:
        raise ValueError(f"{dataset.name} measurements exceed the float64 numeric range") from error
    mean += origin
    voxel_volume = math.prod(row["spacing_zyx_um"])
    volume = count * voxel_volume
    integral = mean * volume
    std = math.sqrt(max(m2 / count, 0.0))
    if not all(math.isfinite(value) for value in (voxel_volume, volume, mean, std, integral)) or voxel_volume <= 0:
        raise ValueError(f"{dataset.name} calibrated measurements exceed the float64 numeric range")
    row.update({
        "status": "ok",
        "roi_zyx": [list(pair) for pair in roi],
        "voxel_count": count,
        "volume_um3": volume,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
        "integral": integral,
        "ri_divisor": divisor,
        "selected_voxel_count": selected_count if threshold is not None else None,
        "selected_volume_um3": selected_count * voxel_volume if threshold is not None else None,
    })
    return row


def analyze_acquisition(
    path: str | Path,
    *,
    timepoints: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    roi: Sequence[Sequence[int]] | None = None,
    threshold: float | None = None,
    block_depth: int = 16,
) -> list[dict[str, Any]]:
    """Measure native volumes with memory bounded by Z-block depth.

    Defaults select HT at every numeric-sorted acquisition. ``timepoints`` are
    zero-based HT indices, returned in acquisition order. ``channels`` selects
    ``HT`` and/or explicit FL names. A known channel absent at an acquisition
    produces a ``status='missing'`` row with null statistics; unknown names fail.

    ``roi=((z0,z1),(y0,y1),(x0,x1))`` contains native integer half-open bounds,
    checked against every selected dataset. The same indices need not describe
    the same physical region across channels. No registration is applied.

    ``std`` is population standard deviation. ``integral`` is mean intensity
    times ROI volume in cubic micrometers. Optional threshold selection uses
    ``value >= threshold`` in physical RI or native FL counts. Results include
    the threshold and units, and do not imply objects, segmentation or dry mass.

    Statistics use float64 arithmetic. Integer HT storage is divided by 10000;
    floating HT requires a full bounded scan to determine a single dataset-wide
    divisor (10000 if any value exceeds 100, otherwise 1). That scan includes
    pixels outside the ROI and rejects nonfinite values. FL reads only the ROI.
    Original source data and calibration are never modified.
    """
    depth = _block_depth(block_depth)
    if roi is not None:
        roi = _volume_roi(roi, None)
    if threshold is not None:
        try:
            if isinstance(threshold, (bool, np.bool_)):
                raise ValueError
            threshold = float(threshold)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("threshold must be a finite number") from error
        if not math.isfinite(threshold):
            raise ValueError("threshold must be a finite number")
    source = Path(path).resolve()
    with h5py.File(source, "r") as file:
        info = _read_info(file)
        indices = _select_timepoints(timepoints, len(info.timepoints))
        selected_channels = _select_channels(channels, ["HT", *info.fl_channels])
        rows = []
        # Validate all requested bounds before any pixel reads.
        selections = []
        for index in indices:
            for channel in selected_channels:
                row = _describe_volume(file, info, info.timepoints[index], index, channel)
                bounds = _volume_roi(roi, tuple(row["shape_zyx"])) if row["shape_zyx"] is not None else None
                selections.append((row, bounds))
        for row, bounds in selections:
            row.update({
                "source": str(source),
                "axis_order": "ZYX",
                "roi_zyx": [list(pair) for pair in roi] if roi is not None else None,
                "threshold": threshold,
                "threshold_operator": ">=" if threshold is not None else None,
                "integral_unit": f"{row['value_unit']}*um^3",
                "voxel_count": None,
                "volume_um3": None,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "integral": None,
                "ri_divisor": None,
                "selected_voxel_count": None,
                "selected_volume_um3": None,
            })
            if bounds is not None:
                dataset = _volume_dataset(file[row["dataset_path"]])
                _measure_volume(dataset, row, bounds, threshold, depth)
            rows.append(row)
        return rows
