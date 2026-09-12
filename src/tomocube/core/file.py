"""
TCF File I/O - Classes for reading and loading TCF files.

This module provides:
    - TCFFile: High-level interface for TCF file metadata
    - TCFFileLoader: File loading and data access operations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tomocube.core.constants import (
    ATTR_DEVICE_MODEL_TYPE,
    ATTR_DEVICE_SERIAL,
    ATTR_MAGNIFICATION,
    ATTR_NA,
    ATTR_OFFSET_Z,
    ATTR_REG_ROTATION,
    ATTR_REG_TRANSLATION_X,
    ATTR_REG_TRANSLATION_Y,
    ATTR_RESOLUTION_X,
    ATTR_RESOLUTION_Y,
    ATTR_RESOLUTION_Z,
    ATTR_RI,
    ATTR_RI_MAX,
    ATTR_RI_MIN,
    ATTR_SOFTWARE_VERSION,
    DEFAULT_FL_RES_X,
    DEFAULT_FL_RES_Y,
    DEFAULT_FL_RES_Z,
    DEFAULT_HT_RES_X,
    DEFAULT_HT_RES_Y,
    DEFAULT_HT_RES_Z,
    PATH_DATA_3D,
    PATH_DATA_2D_MIP,
    PATH_DATA_3D_FL,
    PATH_FL_REGISTRATION,
    PATH_INFO_DEVICE,
    get_instrument_defaults,
)
from tomocube.core.exceptions import TCFFileError, TCFParseError
from tomocube.core.types import RegistrationParams

logger = logging.getLogger(__name__)


def _as_group(item: h5py.Group | h5py.Dataset | h5py.Datatype) -> h5py.Group:
    """Require a group, including when Python assertions are disabled."""
    if not isinstance(item, h5py.Group):
        raise TCFFileError(f"Expected an HDF5 group at {item.name}")
    return item


def _as_dataset(item: h5py.Group | h5py.Dataset | h5py.Datatype) -> h5py.Dataset:
    """Require a dataset and include its path in format errors."""
    if not isinstance(item, h5py.Dataset):
        raise TCFFileError(f"Expected an HDF5 dataset at {item.name}")
    return item


def _read_attribute(item: h5py.Group | h5py.Dataset, name: str, default: Any = None) -> Any:
    """Read one metadata value stored as either a scalar or singleton array."""
    if name not in item.attrs:
        return default
    value = np.asarray(item.attrs[name])
    if value.size != 1:
        raise TCFParseError(f"{item.name} attribute {name} must contain exactly one value")
    scalar = value.item()
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else scalar


def _read_float_attribute(
    item: h5py.Group | h5py.Dataset, name: str, default: float | None = None
) -> float | None:
    value = _read_attribute(item, name, default)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TCFParseError(f"{item.name} attribute {name} must be numeric") from error
    if not np.isfinite(number):
        raise TCFParseError(f"{item.name} attribute {name} must be finite")
    return number


def _volume_dataset(item: h5py.Group | h5py.Dataset | h5py.Datatype) -> h5py.Dataset:
    dataset = _as_dataset(item)
    if dataset.ndim != 3 or any(size == 0 for size in dataset.shape):
        raise TCFFileError(f"{dataset.name} must be a non-empty (Z, Y, X) volume; got {dataset.shape}")
    if dataset.dtype.kind not in "iuf":
        raise TCFFileError(f"{dataset.name} must contain real numeric values; got {dataset.dtype}")
    return dataset


def _timepoint_key(name: str) -> tuple:
    """Sort numeric acquisition keys numerically, including unpadded keys."""
    return (0, int(name), name) if name.isdecimal() else (1, name)


def _physical_ri(raw: np.ndarray) -> np.ndarray:
    """Apply the TCF RI scale consistently to volumes and stored projections."""
    scaled = np.issubdtype(raw.dtype, np.integer) or raw.max() > 100
    data = raw.astype(np.float32)
    return data / 10000.0 if scaled else data


@dataclass
class TCFFile:
    """
    High-level interface for TCF file data.

    Usage:
        with h5py.File(path, 'r') as f:
            tcf = TCFFile.from_hdf5(f)
            print(tcf.ht_shape)
            print(tcf.has_fluorescence)
    """

    # HT data info
    ht_shape: tuple[int, int, int] = (0, 0, 0)  # (Z, Y, X)
    ht_resolution: tuple[float, float, float] = (DEFAULT_HT_RES_Z, DEFAULT_HT_RES_Y, DEFAULT_HT_RES_X)  # (Z, Y, X) um/px
    ri_min: float | None = None
    ri_max: float | None = None

    # FL data info
    has_fluorescence: bool = False
    fl_channels: list[str] = field(default_factory=lambda: [])
    fl_shapes: dict[str, tuple[int, int, int]] = field(default_factory=lambda: {})
    fl_resolution: tuple[float, float, float] = (DEFAULT_FL_RES_Z, DEFAULT_FL_RES_Y, DEFAULT_FL_RES_X)  # (Z, Y, X) um/px

    # Registration
    registration: RegistrationParams | None = None

    # Timepoints
    timepoints: list[str] = field(default_factory=lambda: [])

    # Device info
    magnification: float | None = None
    numerical_aperture: float | None = None
    medium_ri: float | None = None
    device_model: str | None = None
    device_serial: str | None = None
    software_version: str | None = None

    @classmethod
    def from_hdf5(cls, f: h5py.File) -> TCFFile:
        """Create TCFFile from an open HDF5 file handle."""
        tcf = cls()

        # Read root attributes for device info first (needed for defaults)
        for attr, field_name in (
            (ATTR_DEVICE_MODEL_TYPE, "device_model"),
            (ATTR_DEVICE_SERIAL, "device_serial"),
            (ATTR_SOFTWARE_VERSION, "software_version"),
        ):
            value = _read_attribute(f, attr)
            if value is not None:
                setattr(tcf, field_name, str(value))

        # Device info from Info/Device
        if PATH_INFO_DEVICE in f:
            dev = _as_group(f[PATH_INFO_DEVICE])
            tcf.magnification = _read_float_attribute(dev, ATTR_MAGNIFICATION)
            tcf.numerical_aperture = _read_float_attribute(dev, ATTR_NA)
            tcf.medium_ri = _read_float_attribute(dev, ATTR_RI)

        # Get instrument-specific defaults
        inst_defaults = get_instrument_defaults(tcf.device_model, tcf.magnification)

        if PATH_DATA_3D not in f:
            raise TCFFileError(f"Missing required HT group: {PATH_DATA_3D}")
        group = _as_group(f[PATH_DATA_3D])
        tcf.timepoints = sorted(group.keys(), key=_timepoint_key)
        if not tcf.timepoints:
            raise TCFFileError(f"No HT timepoints in {PATH_DATA_3D}")
        # Validate dataset headers only; volume pixels remain lazy-loaded.
        for tp in tcf.timepoints:
            _volume_dataset(group[tp])
        ds = _volume_dataset(group[tcf.timepoints[0]])
        tcf.ht_shape = ds.shape
        for attr, field_name in ((ATTR_RI_MIN, "ri_min"), (ATTR_RI_MAX, "ri_max")):
            value = _read_float_attribute(ds, attr)
            if value is not None:
                setattr(tcf, field_name, value / 10000.0 if value > 100 else value)

        # Check for fluorescence
        if PATH_DATA_3D_FL in f:
            fl_group = _as_group(f[PATH_DATA_3D_FL])
            for ch in sorted(fl_group.keys()):
                ch_group = _as_group(fl_group[ch])
                ch_timepoints = sorted(ch_group.keys(), key=_timepoint_key)
                if ch_timepoints:
                    for tp in ch_timepoints:
                        _volume_dataset(ch_group[tp])
                    ch_ds = _volume_dataset(ch_group[ch_timepoints[0]])
                    tcf.fl_channels.append(ch)
                    tcf.fl_shapes[ch] = ch_ds.shape
            tcf.has_fluorescence = bool(tcf.fl_channels)

        # Always retain measured HT spacing, including acquisitions without FL.
        tcf.registration = load_registration_params(f, inst_defaults)
        params = tcf.registration
        tcf.ht_resolution = (params.ht_res_z, params.ht_res_y, params.ht_res_x)
        tcf.fl_resolution = (params.fl_res_z, params.fl_res_y, params.fl_res_x)

        return tcf


def load_registration_params(f: h5py.File, inst_defaults: dict[str, float] | None = None) -> RegistrationParams:
    """
    Load FL-HT registration parameters from TCF file.

    Args:
        f: Open HDF5 file handle
        inst_defaults: Instrument-specific defaults (from get_instrument_defaults)

    Returns:
        RegistrationParams with values from file (defaults used for missing attrs)

    Note:
        Missing paths or resolution attributes use instrument defaults. Present
        but invalid values raise TCFParseError instead of changing calibration.
    """
    # Use provided instrument defaults or fall back to global defaults
    if inst_defaults is None:
        inst_defaults = {
            "ht_res_xy": DEFAULT_HT_RES_X,
            "ht_res_z": DEFAULT_HT_RES_Z,
            "fl_res_xy": DEFAULT_FL_RES_X,
            "fl_res_z": DEFAULT_FL_RES_Z,
        }
    
    params = RegistrationParams()

    # XY registration params
    if PATH_FL_REGISTRATION in f:
        reg = _as_group(f[PATH_FL_REGISTRATION])
        if ATTR_REG_ROTATION in reg.attrs:
            params.rotation = _read_float_attribute(reg, ATTR_REG_ROTATION)
        # Note: params.scale from file is ignored - scaling is based purely on resolution ratios
        # since both HT and FL cover the same physical FOV
        if ATTR_REG_TRANSLATION_X in reg.attrs:
            params.translation_x = _read_float_attribute(reg, ATTR_REG_TRANSLATION_X)
        if ATTR_REG_TRANSLATION_Y in reg.attrs:
            params.translation_y = _read_float_attribute(reg, ATTR_REG_TRANSLATION_Y)
    else:
        logger.debug("FL registration path not found, using default parameters")

    for prefix, path in (("ht", PATH_DATA_3D), ("fl", PATH_DATA_3D_FL)):
        group = _as_group(f[path]) if path in f else None
        missing = []
        for axis, attr in (("x", ATTR_RESOLUTION_X), ("y", ATTR_RESOLUTION_Y), ("z", ATTR_RESOLUTION_Z)):
            default = inst_defaults[f"{prefix}_res_{'z' if axis == 'z' else 'xy'}"]
            value = _read_float_attribute(group, attr, default) if group is not None else default
            if value <= 0 or not np.isfinite(value):
                raise TCFParseError(f"{path} attribute {attr} must be positive and finite")
            setattr(params, f"{prefix}_res_{axis}", value)
            if group is not None and attr not in group.attrs:
                missing.append(attr)
        if missing:
            logger.warning("Missing %s attributes %s; using instrument defaults", path, ", ".join(missing))

    # FL Z offsets - read per-channel offsets
    channel_offsets: dict[str, float] = {}
    if PATH_DATA_3D_FL in f:
        fl_group = _as_group(f[PATH_DATA_3D_FL])
        for ch_name in sorted(fl_group.keys()):
            ch_path = f"{PATH_DATA_3D_FL}/{ch_name}"
            if ch_path in f:
                ch = _as_group(f[ch_path])
                if ATTR_OFFSET_Z in ch.attrs:
                    offset = _read_float_attribute(ch, ATTR_OFFSET_Z)
                    channel_offsets[ch_name] = offset

    if channel_offsets:
        params.channel_offsets_z = channel_offsets
        params.fl_offset_z = next(iter(channel_offsets.values()))

    return params


class TCFFileLoader:
    """
    Handles all file I/O operations for TCF files.

    Separates file loading logic from display logic, following SRP.

    Usage:
        loader = TCFFileLoader(path)
        loader.load()
        loader.load_timepoint(0)
        data = loader.data_3d
        loader.close()

    Or with context manager:
        with TCFFileLoader(path) as loader:
            loader.load_timepoint(0)
            print(loader.data_3d.shape)
    """

    def __init__(self, tcf_path: str | Path) -> None:
        """
        Initialize the loader.

        Args:
            tcf_path: Path to TCF file
        """
        self.tcf_path = Path(tcf_path)
        self._file: h5py.File | None = None
        self._tcf_info: TCFFile | None = None
        self._reg_params: RegistrationParams | None = None

        # Cached data
        self._data_3d: np.ndarray | None = None
        self._data_mip: np.ndarray | None = None
        self._fl_data: dict[str, np.ndarray] = {}
        self._current_timepoint: str | None = None

    @property
    def file(self) -> h5py.File:
        """Get HDF5 file handle (raises if not loaded)."""
        if self._file is None:
            raise RuntimeError("File not loaded. Call load() first.")
        return self._file

    @property
    def tcf_info(self) -> TCFFile:
        """Get TCF file info (raises if not loaded)."""
        if self._tcf_info is None:
            raise RuntimeError("File not loaded. Call load() first.")
        return self._tcf_info

    @property
    def reg_params(self) -> RegistrationParams:
        """Get registration parameters."""
        if self._reg_params is None:
            return RegistrationParams()
        return self._reg_params

    @property
    def data_3d(self) -> np.ndarray:
        """Get current 3D data array."""
        if self._data_3d is None:
            raise RuntimeError("No data loaded. Call load_timepoint() first.")
        return self._data_3d

    @property
    def data_mip(self) -> np.ndarray:
        """Get current MIP."""
        if self._data_mip is None:
            raise RuntimeError("No data loaded. Call load_timepoint() first.")
        return self._data_mip

    @property
    def fl_data(self) -> dict[str, np.ndarray]:
        """Get fluorescence data by channel."""
        return self._fl_data

    @property
    def timepoints(self) -> list[str]:
        """Get list of available timepoints."""
        return self.tcf_info.timepoints

    @property
    def current_timepoint(self) -> str:
        """Acquisition key for the currently loaded arrays."""
        if self._current_timepoint is None:
            raise RuntimeError("No data loaded. Call load_timepoint() first.")
        return self._current_timepoint

    @property
    def has_fluorescence(self) -> bool:
        """Check if file has fluorescence data."""
        return self.tcf_info.has_fluorescence

    @property
    def fl_channels(self) -> list[str]:
        """Get list of fluorescence channels."""
        return self.tcf_info.fl_channels

    def load(self) -> None:
        """
        Load and parse the TCF file.

        Opens the HDF5 file and extracts metadata. Does not load volume data
        until load_timepoint() is called.
        """
        if self._file is not None and self._file.id.valid:
            return
        self.close()
        logger.debug(f"Loading TCF file: {self.tcf_path}")
        print(f"Loading: {self.tcf_path.name}")

        file = h5py.File(self.tcf_path, "r")
        try:
            info = TCFFile.from_hdf5(file)
        except BaseException:
            file.close()
            raise
        self._file = file
        self._tcf_info = info
        self._reg_params = info.registration

        print(f"  Timepoints: {len(self.timepoints)}")
        print(f"  Shape: {self._tcf_info.ht_shape}")

        if self.has_fluorescence:
            print(f"  FL channels: {self.fl_channels}")

    def load_timepoint(self, idx: int) -> None:
        """
        Load data for a specific timepoint.

        Args:
            idx: Timepoint index
        """
        if idx < 0 or idx >= len(self.timepoints):
            raise IndexError(f"Timepoint index {idx} out of range [0, {len(self.timepoints)})")

        tp = self.timepoints[idx]

        # Stage all arrays before replacing the current acquisition. A failed
        # read must not mix new HT pixels with a previous MIP or FL channel.
        data_3d = _physical_ri(np.asarray(_volume_dataset(self.file[f"{PATH_DATA_3D}/{tp}"])))

        # Load or compute MIP (in physical RI units)
        mip_path = f"{PATH_DATA_2D_MIP}/{tp}"
        if mip_path in self.file:
            mip_ds = _as_dataset(self.file[mip_path])
            if mip_ds.shape != data_3d.shape[1:] or mip_ds.dtype.kind not in "iuf":
                raise TCFFileError(f"{mip_path} must be a numeric projection with shape {data_3d.shape[1:]}")
            data_mip = _physical_ri(np.asarray(mip_ds))
        else:
            data_mip = np.max(data_3d, axis=0)

        fl_data = {}
        for ch in self.fl_channels:
            path = f"{PATH_DATA_3D_FL}/{ch}/{tp}"
            if path in self.file:
                fl_data[ch] = np.asarray(_volume_dataset(self.file[path]))

        self._data_3d = data_3d
        self._data_mip = data_mip
        self._fl_data = fl_data
        self._current_timepoint = tp

    def get_fl_contrast(self, channel: str) -> tuple[float, float]:
        """
        Get suggested contrast range for a fluorescence channel.

        Args:
            channel: Channel name (e.g., "CH0")

        Returns:
            Tuple of (vmin, vmax) based on percentiles
        """
        if channel not in self._fl_data:
            return (0.0, 1.0)

        fl = self._fl_data[channel]
        if np.any(fl > 0):
            vmin = float(np.percentile(fl[fl > 0], 5))
            vmax = float(np.percentile(fl[fl > 0], 99))
            return (vmin, vmax)
        return (0.0, 1.0)

    def close(self) -> None:
        """Close the HDF5 file and release resources."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                logger.debug("Error closing HDF5 file", exc_info=True)
            finally:
                self._file = None

        self._data_3d = None
        self._data_mip = None
        self._fl_data.clear()
        self._tcf_info = None
        self._reg_params = None
        self._current_timepoint = None

    def __enter__(self) -> TCFFileLoader:
        """Context manager entry."""
        self.load()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.close()
