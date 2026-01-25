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
from typing import TYPE_CHECKING

import h5py
import numpy as np

from tomocube.core.constants import (
    ATTR_MAGNIFICATION,
    ATTR_NA,
    ATTR_OFFSET_Z,
    ATTR_REG_ROTATION,
    ATTR_REG_SCALE,
    ATTR_REG_TRANSLATION_X,
    ATTR_REG_TRANSLATION_Y,
    ATTR_RESOLUTION_X,
    ATTR_RESOLUTION_Y,
    ATTR_RESOLUTION_Z,
    ATTR_RI,
    ATTR_RI_MAX,
    ATTR_RI_MIN,
    DEFAULT_FL_RES_X,
    DEFAULT_FL_RES_Y,
    DEFAULT_FL_RES_Z,
    DEFAULT_HT_RES_X,
    DEFAULT_HT_RES_Y,
    DEFAULT_HT_RES_Z,
    PATH_DATA_3D,
    PATH_DATA_3D_FL,
    PATH_FL_REGISTRATION,
    PATH_INFO_DEVICE,
)
from tomocube.core.types import RegistrationParams

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _as_group(item: h5py.Group | h5py.Dataset | h5py.Datatype) -> h5py.Group:
    """Cast h5py item to Group, asserting type."""
    assert isinstance(item, h5py.Group), f"Expected Group, got {type(item)}"
    return item


def _as_dataset(item: h5py.Group | h5py.Dataset | h5py.Datatype) -> h5py.Dataset:
    """Cast h5py item to Dataset, asserting type."""
    assert isinstance(item, h5py.Dataset), f"Expected Dataset, got {type(item)}"
    return item


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

    @classmethod
    def from_hdf5(cls, f: h5py.File) -> TCFFile:
        """Create TCFFile from an open HDF5 file handle."""
        tcf = cls()

        # Get timepoints
        if PATH_DATA_3D in f:
            group = _as_group(f[PATH_DATA_3D])
            tcf.timepoints = sorted(group.keys())

        # Get HT shape and resolution
        if tcf.timepoints:
            first_tp = tcf.timepoints[0]
            if f"{PATH_DATA_3D}/{first_tp}" in f:
                ds = _as_dataset(f[f"{PATH_DATA_3D}/{first_tp}"])
                tcf.ht_shape = ds.shape

            # Resolution from attributes
            data_3d = _as_group(f[PATH_DATA_3D])
            tcf.ht_resolution = (
                float(np.asarray(data_3d.attrs.get(ATTR_RESOLUTION_Z, [DEFAULT_HT_RES_Z]))[0]),
                float(np.asarray(data_3d.attrs.get(ATTR_RESOLUTION_Y, [DEFAULT_HT_RES_Y]))[0]),
                float(np.asarray(data_3d.attrs.get(ATTR_RESOLUTION_X, [DEFAULT_HT_RES_X]))[0]),
            )

            # RI range
            ds = _as_dataset(f[f"{PATH_DATA_3D}/{first_tp}"])
            if ATTR_RI_MIN in ds.attrs:
                tcf.ri_min = float(np.asarray(ds.attrs[ATTR_RI_MIN])[0])
            if ATTR_RI_MAX in ds.attrs:
                tcf.ri_max = float(np.asarray(ds.attrs[ATTR_RI_MAX])[0])

        # Check for fluorescence
        if PATH_DATA_3D_FL in f:
            tcf.has_fluorescence = True
            fl_group = _as_group(f[PATH_DATA_3D_FL])
            tcf.fl_channels = sorted(fl_group.keys())

            # FL resolution
            tcf.fl_resolution = (
                float(np.asarray(fl_group.attrs.get(ATTR_RESOLUTION_Z, [DEFAULT_FL_RES_Z]))[0]),
                float(np.asarray(fl_group.attrs.get(ATTR_RESOLUTION_Y, [DEFAULT_FL_RES_Y]))[0]),
                float(np.asarray(fl_group.attrs.get(ATTR_RESOLUTION_X, [DEFAULT_FL_RES_X]))[0]),
            )

            # FL shapes per channel
            for ch in tcf.fl_channels:
                ch_group = _as_group(fl_group[ch])
                ch_timepoints = sorted(ch_group.keys())
                if ch_timepoints:
                    ch_ds = _as_dataset(ch_group[ch_timepoints[0]])
                    tcf.fl_shapes[ch] = ch_ds.shape

            # Registration params
            tcf.registration = load_registration_params(f)

        # Device info
        if PATH_INFO_DEVICE in f:
            dev = _as_group(f[PATH_INFO_DEVICE])
            if ATTR_MAGNIFICATION in dev.attrs:
                tcf.magnification = float(np.asarray(dev.attrs[ATTR_MAGNIFICATION])[0])
            if ATTR_NA in dev.attrs:
                tcf.numerical_aperture = float(np.asarray(dev.attrs[ATTR_NA])[0])
            if ATTR_RI in dev.attrs:
                tcf.medium_ri = float(np.asarray(dev.attrs[ATTR_RI])[0])

        return tcf


def load_registration_params(f: h5py.File) -> RegistrationParams:
    """
    Load FL-HT registration parameters from TCF file.

    Args:
        f: Open HDF5 file handle

    Returns:
        RegistrationParams with values from file (defaults used for missing attrs)

    Note:
        This function is lenient - missing paths or attributes are logged
        and defaults are used. This allows processing of partially complete files.
    """
    params = RegistrationParams()

    # XY registration params
    if PATH_FL_REGISTRATION in f:
        reg = _as_group(f[PATH_FL_REGISTRATION])
        if ATTR_REG_ROTATION in reg.attrs:
            params.rotation = float(np.asarray(reg.attrs[ATTR_REG_ROTATION])[0])
        if ATTR_REG_SCALE in reg.attrs:
            params.scale = float(np.asarray(reg.attrs[ATTR_REG_SCALE])[0])
        if ATTR_REG_TRANSLATION_X in reg.attrs:
            params.translation_x = float(np.asarray(reg.attrs[ATTR_REG_TRANSLATION_X])[0])
        if ATTR_REG_TRANSLATION_Y in reg.attrs:
            params.translation_y = float(np.asarray(reg.attrs[ATTR_REG_TRANSLATION_Y])[0])
    else:
        logger.debug("FL registration path not found, using default parameters")

    # HT resolution
    if PATH_DATA_3D in f:
        ht = _as_group(f[PATH_DATA_3D])
        if ATTR_RESOLUTION_X in ht.attrs:
            params.ht_res_x = float(np.asarray(ht.attrs[ATTR_RESOLUTION_X])[0])
        if ATTR_RESOLUTION_Y in ht.attrs:
            params.ht_res_y = float(np.asarray(ht.attrs[ATTR_RESOLUTION_Y])[0])
        if ATTR_RESOLUTION_Z in ht.attrs:
            params.ht_res_z = float(np.asarray(ht.attrs[ATTR_RESOLUTION_Z])[0])
    else:
        logger.debug("HT data path not found, using default resolution")

    # FL resolution
    if PATH_DATA_3D_FL in f:
        fl = _as_group(f[PATH_DATA_3D_FL])
        if ATTR_RESOLUTION_X in fl.attrs:
            params.fl_res_x = float(np.asarray(fl.attrs[ATTR_RESOLUTION_X])[0])
        if ATTR_RESOLUTION_Y in fl.attrs:
            params.fl_res_y = float(np.asarray(fl.attrs[ATTR_RESOLUTION_Y])[0])
        if ATTR_RESOLUTION_Z in fl.attrs:
            params.fl_res_z = float(np.asarray(fl.attrs[ATTR_RESOLUTION_Z])[0])
    else:
        logger.debug("FL data path not found, using default resolution")

    # FL Z offset
    if f"{PATH_DATA_3D_FL}/CH0" in f:
        ch = _as_group(f[f"{PATH_DATA_3D_FL}/CH0"])
        if ATTR_OFFSET_Z in ch.attrs:
            params.fl_offset_z = float(np.asarray(ch.attrs[ATTR_OFFSET_Z])[0])

    # Validate resolution values are positive
    if params.ht_res_x <= 0 or params.ht_res_y <= 0 or params.ht_res_z <= 0:
        logger.warning("Invalid HT resolution values, using defaults")
        params.ht_res_x = DEFAULT_HT_RES_X
        params.ht_res_y = DEFAULT_HT_RES_Y
        params.ht_res_z = DEFAULT_HT_RES_Z

    if params.fl_res_x <= 0 or params.fl_res_y <= 0 or params.fl_res_z <= 0:
        logger.warning("Invalid FL resolution values, using defaults")
        params.fl_res_x = DEFAULT_FL_RES_X
        params.fl_res_y = DEFAULT_FL_RES_Y
        params.fl_res_z = DEFAULT_FL_RES_Z

    return params


class TCFFileLoader:
    """
    Handles all file I/O operations for TCF files.

    Separates file loading logic from display logic, following SRP.

    Usage:
        loader = TCFFileLoader(path)
        loader.load()
        data = loader.get_timepoint_data(0)
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
        logger.debug(f"Loading TCF file: {self.tcf_path}")
        print(f"Loading: {self.tcf_path.name}")

        self._file = h5py.File(self.tcf_path, "r")
        self._tcf_info = TCFFile.from_hdf5(self._file)

        if self._tcf_info.registration is not None:
            self._reg_params = self._tcf_info.registration

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

        # Load HT volume
        self._data_3d = np.asarray(self.file[f"Data/3D/{tp}"])

        # Load or compute MIP
        mip_path = f"Data/2DMIP/{tp}"
        if mip_path in self.file:
            self._data_mip = np.asarray(self.file[mip_path])
        else:
            self._data_mip = np.max(self._data_3d, axis=0)

        # Load fluorescence if available
        if self.has_fluorescence:
            self._load_fluorescence(tp)

    def _load_fluorescence(self, timepoint: str) -> None:
        """Load fluorescence data for all channels."""
        self._fl_data.clear()

        for ch in self.fl_channels:
            path = f"Data/3DFL/{ch}/{timepoint}"
            if path in self.file:
                self._fl_data[ch] = np.asarray(self.file[path])

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
