"""
TCF Viewer Components - Helper classes for viewers.

This module provides:
    - FluorescenceMapper: FL-to-HT coordinate mapping for visualization
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from tomocube.core.types import RegistrationParams


@dataclass
class FlSliceResult:
    """Result of getting an FL slice with physical coordinate info."""
    data: np.ndarray | None  # The FL slice data (native resolution)
    extent: list[float]  # Physical extent [xmin, xmax, ymax, ymin] for imshow
    in_range: bool  # Whether the slice is within FL Z range


class FluorescenceMapper:
    """
    Handles mapping fluorescence data to holotomography coordinates.

    Now supports two modes:
    - Native resolution: Returns FL at original resolution with physical extent
    - Resampled: Returns FL resampled to HT pixel dimensions (legacy)

    The native resolution approach is preferred as it preserves the true
    physical relationship between HT and FL data.
    """

    def __init__(
        self,
        reg_params: RegistrationParams,
        z_offset_mode: str = "start",
        fl_shape: tuple[int, int, int] | None = None,
        ht_shape: tuple[int, int, int] | None = None,
    ) -> None:
        """
        Initialize the mapper.

        Args:
            reg_params: Registration parameters from TCF file
            z_offset_mode: How to interpret fl_offset_z:
                - "start": OffsetZ is HT Z position where FL slice 0 starts (default)
                - "center": OffsetZ is HT Z position of FL volume center
                - "auto": Center FL on HT volume, ignoring OffsetZ
            fl_shape: FL volume shape (Z, Y, X) - required for center/auto modes
            ht_shape: HT volume shape (Z, Y, X) - required for auto mode
        """
        self.params = reg_params
        self.z_offset_mode = z_offset_mode
        self.fl_shape = fl_shape
        self.ht_shape = ht_shape
        
        # Compute effective Z offset based on mode
        self.effective_offset_z = self._compute_effective_offset(
            z_offset_mode, fl_shape, ht_shape
        )
        
        # Precompute physical dimensions if shapes are available
        if fl_shape is not None:
            self.fl_fov_z = fl_shape[0] * reg_params.fl_res_z
            self.fl_fov_y = fl_shape[1] * reg_params.fl_res_y
            self.fl_fov_x = fl_shape[2] * reg_params.fl_res_x
        
        if ht_shape is not None:
            self.ht_fov_z = ht_shape[0] * reg_params.ht_res_z
            self.ht_fov_y = ht_shape[1] * reg_params.ht_res_y
            self.ht_fov_x = ht_shape[2] * reg_params.ht_res_x
            
            # FL XY offset (center FL in HT FOV)
            if fl_shape is not None:
                self.fl_x_offset = (self.ht_fov_x - self.fl_fov_x) / 2
                self.fl_y_offset = (self.ht_fov_y - self.fl_fov_y) / 2

    def _compute_effective_offset(
        self,
        mode: str,
        fl_shape: tuple[int, int, int] | None,
        ht_shape: tuple[int, int, int] | None,
    ) -> float:
        """Compute effective Z offset based on mode."""
        file_offset = self.params.fl_offset_z
        
        if mode == "start":
            return file_offset
        
        elif mode == "center":
            if fl_shape is None:
                return file_offset
            fl_z = fl_shape[0]
            fl_center_z_um = (fl_z * self.params.fl_res_z) / 2
            # OffsetZ is center, so FL starts at offset - half FL height
            return file_offset - fl_center_z_um
        
        elif mode == "auto":
            if fl_shape is None or ht_shape is None:
                return file_offset
            fl_z = fl_shape[0]
            ht_z = ht_shape[0]
            fl_total_z_um = fl_z * self.params.fl_res_z
            ht_total_z_um = ht_z * self.params.ht_res_z
            # Center FL in HT volume
            return (ht_total_z_um - fl_total_z_um) / 2
        
        return file_offset

    def get_ht_xy_extent(self) -> list[float]:
        """Get HT extent for XY view [xmin, xmax, ymax, ymin]."""
        return [0, self.ht_fov_x, self.ht_fov_y, 0]
    
    def get_fl_xy_extent(self) -> list[float]:
        """Get FL extent for XY view [xmin, xmax, ymax, ymin]."""
        return [
            self.fl_x_offset,
            self.fl_x_offset + self.fl_fov_x,
            self.fl_y_offset + self.fl_fov_y,
            self.fl_y_offset
        ]

    def get_xy_slice_native(
        self,
        fl_3d: np.ndarray,
        ht_z: int,
    ) -> FlSliceResult:
        """
        Get FL slice for XY view at native resolution.

        Args:
            fl_3d: 3D fluorescence volume
            ht_z: HT Z-slice index

        Returns:
            FlSliceResult with native resolution data and physical extent
        """
        # Convert HT Z to FL Z using effective offset
        ht_z_um = ht_z * self.params.ht_res_z
        fl_z_um = ht_z_um - self.effective_offset_z
        fl_z_idx = fl_z_um / self.params.fl_res_z

        extent = self.get_fl_xy_extent()

        if fl_z_idx < 0 or fl_z_idx >= fl_3d.shape[0]:
            # Return empty array with correct shape for out-of-range
            empty = np.zeros((fl_3d.shape[1], fl_3d.shape[2]), dtype=np.float32)
            return FlSliceResult(data=empty, extent=extent, in_range=False)

        fl_z_idx = int(round(np.clip(fl_z_idx, 0, fl_3d.shape[0] - 1)))
        fl_slice = fl_3d[fl_z_idx].astype(np.float32)
        
        return FlSliceResult(data=fl_slice, extent=extent, in_range=True)

    def get_xy_slice(
        self,
        fl_3d: np.ndarray,
        ht_z: int,
        ht_shape: tuple[int, int, int],
    ) -> np.ndarray | None:
        """
        Get FL slice for XY view at a given HT Z position.
        
        DEPRECATED: Use get_xy_slice_native() for physical coordinates.

        Args:
            fl_3d: 3D fluorescence volume
            ht_z: HT Z-slice index
            ht_shape: HT volume shape (Z, Y, X)

        Returns:
            2D array matching HT XY dimensions, or None if out of FL range
        """
        # Convert HT Z to FL Z using effective offset
        ht_z_um = ht_z * self.params.ht_res_z
        fl_z_um = ht_z_um - self.effective_offset_z
        fl_z_idx = fl_z_um / self.params.fl_res_z

        if fl_z_idx < 0 or fl_z_idx >= fl_3d.shape[0]:
            return None

        fl_z_idx = int(round(np.clip(fl_z_idx, 0, fl_3d.shape[0] - 1)))
        fl_slice = fl_3d[fl_z_idx].astype(float)

        # Resize to HT dimensions
        ht_h, ht_w = ht_shape[1], ht_shape[2]
        zoom_factors = (ht_h / fl_slice.shape[0], ht_w / fl_slice.shape[1])
        return np.asarray(ndimage.zoom(fl_slice, zoom_factors, order=1))

    def get_ht_xz_extent(self) -> list[float]:
        """Get HT extent for XZ view [xmin, xmax, zmax, zmin]."""
        return [0, self.ht_fov_x, self.ht_fov_z, 0]
    
    def get_fl_xz_extent(self) -> list[float]:
        """Get FL extent for XZ view [xmin, xmax, zmax, zmin]."""
        return [
            self.fl_x_offset,
            self.fl_x_offset + self.fl_fov_x,
            self.effective_offset_z + self.fl_fov_z,
            self.effective_offset_z
        ]
    
    def get_ht_yz_extent(self) -> list[float]:
        """Get HT extent for YZ view [ymin, ymax, zmax, zmin]."""
        return [0, self.ht_fov_y, self.ht_fov_z, 0]
    
    def get_fl_yz_extent(self) -> list[float]:
        """Get FL extent for YZ view [ymin, ymax, zmax, zmin]."""
        return [
            self.fl_y_offset,
            self.fl_y_offset + self.fl_fov_y,
            self.effective_offset_z + self.fl_fov_z,
            self.effective_offset_z
        ]

    def get_xz_slice_native(
        self,
        fl_3d: np.ndarray,
        ht_y_um: float,
    ) -> FlSliceResult:
        """
        Get FL slice for XZ view at native resolution.

        Args:
            fl_3d: 3D fluorescence volume
            ht_y_um: Y position in physical units (µm)

        Returns:
            FlSliceResult with native resolution data and physical extent
        """
        # Convert physical Y to FL Y index
        fl_y_um = ht_y_um - self.fl_y_offset
        fl_y_idx = int(round(fl_y_um / self.params.fl_res_y))
        
        extent = self.get_fl_xz_extent()
        
        if fl_y_idx < 0 or fl_y_idx >= fl_3d.shape[1]:
            empty = np.zeros((fl_3d.shape[0], fl_3d.shape[2]), dtype=np.float32)
            return FlSliceResult(data=empty, extent=extent, in_range=False)
        
        fl_y_idx = np.clip(fl_y_idx, 0, fl_3d.shape[1] - 1)
        fl_slice = fl_3d[:, fl_y_idx, :].astype(np.float32)
        
        return FlSliceResult(data=fl_slice, extent=extent, in_range=True)

    def get_yz_slice_native(
        self,
        fl_3d: np.ndarray,
        ht_x_um: float,
    ) -> FlSliceResult:
        """
        Get FL slice for YZ view at native resolution.

        Args:
            fl_3d: 3D fluorescence volume
            ht_x_um: X position in physical units (µm)

        Returns:
            FlSliceResult with native resolution data and physical extent
        """
        # Convert physical X to FL X index
        fl_x_um = ht_x_um - self.fl_x_offset
        fl_x_idx = int(round(fl_x_um / self.params.fl_res_x))
        
        extent = self.get_fl_yz_extent()
        
        if fl_x_idx < 0 or fl_x_idx >= fl_3d.shape[2]:
            empty = np.zeros((fl_3d.shape[0], fl_3d.shape[1]), dtype=np.float32)
            return FlSliceResult(data=empty, extent=extent, in_range=False)
        
        fl_x_idx = np.clip(fl_x_idx, 0, fl_3d.shape[2] - 1)
        fl_slice = fl_3d[:, :, fl_x_idx].astype(np.float32)
        
        return FlSliceResult(data=fl_slice, extent=extent, in_range=True)

    def get_xz_slice(
        self,
        fl_3d: np.ndarray,
        ht_y: int,
        ht_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """
        Get FL slice for XZ view at a given HT Y position.

        Args:
            fl_3d: 3D fluorescence volume
            ht_y: HT Y index
            ht_shape: HT volume shape (Z, Y, X)

        Returns:
            2D array (Z, X) mapped to HT coordinates
        """
        fl_shape = fl_3d.shape

        # Map HT Y to FL Y
        fl_y = int(ht_y * fl_shape[1] / ht_shape[1])
        fl_y = np.clip(fl_y, 0, fl_shape[1] - 1)

        fl_xz = fl_3d[:, fl_y, :].astype(float)
        fl_xz = np.asarray(ndimage.zoom(fl_xz, (1, ht_shape[2] / fl_shape[2]), order=1))

        return self._map_fl_z_to_ht(fl_xz, fl_shape[0], ht_shape[0])

    def get_yz_slice(
        self,
        fl_3d: np.ndarray,
        ht_x: int,
        ht_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """
        Get FL slice for YZ view at a given HT X position.

        Args:
            fl_3d: 3D fluorescence volume
            ht_x: HT X index
            ht_shape: HT volume shape (Z, Y, X)

        Returns:
            2D array (Z, Y) mapped to HT coordinates
        """
        fl_shape = fl_3d.shape

        # Map HT X to FL X
        fl_x = int(ht_x * fl_shape[2] / ht_shape[2])
        fl_x = np.clip(fl_x, 0, fl_shape[2] - 1)

        fl_yz = fl_3d[:, :, fl_x].astype(float)
        fl_yz = np.asarray(ndimage.zoom(fl_yz, (1, ht_shape[1] / fl_shape[1]), order=1))

        return self._map_fl_z_to_ht(fl_yz, fl_shape[0], ht_shape[0])

    def _map_fl_z_to_ht(
        self,
        fl_slice: np.ndarray,
        fl_z_size: int,
        ht_z_size: int,
    ) -> np.ndarray:
        """Map FL Z indices to HT Z coordinates."""
        output = np.zeros((ht_z_size, fl_slice.shape[1]))

        for ht_z in range(ht_z_size):
            ht_z_um = ht_z * self.params.ht_res_z
            fl_z_um = ht_z_um - self.effective_offset_z
            fl_z_idx = int(round(fl_z_um / self.params.fl_res_z))

            if 0 <= fl_z_idx < fl_z_size:
                output[ht_z, :] = fl_slice[fl_z_idx, :]

        return output
