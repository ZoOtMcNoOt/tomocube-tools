"""
TCF Viewer Components - Helper classes for viewers.

This module provides:
    - FluorescenceMapper: FL-to-HT coordinate mapping for visualization
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from tomocube.core.types import RegistrationParams


class FluorescenceMapper:
    """
    Handles mapping fluorescence data to holotomography coordinates.

    Separates the complex FL-to-HT coordinate transformation logic
    from the display code.
    """

    def __init__(self, reg_params: RegistrationParams) -> None:
        """
        Initialize the mapper.

        Args:
            reg_params: Registration parameters from TCF file
        """
        self.params = reg_params

    def get_xy_slice(
        self,
        fl_3d: np.ndarray,
        ht_z: int,
        ht_shape: tuple[int, int, int],
    ) -> np.ndarray | None:
        """
        Get FL slice for XY view at a given HT Z position.

        Args:
            fl_3d: 3D fluorescence volume
            ht_z: HT Z-slice index
            ht_shape: HT volume shape (Z, Y, X)

        Returns:
            2D array matching HT XY dimensions, or None if out of FL range
        """
        # Convert HT Z to FL Z
        ht_z_um = ht_z * self.params.ht_res_z
        fl_z_um = ht_z_um - self.params.fl_offset_z
        fl_z_idx = fl_z_um / self.params.fl_res_z

        if fl_z_idx < 0 or fl_z_idx >= fl_3d.shape[0]:
            return None

        fl_z_idx = int(round(np.clip(fl_z_idx, 0, fl_3d.shape[0] - 1)))
        fl_slice = fl_3d[fl_z_idx].astype(float)

        # Resize to HT dimensions
        ht_h, ht_w = ht_shape[1], ht_shape[2]
        zoom_factors = (ht_h / fl_slice.shape[0], ht_w / fl_slice.shape[1])
        return np.asarray(ndimage.zoom(fl_slice, zoom_factors, order=1))

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
            fl_z_um = ht_z_um - self.params.fl_offset_z
            fl_z_idx = int(round(fl_z_um / self.params.fl_res_z))

            if 0 <= fl_z_idx < fl_z_size:
                output[ht_z, :] = fl_slice[fl_z_idx, :]

        return output
