"""Sample fluorescence in calibrated HT coordinates without changing source data.

Voxel centers are at index * spacing. XY volumes align at their geometric
centers, rotate in physical YX coordinates, then translate in HT micrometers.
Exports and all viewers share this mapping.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from tomocube.core.config import vprint
from tomocube.core.types import RegistrationParams

Z_OFFSET_MODES = ("start", "center", "auto")


def _prepare_registration(fl_data, ht_shape, params, channel, z_offset_mode):
    data = np.asarray(fl_data)
    if data.ndim not in (2, 3) or data.size == 0:
        raise ValueError("fl_data must be a non-empty 2D or 3D array")
    if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        raise ValueError("fl_data must contain real numeric intensities")
    if len(ht_shape) != data.ndim or any(
        not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0
        for n in ht_shape
    ):
        raise ValueError("ht_shape must have matching dimensionality and positive integer dimensions")
    if z_offset_mode not in Z_OFFSET_MODES:
        raise ValueError(f"z_offset_mode must be one of {Z_OFFSET_MODES}")
    float32_limit = np.finfo(np.float32).max
    if not np.isfinite(data).all() or data.max() > float32_limit or data.min() < -float32_limit:
        raise ValueError("fl_data must contain finite intensities representable as float32")
    # scipy samples native integer/float arrays directly and returns float32.
    # Half precision is the exception: ndimage does not accept float16 input.
    if data.dtype.kind == "f" and data.dtype.itemsize == 2:
        data = data.astype(np.float32)
    elif data.dtype.itemsize > 8:
        raise ValueError("fl_data requires a standard integer, float32 or float64 dtype")

    ht_spacing = np.array([params.ht_res_z, params.ht_res_y, params.ht_res_x])
    fl_spacing = np.array([params.fl_res_z, params.fl_res_y, params.fl_res_x])
    if not (np.isfinite(ht_spacing).all() and np.isfinite(fl_spacing).all()
            and (ht_spacing > 0).all() and (fl_spacing > 0).all()):
        raise ValueError("HT and FL resolutions must be finite and positive")
    file_offset = params.get_offset_z(channel)
    if not np.isfinite([params.rotation, params.translation_y, params.translation_x, file_offset]).all():
        raise ValueError("Registration rotation, translation and Z offset must be finite")

    # Forward rotation in YX order: [[cos, -sin], [sin, cos]]. Translation
    # follows rotation, so the inverse rotates HT coordinates AND translation.
    cos_r, sin_r = np.cos(params.rotation), np.sin(params.rotation)
    inverse_rotation = np.array([[cos_r, sin_r], [-sin_r, cos_r]])
    physical_to_fl = np.diag(1 / fl_spacing[1:]) @ inverse_rotation
    matrix_xy = physical_to_fl @ np.diag(ht_spacing[1:])
    fl_center = (np.array(data.shape[-2:]) - 1) / 2
    ht_center = (np.array(ht_shape[-2:]) - 1) / 2
    translation = np.array([params.translation_y, params.translation_x])
    offset_xy = fl_center - matrix_xy @ ht_center - physical_to_fl @ translation
    if data.ndim == 2:
        return data, matrix_xy, offset_xy

    fl_center_z = (data.shape[0] - 1) * fl_spacing[0] / 2
    if z_offset_mode == "start":
        offset_z = file_offset
    elif z_offset_mode == "center":
        offset_z = file_offset - fl_center_z
    else:
        profile = np.maximum(data.sum(axis=(1, 2), dtype=np.float64), 0)
        signal = profile.sum()
        signal_center = (np.arange(data.shape[0]) @ profile / signal * fl_spacing[0]
                         if signal > 0 else fl_center_z)
        offset_z = (ht_shape[0] - 1) * ht_spacing[0] / 2 - signal_center

    matrix = np.zeros((3, 3))
    matrix[0, 0] = ht_spacing[0] / fl_spacing[0]
    matrix[1:, 1:] = matrix_xy
    offset = np.r_[-offset_z / fl_spacing[0], offset_xy]
    vprint(f"[registration] {channel or 'FL'} {data.shape} → HT {ht_shape}; "
           f"Z mode={z_offset_mode}, first FL center at {offset_z:.4g} μm")
    return data, matrix, offset


def _sample_coordinates(data, coordinates):
    """Linear sampling including endpoints; zero outside the sampled volume."""
    in_bounds = np.ones(coordinates.shape[1:], dtype=bool)
    for axis, size in enumerate(data.shape):
        coord = coordinates[axis]
        # Trigonometric roundoff must not erase boundary voxels at 90/180 degrees.
        coord[np.abs(coord) < 1e-9] = 0
        coord[np.abs(coord - (size - 1)) < 1e-9] = size - 1
        in_bounds &= (coord >= 0) & (coord <= size - 1)
    result = ndimage.map_coordinates(
        data, coordinates, order=1, mode="constant", cval=0, output=np.float32,
        prefilter=False,
    )
    return result, bool(in_bounds.any())


class FluorescenceRegistration:
    """A prepared 3D registration that samples only the requested HT plane.

    Source data and parameters must remain unchanged while this object is used.
    Construct a new registration after selecting another acquisition or channel.
    """

    def __init__(self, fl_data: np.ndarray, ht_shape: tuple[int, int, int],
                 params: RegistrationParams | None = None, channel: str | None = None,
                 z_offset_mode: str = "start", *, translation_um=(0, 0, 0)):
        params = params or RegistrationParams()
        self.data, self.matrix, self.offset = _prepare_registration(
            fl_data, ht_shape, params, channel, z_offset_mode,
        )
        if self.data.ndim != 3:
            raise ValueError("Plane sampling requires a 3D fluorescence volume")
        self.ht_shape = tuple(ht_shape)
        self.fl_res_z = params.fl_res_z
        self.ht_spacing = np.array([params.ht_res_z, params.ht_res_y, params.ht_res_x])
        translation = np.asarray(translation_um, dtype=float)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("translation_um must contain three finite ZYX micrometer values")
        self.offset = self.offset - self.matrix @ (translation / self.ht_spacing)

    @property
    def voxel_to_world(self) -> np.ndarray:
        """Native FL ZYX indices to HT-world ZYX micrometers (4x4 affine).

        This is the forward form of the sampler's inverse mapping. A viewer
        can place original FL voxels without allocating a resampled volume.
        """
        linear = np.diag(self.ht_spacing) @ np.linalg.inv(self.matrix)
        affine = np.eye(4)
        affine[:3, :3] = linear
        affine[:3, 3] = -linear @ self.offset
        return affine

    def sample_plane(self, axis: int, index: int, z_offset_um: float = 0) -> tuple[np.ndarray, bool]:
        """Return a float32 HT plane (axis 0/1/2 = XY/XZ/YZ) and overlap flag.

        A positive manual Z offset moves FL toward increasing HT Z. This display
        adjustment does not change the file's calibration or stored intensities.
        """
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.ht_shape[axis]:
            raise ValueError("Plane index is outside the HT volume")
        if not np.isfinite(z_offset_um):
            raise ValueError("Manual Z offset must be finite")
        varying_axes = [i for i in range(3) if i != axis]
        grid = np.indices([self.ht_shape[i] for i in varying_axes], dtype=np.float64)
        coordinates = np.einsum("ij,jhw->ihw", self.matrix[:, varying_axes], grid)
        coordinates += (self.offset + self.matrix[:, axis] * index)[:, None, None]
        coordinates[0] -= z_offset_um / self.fl_res_z
        return _sample_coordinates(self.data, coordinates)


def register_fl_to_ht(
    fl_data: np.ndarray,
    ht_shape: tuple[int, ...],
    params: RegistrationParams | None = None,
    channel: str | None = None,
    z_offset_mode: str = "start",
    *,
    translation_um=(0, 0, 0),
) -> np.ndarray:
    """Register 2D/3D FL to the matching HT shape using linear sampling.

    Independent XYZ spacings determine scale; metadata Scale is not applied a
    second time. XY translation follows rotation in physical units. The first
    voxel center is at zero; geometric centers use (size - 1) / 2.
    start places the first FL center at channel OffsetZ; center places its
    geometric center there; auto aligns the nonnegative plane-sum weighted FL
    Z center to the HT geometric center (geometric fallback for zero signal).
    Outside FL voxel centers output is zero. Invalid inputs raise ValueError.
    """
    params = params or RegistrationParams()
    if np.ndim(fl_data) == 2:
        if np.any(np.asarray(translation_um) != 0):
            raise ValueError("Residual translation requires a 3D fluorescence volume")
        data, matrix, offset = _prepare_registration(fl_data, ht_shape, params, channel, z_offset_mode)
        grid = np.indices(ht_shape, dtype=np.float64)
        coordinates = np.einsum("ij,jhw->ihw", matrix, grid) + offset[:, None, None]
        return _sample_coordinates(data, coordinates)[0]
    registration = FluorescenceRegistration(
        fl_data, ht_shape, params, channel, z_offset_mode, translation_um=translation_um,
    )
    output = np.empty(ht_shape, dtype=np.float32)
    for z in range(ht_shape[0]):
        output[z] = registration.sample_plane(0, z)[0]
    return output
