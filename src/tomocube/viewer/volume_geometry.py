"""Calibrated native-volume placement and render-only cropping for napari.

Napari clipping planes use data coordinates. A world-space plane normal maps
back to data with A.T, while its position maps with the inverse affine. Keeping
both operations here prevents crop and manual FL movement from disagreeing.
"""
from __future__ import annotations

import numpy as np


def native_slice_conflicts(viewer, axis: int) -> list[str]:
    """Layers whose inverse transform mixes the sliced axis into visible axes.

    Napari's native slicer ignores that out-of-plane component. Test through
    public coordinate methods so additional layer scale/translation is included.
    """
    direction = np.eye(3)[axis]
    displayed = [other for other in range(3) if other != axis]
    conflicts = []
    for layer in viewer.layers:
        if np.ndim(layer.data) != 3:
            continue
        mapped = np.asarray(layer.world_to_data(direction)) - layer.world_to_data(np.zeros(3))
        if np.any(np.abs(mapped[displayed]) > 1e-8):
            conflicts.append(layer.name)
    return conflicts


def constrain_native_slicing(viewer):
    """Keep UI slice mode on an axis napari can represent faithfully.

    Napari's core callbacks run first, so its unsupported-slicing warning may
    appear during a request. Restore a supported order and show an actionable
    notification before returning control to the GUI. Native 3D rendering
    remains unconstrained.
    """
    def on_slice_change(event):
        if viewer.dims.ndisplay != 2:
            return
        conflicts = native_slice_conflicts(viewer, viewer.dims.order[0])
        if not conflicts:
            return
        safe_axis = next((axis for axis in range(3) if not native_slice_conflicts(viewer, axis)), None)
        if event.type == "order":
            event.blocked = True
        if safe_axis is None:
            viewer.dims.ndisplay = 3
        else:
            viewer.dims.order = (safe_axis, *(axis for axis in range(3) if axis != safe_axis))
        from napari.utils.notifications import show_warning
        show_warning(
            "Native napari slicing cannot represent this plane for " + ", ".join(conflicts)
            + ". Restored a supported view. Use the Tomocube slice viewer or GIF export "
            "for calibrated orthogonal planes."
        )

    viewer.dims.events.order.connect(on_slice_change)
    viewer.dims.events.ndisplay.connect(on_slice_change)


def world_clip_planes(voxel_to_world: np.ndarray, bounds_um: np.ndarray) -> list[dict]:
    """Six inward-facing native-data planes enclosing a ZYX world-space box."""
    linear = voxel_to_world[:3, :3]
    translation = voxel_to_world[:3, 3]
    planes = []
    for axis in range(3):
        for side, sign in ((0, 1), (1, -1)):
            position = np.zeros(3)
            position[axis] = bounds_um[axis, side]
            normal = linear[axis] * sign  # A.T @ (signed world unit normal)
            planes.append({
                "position": tuple(np.linalg.solve(linear, position - translation)),
                "normal": tuple(normal / np.linalg.norm(normal)),
                "enabled": True,
            })
    return planes


class VolumeGeometry:
    """Apply HT crop bounds and world-Z adjustments without touching pixels.

    Layers use a complete voxel-to-world affine with identity scale/translate.
    Crop ranges include both endpoints in HT indices; boundaries lie half a
    sample outside those centers. A full-range crop removes all clipping,
    including for fluorescence that extends beyond the HT field of view.
    """

    def __init__(self, ht_shape, ht_spacing):
        self.ht_shape = tuple(ht_shape)
        self.ht_spacing = np.asarray(ht_spacing, dtype=float)
        self.ranges = tuple((0, size - 1) for size in self.ht_shape)
        self.fl_z_offset_um = 0.0
        self._layers = []

    def add_layer(self, layer, voxel_to_world, *, fluorescence=False):
        self._layers.append((layer, np.array(voxel_to_world, dtype=float, copy=True), fluorescence))
        self._apply()

    def set_crop(self, ranges):
        candidate = np.asarray(ranges)
        if (candidate.shape != (3, 2) or candidate.dtype.kind not in "iu"
                or np.any(candidate[:, 0] < 0)
                or np.any(candidate[:, 0] > candidate[:, 1])
                or np.any(candidate[:, 1] >= self.ht_shape)):
            raise ValueError("Crop ranges must be inclusive HT index pairs within each axis")
        self.ranges = tuple(tuple(int(value) for value in pair) for pair in candidate)
        self._apply()

    def set_fl_z_offset(self, offset_um):
        if not np.isscalar(offset_um) or not np.isfinite(offset_um):
            raise ValueError("Manual FL Z offset must be finite")
        self.fl_z_offset_um = float(offset_um)
        self._apply()

    def _apply(self):
        cropping = self.ranges != tuple((0, size - 1) for size in self.ht_shape)
        bounds = (np.asarray(self.ranges) + (-0.5, 0.5)) * self.ht_spacing[:, None]
        for layer, base_affine, fluorescence in self._layers:
            affine = base_affine.copy()
            if fluorescence:
                affine[0, 3] += self.fl_z_offset_um
                layer.metadata["display_z_offset_um"] = self.fl_z_offset_um
            layer.affine = affine
            layer.experimental_clipping_planes = world_clip_planes(affine, bounds) if cropping else []
