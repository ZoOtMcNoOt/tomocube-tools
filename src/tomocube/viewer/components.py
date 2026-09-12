"""Shared calibrated plane mapping and controls for Matplotlib viewers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, TextArea, VPacker
from matplotlib.lines import Line2D

from tomocube.core.types import RegistrationParams
from tomocube.processing.registration import FluorescenceRegistration


def plane_extent(shape, spacing, axis=0) -> list[float]:
    """Image edges in micrometers, with centers at index * spacing (ZYX)."""
    vertical, horizontal = [i for i in range(3) if i != axis]
    dy, dx = spacing[vertical], spacing[horizontal]
    return [-dx / 2, (shape[horizontal] - 0.5) * dx,
            (shape[vertical] - 0.5) * dy, -dy / 2]


def contrast_limits(data) -> tuple[float, float]:
    """Nondegenerate range for controls, including uniform acquisitions."""
    low, high = float(np.min(data)), float(np.max(data))
    if low == high:
        padding = max(abs(low) * 1e-4, 1e-6)
        return low - padding, high + padding
    return low, high


def configure_position_slider(slider, count, spacing, index):
    """Update range, tick step and value without navigation callbacks."""
    slider.eventson = False
    slider.valmin = 0
    slider.valmax = max(count - 1, 1) * spacing
    slider.valstep = spacing
    slider.ax.set_xlim(0, slider.valmax)
    slider.set_val(index * spacing)
    slider.set_active(count > 1)
    slider.ax.set_visible(count > 1)
    slider.eventson = True


def adjust_slider(slider, direction):
    """Move by one sample for position sliders, or 2% for continuous controls."""
    if not slider.active:
        return
    step = slider.valstep or (slider.valmax - slider.valmin) * 0.02
    slider.set_val(np.clip(slider.val + direction * step, slider.valmin, slider.valmax))


def add_scale_bar(ax, width_um):
    """Anchor a calibrated bar inside the axes, independent of FOV height."""
    target = width_um * 0.2
    magnitude = 10 ** np.floor(np.log10(target))
    length = max(n * magnitude for n in (1, 2, 5) if n * magnitude <= target)
    bar = AuxTransformBox(ax.transData)
    bar.add_artist(Line2D([0, length], [0, 0], color="white", linewidth=3))
    label = TextArea(f"{length:g} μm", textprops={"color": "white", "size": 9})
    content = VPacker(children=[bar, label], align="center", pad=0, sep=4)
    artist = AnchoredOffsetbox(loc="lower left", child=content, pad=0.5,
                               borderpad=0.7, frameon=True)
    artist.patch.set_facecolor("#202020")
    artist.patch.set_edgecolor("none")
    artist.patch.set_alpha(0.8)
    ax.add_artist(artist)
    return artist


@dataclass
class FlSliceResult:
    """FL on the HT plane, its physical extent and spatial overlap flag."""
    data: np.ndarray
    extent: list[float]
    in_range: bool


class FluorescenceMapper:
    """Display planes sampled by the same registration as scientific exports.

    Keeps at most one sampled plane per axis; navigation never builds an entire
    registered volume. Recreate the mapper when acquisition or channel changes.
    """

    def __init__(self, fl_data: np.ndarray, ht_shape: tuple[int, int, int],
                 reg_params: RegistrationParams, channel: str | None = None,
                 z_offset_mode: str = "start"):
        self.registration = FluorescenceRegistration(fl_data, ht_shape, reg_params, channel, z_offset_mode)
        self.ht_shape = ht_shape
        self.spacing = (reg_params.ht_res_z, reg_params.ht_res_y, reg_params.ht_res_x)
        self._cache: dict[int, tuple[tuple, FlSliceResult]] = {}

    def get_slice(self, axis: int, index: int, z_offset_um: float = 0) -> FlSliceResult:
        key = (index, z_offset_um)
        cached = self._cache.get(axis)
        if cached is not None and cached[0] == key:
            return cached[1]
        data, overlap = self.registration.sample_plane(axis, index, z_offset_um)
        result = FlSliceResult(data, plane_extent(self.ht_shape, self.spacing, axis), overlap)
        self._cache[axis] = (key, result)
        return result
