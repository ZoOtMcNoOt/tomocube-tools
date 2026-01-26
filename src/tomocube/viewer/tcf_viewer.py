"""
TCF File Viewer - Interactive 3D Holotomography Visualizer.

Features:
    - Navigate Z-slices with slider, keyboard, or scroll wheel
    - XY, XZ, YZ orthogonal views with crosshairs
    - Physical scale bars and axis labels in micrometers
    - Colorbar showing refractive index values
    - Fluorescence overlay with separate intensity colorbar
    - Adjustable contrast with auto and percentile options
    - Multiple colormaps with invert option
    - Export slices or MIP as PNG

Keyboard shortcuts:
    Up/Down or scroll    Navigate Z-slices
    Home/End             Jump to first/last slice
    A                    Auto-contrast (current slice)
    G                    Auto-contrast (global)
    R                    Reset view
    S                    Save current slice as PNG
    M                    Save MIP as PNG
    I                    Invert colormap
    F                    Toggle fluorescence overlay
    D                    Distance measurement mode
    P                    Polygon/area measurement mode
    C                    Clear all measurements
    1-6                  Switch colormap
    Escape/Q             Quit (or cancel measurement)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.widgets import Button, RadioButtons, RangeSlider, Slider

from tomocube.core.file import TCFFileLoader
from tomocube.core.types import ViewerState
from tomocube.processing.image import normalize_with_bounds
from tomocube.viewer.components import FluorescenceMapper
from tomocube.viewer.measurements import MeasurementTool

if TYPE_CHECKING:
    from matplotlib.backend_bases import Event

logger = logging.getLogger(__name__)


class TCFViewer:
    """
    Interactive viewer for Tomocube TCF files.

    Displays holotomography data with proper scientific visualization:
    - Physical units (micrometers) on axes
    - Scale bars showing physical dimensions
    - Colorbars with refractive index values
    - Fluorescence overlay with intensity scale

    Optimized for responsiveness using set_data() updates.
    """

    COLORMAPS = ["gray", "viridis", "inferno", "turbo", "coolwarm", "bone"]
    DARK_BG = "#1e1e1e"
    DARK_FG = "#2d2d2d"

    def __init__(self, tcf_path: str):
        self.tcf_path = Path(tcf_path)

        # Component classes
        self._loader: TCFFileLoader | None = None
        self.s: ViewerState = ViewerState()
        self._fl_mapper: FluorescenceMapper | None = None
        self._fig: Figure | None = None

        # Image references for fast updates
        self._im_xy: AxesImage | None = None
        self._im_xz: AxesImage | None = None
        self._im_yz: AxesImage | None = None
        self._im_fl_xy: AxesImage | None = None
        self._im_fl_xz: AxesImage | None = None
        self._im_fl_yz: AxesImage | None = None

        # Crosshair references
        self._crosshairs: dict = {}

        # Title references for updates
        self._title_xy = None
        self._title_xz = None
        self._title_yz = None

        # Measurement tool
        self._measurement_tool: MeasurementTool | None = None

        try:
            self._load_file()
            self._setup_figure()
            self._connect_events()
        except Exception:
            self.close()
            raise

    @property
    def loader(self) -> TCFFileLoader:
        assert self._loader is not None, "Loader not initialized"
        return self._loader

    @property
    def fig(self) -> Figure:
        assert self._fig is not None, "Figure not created"
        return self._fig

    @property
    def res_xy(self) -> float:
        """XY resolution in um/pixel."""
        return self.loader.reg_params.ht_res_x

    @property
    def res_z(self) -> float:
        """Z resolution in um/slice."""
        return self.loader.reg_params.ht_res_z

    # =========================================================================
    # File Loading
    # =========================================================================

    def _load_file(self) -> None:
        """Load TCF file using TCFFileLoader component."""
        self._loader = TCFFileLoader(self.tcf_path)
        self._loader.load()

        if self._loader.has_fluorescence:
            self._fl_mapper = FluorescenceMapper(self._loader.reg_params)
            self.s.current_fl_channel = self._loader.fl_channels[0]

        self._load_timepoint(0)

    def _load_timepoint(self, idx: int) -> None:
        """Load data for a specific timepoint."""
        self.loader.load_timepoint(idx)
        self.s.current_timepoint = idx

        shape = self.loader.data_3d.shape
        self.s.current_z = shape[0] // 2
        self.s.current_y = shape[1] // 2
        self.s.current_x = shape[2] // 2

        ch = self.s.current_fl_channel
        if ch and ch in self.loader.fl_data:
            self.s.fl_vmin, self.s.fl_vmax = self.loader.get_fl_contrast(ch)

        self._auto_contrast_global()

    # =========================================================================
    # Display Helpers
    # =========================================================================

    def _auto_contrast_global(self) -> None:
        """Set contrast from global percentiles."""
        p1, p99 = np.percentile(self.loader.data_3d, [1, 99])
        self.s.vmin, self.s.vmax = float(p1), float(p99)

    def _auto_contrast_slice(self) -> None:
        """Set contrast from current slice percentiles."""
        slice_data = self.loader.data_3d[self.s.current_z]
        p1, p99 = np.percentile(slice_data, [1, 99])
        self.s.vmin, self.s.vmax = float(p1), float(p99)

    def _get_cmap(self) -> str:
        """Get current colormap name with optional inversion."""
        return self.s.get_cmap()

    def _format_ri(self, val: float) -> str:
        """Format raw value as refractive index."""
        return f"{val:.4f}"

    def _get_extent_xy(self) -> list[float]:
        """Get extent for XY view in micrometers."""
        shape = self.loader.data_3d.shape
        return [0, shape[2] * self.res_xy, shape[1] * self.res_xy, 0]

    def _get_extent_xz(self) -> list[float]:
        """Get extent for XZ view in micrometers."""
        shape = self.loader.data_3d.shape
        return [0, shape[2] * self.res_xy, shape[0] * self.res_z, 0]

    def _get_extent_yz(self) -> list[float]:
        """Get extent for YZ view in micrometers."""
        shape = self.loader.data_3d.shape
        return [0, shape[1] * self.res_xy, shape[0] * self.res_z, 0]

    # =========================================================================
    # Figure Setup
    # =========================================================================

    def _setup_figure(self) -> None:
        """Create the figure and all UI elements."""
        self._fig = plt.figure(figsize=(16, 10), facecolor=self.DARK_BG)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(f"TCF Viewer - {self.tcf_path.name}")

        # Main axes for views - adjusted layout for colorbars
        self.ax_xy = self.fig.add_axes((0.05, 0.28, 0.38, 0.52), facecolor=self.DARK_FG)
        self.ax_xz = self.fig.add_axes((0.50, 0.53, 0.20, 0.27), facecolor=self.DARK_FG)
        self.ax_yz = self.fig.add_axes((0.50, 0.28, 0.20, 0.22), facecolor=self.DARK_FG)
        self.ax_hist = self.fig.add_axes((0.75, 0.28, 0.20, 0.52), facecolor=self.DARK_FG)

        # Colorbar axes
        self.ax_cbar_ht = self.fig.add_axes((0.44, 0.28, 0.015, 0.52), facecolor=self.DARK_FG)
        if self.loader.has_fluorescence:
            self.ax_cbar_fl = self.fig.add_axes((0.71, 0.28, 0.015, 0.52), facecolor=self.DARK_FG)

        for ax in [self.ax_xy, self.ax_xz, self.ax_yz, self.ax_hist]:
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("white")

        self._setup_sliders()
        self._setup_buttons()
        self._setup_info_text()
        self._initial_display()
        self._setup_measurement_tool()

    def _setup_sliders(self) -> None:
        """Create navigation and contrast sliders."""
        slider_color = "#4a9eff"
        data = self.loader.data_3d
        z_max = data.shape[0] - 1
        y_max = data.shape[1] - 1

        # Z slider - show in micrometers
        z_um_max = z_max * self.res_z
        ax_z = self.fig.add_axes((0.05, 0.18, 0.38, 0.025), facecolor=self.DARK_FG)
        self.z_slider = Slider(ax_z, "Z (um)", 0, z_um_max,
                               valinit=self.s.current_z * self.res_z,
                               color=slider_color)
        self.z_slider.label.set_color("white")
        self.z_slider.valtext.set_color("white")
        self.z_slider.on_changed(self._on_z_change)

        # Y slider - show in micrometers
        y_um_max = y_max * self.res_xy
        ax_y = self.fig.add_axes((0.05, 0.14, 0.38, 0.025), facecolor=self.DARK_FG)
        self.y_slider = Slider(ax_y, "Y (um)", 0, y_um_max,
                               valinit=self.s.current_y * self.res_xy,
                               color=slider_color)
        self.y_slider.label.set_color("white")
        self.y_slider.valtext.set_color("white")
        self.y_slider.on_changed(self._on_y_change)

        # Contrast slider - show RI values
        data_min, data_max = float(data.min()), float(data.max())
        ax_c = self.fig.add_axes((0.05, 0.10, 0.38, 0.025), facecolor=self.DARK_FG)
        self.contrast_slider = RangeSlider(ax_c, "RI", data_min, data_max,
                                           valinit=(self.s.vmin, self.s.vmax), color=slider_color)
        self.contrast_slider.label.set_color("white")
        self.contrast_slider.valtext.set_color("white")
        self.contrast_slider.on_changed(self._on_contrast_change)

        # FL alpha slider (if applicable)
        if self.loader.tcf_info.has_fluorescence:
            ax_fl = self.fig.add_axes((0.05, 0.06, 0.38, 0.025), facecolor=self.DARK_FG)
            self.fl_alpha_slider = Slider(ax_fl, "FL alpha", 0, 1, valinit=0.5, color="#50c878")
            self.fl_alpha_slider.label.set_color("white")
            self.fl_alpha_slider.valtext.set_color("white")
            self.fl_alpha_slider.on_changed(self._on_fl_alpha_change)

        # Timepoint slider
        if len(self.loader.timepoints) > 1:
            ax_t = self.fig.add_axes((0.50, 0.18, 0.20, 0.025), facecolor=self.DARK_FG)
            self.tp_slider = Slider(ax_t, "T", 0, len(self.loader.timepoints) - 1,
                                    valinit=0, valstep=1, color=slider_color)
            self.tp_slider.label.set_color("white")
            self.tp_slider.valtext.set_color("white")
            self.tp_slider.on_changed(self._on_timepoint_change)

    def _setup_buttons(self) -> None:
        """Create control buttons."""
        btn_w, btn_h = 0.07, 0.03
        btn_y = 0.92

        buttons = [
            (0.05, "Auto", self._on_auto_contrast),
            (0.13, "Global", self._on_global_contrast),
            (0.21, "Reset", self._on_reset),
            (0.29, "Invert", self._on_invert),
            (0.37, "Save", self._on_save_slice),
        ]

        if self.loader.tcf_info.has_fluorescence:
            buttons.append((0.45, "FL +/-", self._on_toggle_fluorescence))

        # Measurement buttons on second row
        btn_y2 = 0.88
        meas_buttons = [
            (0.05, "Distance", self._on_start_distance),
            (0.13, "Area", self._on_start_area),
            (0.21, "Clear", self._on_clear_measurements),
        ]

        self._buttons = []
        for x, label, handler in buttons:
            ax = self.fig.add_axes((x, btn_y, btn_w, btn_h), facecolor=self.DARK_FG)
            btn = Button(ax, label, color=self.DARK_FG, hovercolor="#3d3d3d")
            btn.label.set_color("white")
            btn.label.set_fontsize(9)
            btn.on_clicked(handler)
            self._buttons.append(btn)

        # Measurement buttons
        for x, label, handler in meas_buttons:
            ax = self.fig.add_axes((x, btn_y2, btn_w, btn_h), facecolor=self.DARK_FG)
            btn = Button(ax, label, color=self.DARK_FG, hovercolor="#3d3d3d")
            btn.label.set_color("white")
            btn.label.set_fontsize(9)
            btn.on_clicked(handler)
            self._buttons.append(btn)

        # Colormap selector
        ax_cmap = self.fig.add_axes((0.75, 0.06, 0.12, 0.15), facecolor=self.DARK_FG)
        self.cmap_radio = RadioButtons(ax_cmap, self.COLORMAPS, active=0)
        for label in self.cmap_radio.labels:
            label.set_color("white")
            label.set_fontsize(9)
        self.cmap_radio.on_clicked(self._on_cmap_change)

    def _setup_info_text(self) -> None:
        """Create info text display."""
        self.info_text = self.fig.text(0.5, 0.97, "", ha="center", va="top",
                                       color="white", fontsize=10, family="monospace")
        self.pixel_text = self.fig.text(0.5, 0.02, "", ha="center", va="bottom",
                                        color="#888888", fontsize=9, family="monospace")

    # =========================================================================
    # Initial Display Setup (called once)
    # =========================================================================

    def _initial_display(self) -> None:
        """Set up initial display with all static elements."""
        cmap = self._get_cmap()
        data = self.loader.data_3d
        s = self.s

        # Calculate physical positions
        z_um = s.current_z * self.res_z
        y_um = s.current_y * self.res_xy
        x_um = s.current_x * self.res_xy

        # XY view
        extent_xy = self._get_extent_xy()
        xy_slice = data[s.current_z]
        self._im_xy = self.ax_xy.imshow(xy_slice, cmap=cmap, vmin=s.vmin, vmax=s.vmax,
                                         aspect="equal", extent=extent_xy)
        self._setup_fl_overlay(self.ax_xy, "xy", extent_xy)
        self._setup_crosshairs(self.ax_xy, "xy", x_um, y_um)
        self.ax_xy.set_xlabel("X (um)", color="white", fontsize=9)
        self.ax_xy.set_ylabel("Y (um)", color="white", fontsize=9)
        self._title_xy = self.ax_xy.set_title(f"XY plane at Z = {z_um:.1f} um",
                                               color="white", fontsize=10)
        self._add_scale_bar(self.ax_xy, extent_xy[1])

        # XZ view
        extent_xz = self._get_extent_xz()
        xz_slice = data[:, s.current_y, :]
        self._im_xz = self.ax_xz.imshow(xz_slice, cmap=cmap, vmin=s.vmin, vmax=s.vmax,
                                         aspect="auto", extent=extent_xz)
        self._setup_fl_overlay(self.ax_xz, "xz", extent_xz)
        self._setup_crosshairs(self.ax_xz, "xz", x_um, z_um)
        self.ax_xz.set_xlabel("X (um)", color="white", fontsize=8)
        self.ax_xz.set_ylabel("Z (um)", color="white", fontsize=8)
        self._title_xz = self.ax_xz.set_title(f"XZ at Y = {y_um:.1f} um",
                                               color="white", fontsize=9)

        # YZ view
        extent_yz = self._get_extent_yz()
        yz_slice = data[:, :, s.current_x]
        self._im_yz = self.ax_yz.imshow(yz_slice, cmap=cmap, vmin=s.vmin, vmax=s.vmax,
                                         aspect="auto", extent=extent_yz)
        self._setup_fl_overlay(self.ax_yz, "yz", extent_yz)
        self._setup_crosshairs(self.ax_yz, "yz", y_um, z_um)
        self.ax_yz.set_xlabel("Y (um)", color="white", fontsize=8)
        self.ax_yz.set_ylabel("Z (um)", color="white", fontsize=8)
        self._title_yz = self.ax_yz.set_title(f"YZ at X = {x_um:.1f} um",
                                               color="white", fontsize=9)

        # RI Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=s.vmin, vmax=s.vmax))
        sm.set_array([])
        self._cbar_ht = self.fig.colorbar(sm, cax=self.ax_cbar_ht)
        self._cbar_ht.set_label("Refractive Index", color="white", fontsize=9)
        self._cbar_ht.ax.tick_params(colors="white", labelsize=8)

        # FL colorbar placeholder
        if self.loader.has_fluorescence:
            sm_fl = plt.cm.ScalarMappable(cmap="Greens",
                                          norm=plt.Normalize(vmin=s.fl_vmin, vmax=s.fl_vmax))
            sm_fl.set_array([])
            self._cbar_fl = self.fig.colorbar(sm_fl, cax=self.ax_cbar_fl)
            self._cbar_fl.set_label("FL Intensity", color="white", fontsize=9)
            self._cbar_fl.ax.tick_params(colors="white", labelsize=8)
            # Initially hide FL colorbar
            self.ax_cbar_fl.set_visible(s.show_fluorescence)

        # Initial histogram
        self._update_histogram(xy_slice)

        self._update_info_text()

    def _setup_crosshairs(self, ax, view_id: str, x: float, y: float) -> None:
        """Set up crosshair lines for an axis."""
        hline = ax.axhline(y=y, color="#ff6b6b", lw=0.8, alpha=0.7)
        vline = ax.axvline(x=x, color="#50c878", lw=0.8, alpha=0.7)
        self._crosshairs[view_id] = {"h": hline, "v": vline}

    def _setup_fl_overlay(self, ax, plane: str, extent: list[float]) -> None:
        """Set up FL overlay image (initially transparent)."""
        if not self.loader.has_fluorescence:
            return

        ch = self.s.current_fl_channel
        if ch is None or ch not in self.loader.fl_data:
            return

        fl_3d = self.loader.fl_data[ch]

        if plane == "xy":
            fl_slice = self._get_fl_xy_slice(fl_3d)
            if fl_slice is None:
                return
            fl_rgba = self._create_fl_rgba(fl_slice)
            self._im_fl_xy = ax.imshow(fl_rgba, extent=extent, aspect="equal")
            self._im_fl_xy.set_visible(self.s.show_fluorescence)
        elif plane == "xz":
            fl_slice = self._get_fl_xz_slice(fl_3d)
            fl_rgba = self._create_fl_rgba(fl_slice)
            self._im_fl_xz = ax.imshow(fl_rgba, extent=extent, aspect="auto")
            self._im_fl_xz.set_visible(self.s.show_fluorescence)
        else:  # yz
            fl_slice = self._get_fl_yz_slice(fl_3d)
            fl_rgba = self._create_fl_rgba(fl_slice)
            self._im_fl_yz = ax.imshow(fl_rgba, extent=extent, aspect="auto")
            self._im_fl_yz.set_visible(self.s.show_fluorescence)

    def _create_fl_rgba(self, fl_slice: np.ndarray) -> np.ndarray:
        """Create RGBA array for FL overlay."""
        fl_norm = normalize_with_bounds(fl_slice, self.s.fl_vmin, self.s.fl_vmax)
        fl_rgba = np.zeros((*fl_norm.shape, 4), dtype=np.float32)
        fl_rgba[:, :, 1] = fl_norm  # Green channel
        fl_rgba[:, :, 3] = fl_norm * self.s.fl_overlay_alpha
        return fl_rgba

    def _add_scale_bar(self, ax, fov_um: float) -> None:
        """Add a scale bar to the axis."""
        scale_bar_um = 10
        for bar_len in [10, 20, 50, 100]:
            if bar_len < fov_um * 0.3:
                scale_bar_um = bar_len

        x_start = fov_um * 0.05
        y_pos = fov_um * 0.95

        ax.plot([x_start, x_start + scale_bar_um], [y_pos, y_pos],
                color="white", lw=3, solid_capstyle="butt")
        ax.text(x_start + scale_bar_um / 2, y_pos - fov_um * 0.03,
                f"{scale_bar_um} um", color="white", ha="center", va="top",
                fontsize=9, fontweight="bold")

    # =========================================================================
    # Fast Display Update (called on slice changes)
    # =========================================================================

    def _update_display(self) -> None:
        """Fast update using set_data() - no clearing/recreating."""
        data = self.loader.data_3d
        s = self.s

        # Calculate physical positions
        z_um = s.current_z * self.res_z
        y_um = s.current_y * self.res_xy
        x_um = s.current_x * self.res_xy

        # Update image data (fast)
        xy_slice = data[s.current_z]
        xz_slice = data[:, s.current_y, :]
        yz_slice = data[:, :, s.current_x]

        self._im_xy.set_data(xy_slice)
        self._im_xz.set_data(xz_slice)
        self._im_yz.set_data(yz_slice)

        # Update FL overlays if visible
        if s.show_fluorescence and self.loader.has_fluorescence:
            self._update_fl_overlays()

        # Update crosshairs (fast - just set ydata/xdata)
        self._crosshairs["xy"]["h"].set_ydata([y_um, y_um])
        self._crosshairs["xy"]["v"].set_xdata([x_um, x_um])
        self._crosshairs["xz"]["h"].set_ydata([z_um, z_um])
        self._crosshairs["xz"]["v"].set_xdata([x_um, x_um])
        self._crosshairs["yz"]["h"].set_ydata([z_um, z_um])
        self._crosshairs["yz"]["v"].set_xdata([y_um, y_um])

        # Update titles (fast - just set_text)
        self._title_xy.set_text(f"XY plane at Z = {z_um:.1f} um")
        self._title_xz.set_text(f"XZ at Y = {y_um:.1f} um")
        self._title_yz.set_text(f"YZ at X = {x_um:.1f} um")

        self._update_info_text()

    def _update_fl_overlays(self) -> None:
        """Update FL overlay data."""
        ch = self.s.current_fl_channel
        if ch is None or ch not in self.loader.fl_data:
            return

        fl_3d = self.loader.fl_data[ch]

        if self._im_fl_xy is not None:
            fl_slice = self._get_fl_xy_slice(fl_3d)
            if fl_slice is not None:
                self._im_fl_xy.set_data(self._create_fl_rgba(fl_slice))

        if self._im_fl_xz is not None:
            fl_slice = self._get_fl_xz_slice(fl_3d)
            self._im_fl_xz.set_data(self._create_fl_rgba(fl_slice))

        if self._im_fl_yz is not None:
            fl_slice = self._get_fl_yz_slice(fl_3d)
            self._im_fl_yz.set_data(self._create_fl_rgba(fl_slice))

    def _update_contrast(self) -> None:
        """Update contrast/colormap without redrawing everything."""
        s = self.s
        cmap = self._get_cmap()

        # Update HT images
        self._im_xy.set_clim(s.vmin, s.vmax)
        self._im_xy.set_cmap(cmap)
        self._im_xz.set_clim(s.vmin, s.vmax)
        self._im_xz.set_cmap(cmap)
        self._im_yz.set_clim(s.vmin, s.vmax)
        self._im_yz.set_cmap(cmap)

        # Update colorbar
        self._cbar_ht.mappable.set_clim(s.vmin, s.vmax)
        self._cbar_ht.mappable.set_cmap(cmap)

        # Update histogram
        xy_slice = self.loader.data_3d[s.current_z]
        self._update_histogram(xy_slice)

        self._update_info_text()

    def _update_histogram(self, xy_slice: np.ndarray) -> None:
        """Update histogram display."""
        self.ax_hist.clear()
        self.ax_hist.hist(xy_slice.ravel(), bins=100, color="#4a9eff", alpha=0.7)
        self.ax_hist.axvline(self.s.vmin, color="#ff6b6b", ls="--", lw=1.5)
        self.ax_hist.axvline(self.s.vmax, color="#ff6b6b", ls="--", lw=1.5)
        self.ax_hist.set_xlabel("Refractive Index", color="white", fontsize=9)
        self.ax_hist.set_ylabel("Count", color="white", fontsize=9)
        self.ax_hist.set_title("RI Distribution", color="white", fontsize=10)
        self.ax_hist.tick_params(colors="white", labelsize=8)

    def _update_info_text(self) -> None:
        """Update the info text display with physical units."""
        info = self.loader.tcf_info
        data = self.loader.data_3d

        fov_x = data.shape[2] * self.res_xy
        fov_y = data.shape[1] * self.res_xy
        fov_z = data.shape[0] * self.res_z

        parts = [
            f"{info.magnification or '?'}x  NA {info.numerical_aperture or '?'}",
            f"FOV: {fov_x:.0f} x {fov_y:.0f} x {fov_z:.0f} um",
            f"RI: {self._format_ri(self.s.vmin)} - {self._format_ri(self.s.vmax)}",
        ]

        if self.s.show_fluorescence and self.s.current_fl_channel:
            parts.append(f"FL: {self.s.current_fl_channel} (alpha={self.s.fl_overlay_alpha:.1f})")

        self.info_text.set_text("  |  ".join(parts))

    # =========================================================================
    # Fluorescence Helpers
    # =========================================================================

    def _get_fl_xy_slice(self, fl_3d: np.ndarray) -> np.ndarray | None:
        if self._fl_mapper is None:
            return None
        return self._fl_mapper.get_xy_slice(fl_3d, self.s.current_z, self.loader.data_3d.shape)

    def _get_fl_xz_slice(self, fl_3d: np.ndarray) -> np.ndarray:
        shape = self.loader.data_3d.shape
        if self._fl_mapper is None:
            return np.zeros((shape[0], shape[2]))
        return self._fl_mapper.get_xz_slice(fl_3d, self.s.current_y, shape)

    def _get_fl_yz_slice(self, fl_3d: np.ndarray) -> np.ndarray:
        shape = self.loader.data_3d.shape
        if self._fl_mapper is None:
            return np.zeros((shape[0], shape[1]))
        return self._fl_mapper.get_yz_slice(fl_3d, self.s.current_x, shape)

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _connect_events(self) -> None:
        """Connect matplotlib events."""
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _on_z_change(self, val_um: float) -> None:
        """Handle Z slider change (value is in micrometers)."""
        self.s.current_z = int(round(val_um / self.res_z))
        self.s.current_z = np.clip(self.s.current_z, 0, self.loader.data_3d.shape[0] - 1)
        self._update_display()
        self.fig.canvas.draw_idle()

    def _on_y_change(self, val_um: float) -> None:
        """Handle Y slider change (value is in micrometers)."""
        self.s.current_y = int(round(val_um / self.res_xy))
        self.s.current_y = np.clip(self.s.current_y, 0, self.loader.data_3d.shape[1] - 1)
        self._update_display()
        self.fig.canvas.draw_idle()

    def _on_contrast_change(self, val: tuple[float, float]) -> None:
        if val[0] <= val[1]:
            self.s.vmin, self.s.vmax = val
            self._update_contrast()
            self.fig.canvas.draw_idle()

    def _on_fl_alpha_change(self, val: float) -> None:
        self.s.fl_overlay_alpha = val
        if self.s.show_fluorescence:
            self._update_fl_overlays()
            self.fig.canvas.draw_idle()

    def _on_timepoint_change(self, val: float) -> None:
        self._load_timepoint(int(val))
        self._update_sliders()
        self._update_display()
        self.fig.canvas.draw_idle()

    def _update_sliders(self) -> None:
        """Update slider ranges after loading new data."""
        shape = self.loader.data_3d.shape
        self.z_slider.valmax = (shape[0] - 1) * self.res_z
        self.y_slider.valmax = (shape[1] - 1) * self.res_xy

    def _on_cmap_change(self, label: str | None) -> None:
        if label is None:
            return
        self.s.colormap = label
        self._update_contrast()
        self.fig.canvas.draw_idle()

    def _on_auto_contrast(self, event: Event | None = None) -> None:
        self._auto_contrast_slice()
        self.contrast_slider.set_val((self.s.vmin, self.s.vmax))

    def _on_global_contrast(self, event: Event | None = None) -> None:
        self._auto_contrast_global()
        self.contrast_slider.set_val((self.s.vmin, self.s.vmax))

    def _on_reset(self, event: Event | None = None) -> None:
        shape = self.loader.data_3d.shape
        self.s.current_z = shape[0] // 2
        self.s.current_y = shape[1] // 2
        self.s.current_x = shape[2] // 2
        self.s.invert_cmap = False
        self._auto_contrast_global()
        self.z_slider.set_val(self.s.current_z * self.res_z)
        self.y_slider.set_val(self.s.current_y * self.res_xy)
        self.contrast_slider.set_val((self.s.vmin, self.s.vmax))

    def _on_invert(self, event: Event | None = None) -> None:
        self.s.invert_cmap = not self.s.invert_cmap
        self._update_contrast()
        self.fig.canvas.draw_idle()

    def _on_toggle_fluorescence(self, event: Event | None = None) -> None:
        self.s.show_fluorescence = not self.s.show_fluorescence

        # Toggle visibility of FL overlays
        if self._im_fl_xy is not None:
            self._im_fl_xy.set_visible(self.s.show_fluorescence)
        if self._im_fl_xz is not None:
            self._im_fl_xz.set_visible(self.s.show_fluorescence)
        if self._im_fl_yz is not None:
            self._im_fl_yz.set_visible(self.s.show_fluorescence)

        # Toggle FL colorbar visibility
        if hasattr(self, 'ax_cbar_fl'):
            self.ax_cbar_fl.set_visible(self.s.show_fluorescence)

        if self.s.show_fluorescence:
            self._update_fl_overlays()

        self._update_info_text()
        print(f"Fluorescence: {'ON' if self.s.show_fluorescence else 'OFF'}")
        self.fig.canvas.draw_idle()

    def _on_save_slice(self, event: Event | None = None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        z_um = self.s.current_z * self.res_z
        filename = f"{self.tcf_path.stem}_z{z_um:.0f}um_{timestamp}.png"
        filepath = self.tcf_path.parent / filename
        plt.imsave(filepath, self.loader.data_3d[self.s.current_z], cmap=self._get_cmap(),
                   vmin=self.s.vmin, vmax=self.s.vmax)
        print(f"Saved: {filepath}")

    def _on_key(self, event: Event) -> None:
        key = getattr(event, 'key', None)
        if key is None:
            return
        z_max = self.loader.data_3d.shape[0] - 1

        if key in ("up", "w"):
            new_z = min(self.s.current_z + 1, z_max)
            self.z_slider.set_val(new_z * self.res_z)
        elif key in ("down", "s"):
            new_z = max(self.s.current_z - 1, 0)
            self.z_slider.set_val(new_z * self.res_z)
        elif key == "home":
            self.z_slider.set_val(0)
        elif key == "end":
            self.z_slider.set_val(z_max * self.res_z)
        elif key == "a":
            self._on_auto_contrast()
        elif key == "g":
            self._on_global_contrast()
        elif key == "r":
            self._on_reset()
        elif key == "i":
            self._on_invert()
        elif key == "m":
            self._on_save_mip()
        elif key == "f" and self.loader.tcf_info.has_fluorescence:
            self._on_toggle_fluorescence()
        elif key == "d":
            self._on_start_distance()
        elif key == "p":
            self._on_start_area()
        elif key == "c":
            self._on_clear_measurements()
        elif key in ("q", "escape"):
            # Cancel measurement first if active, otherwise quit
            if self._measurement_tool and self._measurement_tool._mode:
                self._measurement_tool.cancel()
            else:
                plt.close(self.fig)
        elif key in "123456":
            idx = int(key) - 1
            if idx < len(self.COLORMAPS):
                self.s.colormap = self.COLORMAPS[idx]
                self._update_contrast()
                self.fig.canvas.draw_idle()

    def _on_scroll(self, event: Event) -> None:
        button = getattr(event, 'button', None)
        inaxes = getattr(event, 'inaxes', None)
        delta = 1 if button == "up" else -1
        shape = self.loader.data_3d.shape
        z_max = shape[0] - 1
        y_max = shape[1] - 1

        if inaxes == self.ax_xy:
            new_z = np.clip(self.s.current_z + delta, 0, z_max)
            self.z_slider.set_val(new_z * self.res_z)
        elif inaxes == self.ax_xz:
            new_y = np.clip(self.s.current_y + delta, 0, y_max)
            self.y_slider.set_val(new_y * self.res_xy)

    def _on_click(self, event: Event) -> None:
        xdata = getattr(event, 'xdata', None)
        ydata = getattr(event, 'ydata', None)
        inaxes = getattr(event, 'inaxes', None)
        if xdata is None or ydata is None:
            return

        shape = self.loader.data_3d.shape

        if inaxes == self.ax_xy:
            self.s.current_x = int(np.clip(xdata / self.res_xy, 0, shape[2] - 1))
            self.s.current_y = int(np.clip(ydata / self.res_xy, 0, shape[1] - 1))
            self.y_slider.set_val(self.s.current_y * self.res_xy)
        elif inaxes == self.ax_xz:
            self.s.current_x = int(np.clip(xdata / self.res_xy, 0, shape[2] - 1))
            self.s.current_z = int(np.clip(ydata / self.res_z, 0, shape[0] - 1))
            self.z_slider.set_val(self.s.current_z * self.res_z)
        elif inaxes == self.ax_yz:
            self.s.current_y = int(np.clip(xdata / self.res_xy, 0, shape[1] - 1))
            self.s.current_z = int(np.clip(ydata / self.res_z, 0, shape[0] - 1))
            self.z_slider.set_val(self.s.current_z * self.res_z)
            self.y_slider.set_val(self.s.current_y * self.res_xy)

        self._update_display()
        self.fig.canvas.draw_idle()

    def _on_motion(self, event: Event) -> None:
        xdata = getattr(event, 'xdata', None)
        ydata = getattr(event, 'ydata', None)
        inaxes = getattr(event, 'inaxes', None)
        if xdata is None or ydata is None:
            return

        data = self.loader.data_3d
        shape = data.shape

        if inaxes == self.ax_xy:
            x_px = int(xdata / self.res_xy)
            y_px = int(ydata / self.res_xy)
            if 0 <= x_px < shape[2] and 0 <= y_px < shape[1]:
                val = data[self.s.current_z, y_px, x_px]
                z_um = self.s.current_z * self.res_z
                self.pixel_text.set_text(
                    f"Position: ({xdata:.1f}, {ydata:.1f}, {z_um:.1f}) um  |  RI = {self._format_ri(val)}"
                )
                self.fig.canvas.draw_idle()
        elif inaxes == self.ax_xz:
            x_px = int(xdata / self.res_xy)
            z_px = int(ydata / self.res_z)
            if 0 <= x_px < shape[2] and 0 <= z_px < shape[0]:
                val = data[z_px, self.s.current_y, x_px]
                y_um = self.s.current_y * self.res_xy
                self.pixel_text.set_text(
                    f"Position: ({xdata:.1f}, {y_um:.1f}, {ydata:.1f}) um  |  RI = {self._format_ri(val)}"
                )
                self.fig.canvas.draw_idle()

    def _on_save_mip(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.tcf_path.stem}_MIP_{timestamp}.png"
        filepath = self.tcf_path.parent / filename
        plt.imsave(filepath, self.loader.data_mip, cmap=self._get_cmap(),
                   vmin=self.s.vmin, vmax=self.s.vmax)
        print(f"Saved: {filepath}")

    # =========================================================================
    # Measurement Tool
    # =========================================================================

    def _setup_measurement_tool(self) -> None:
        """Initialize measurement tool for XY view."""
        self._measurement_tool = MeasurementTool(
            self.ax_xy, self.fig, status_callback=self._on_measurement_status
        )

    def _on_measurement_status(self, message: str) -> None:
        """Handle status updates from measurement tool."""
        self.pixel_text.set_text(message)
        self.fig.canvas.draw_idle()

    def _on_start_distance(self, event: Event | None = None) -> None:
        """Start distance measurement mode."""
        if self._measurement_tool:
            self._measurement_tool.start_distance()

    def _on_start_area(self, event: Event | None = None) -> None:
        """Start area/polygon measurement mode."""
        if self._measurement_tool:
            self._measurement_tool.start_area()

    def _on_clear_measurements(self, event: Event | None = None) -> None:
        """Clear all measurements."""
        if self._measurement_tool:
            self._measurement_tool.clear_all()

    # =========================================================================
    # Public API
    # =========================================================================

    def show(self) -> None:
        """Display the viewer."""
        plt.show()

    def close(self) -> None:
        """Close file and cleanup resources."""
        if self._loader is not None:
            try:
                self._loader.close()
            except Exception:
                logger.debug("Error closing HDF5 file during cleanup", exc_info=True)
            finally:
                self._loader = None

        if self._fig is not None:
            try:
                plt.close(self._fig)
            except Exception:
                logger.debug("Error closing matplotlib figure during cleanup", exc_info=True)
            finally:
                self._fig = None

    def __enter__(self) -> TCFViewer:
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        self.close()


# =============================================================================
# CLI Support
# =============================================================================


def find_tcf_files(directory: str) -> list[str]:
    """Find all TCF files in a directory."""
    tcf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.upper().endswith(".TCF"):
                tcf_files.append(os.path.join(root, file))
    return sorted(tcf_files)


def select_file_dialog() -> str | None:
    """Open file dialog to select TCF file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select TCF File",
            filetypes=[("TCF Files", "*.TCF"), ("All Files", "*.*")]
        )
        root.destroy()
        return file_path or None
    except ImportError:
        print("tkinter not available. Please provide file path as argument.")
        return None


def main() -> None:
    """Main entry point."""
    tcf_path = None

    if len(sys.argv) > 1:
        tcf_path = sys.argv[1]
    else:
        tcf_path = select_file_dialog()

        if not tcf_path:
            from pathlib import Path
            data_dir = Path(__file__).parent.parent.parent.parent / "data"
            if data_dir.exists():
                tcf_files = list(data_dir.rglob("*.TCF"))
                if tcf_files:
                    print("Found TCF files:")
                    for i, f in enumerate(tcf_files[:10]):
                        print(f"  [{i}] {f.name}")
                    try:
                        choice = int(input("\nSelect file number (or Enter for first): ") or "0")
                        tcf_path = str(tcf_files[choice])
                    except (ValueError, IndexError):
                        tcf_path = str(tcf_files[0])

    if not tcf_path or not os.path.exists(tcf_path):
        print("Error: No valid TCF file selected or found.")
        print(f"Usage: python -m tomocube view <path_to_file.TCF>")
        sys.exit(1)

    with TCFViewer(tcf_path) as viewer:
        viewer.show()


if __name__ == "__main__":
    main()
