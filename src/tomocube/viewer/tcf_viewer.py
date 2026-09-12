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
    - Interactive distance and area measurements
    - Export slices or MIP as PNG

Keyboard Shortcuts:
    Navigation:
        Arrow keys      Move the focused slider by one sample
        Scroll wheel     Navigate in focused view
        Home/End         Jump to first/last slice
        Click            Set crosshair position

    Contrast:
        A                Auto-contrast (current slice)
        G                Global auto-contrast
        R                Reset view
        I                Invert colormap
        1-6              Switch colormap

    Fluorescence:
        F                Toggle FL overlay
        N                Show next FL channel

    Measurements:
        D                Distance measurement mode
        P                Polygon/area measurement mode
        C                Clear all measurements

    Export:
        S                Save current slice as PNG
        M                Save MIP as PNG

    General:
        Q, Escape        Quit (or cancel measurement)
"""

from __future__ import annotations

import logging
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
from tomocube.core.exceptions import TCFError
from tomocube.core.types import ViewerState
from tomocube.processing.image import normalize_with_bounds
from tomocube.processing.registration import Z_OFFSET_MODES
from tomocube.viewer.components import (
    FluorescenceMapper, add_scale_bar, adjust_slider, configure_position_slider,
    contrast_limits, plane_extent,
)
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

    def __init__(self, tcf_path: str, z_offset_mode: str = "start", *,
                 timepoint: int = 0, fl_channel: str | None = None, registration_path=None):
        """Initialize the TCF viewer.
        
        Args:
            tcf_path: Path to TCF file
            z_offset_mode: FL Z alignment mode ("start", "center", or "auto")
            timepoint: Zero-based acquisition index in numeric key order
            fl_channel: Channel to display, defaulting to the first available
        """
        self.tcf_path = Path(tcf_path)
        if z_offset_mode not in Z_OFFSET_MODES:
            raise ValueError(f"z_offset_mode must be one of {Z_OFFSET_MODES}")
        self.z_offset_mode = z_offset_mode
        if registration_path is not None and fl_channel is None:
            raise ValueError("A saved registration requires an explicit fluorescence channel")
        self.registration_path = registration_path
        self._initial_timepoint = timepoint
        self._requested_channel = fl_channel

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

        self._histogram_timer = None
        self._histogram_pending: np.ndarray | None = None

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
    def res_x(self) -> float:
        """X resolution in um/pixel."""
        return self.loader.reg_params.ht_res_x

    @property
    def res_y(self) -> float:
        """Y resolution in um/pixel."""
        return self.loader.reg_params.ht_res_y

    @property
    def spacing(self) -> tuple[float, float, float]:
        return self.res_z, self.res_y, self.res_x

    @property
    def res_z(self) -> float:
        """Z resolution in um/slice."""
        return self.loader.reg_params.ht_res_z

    # =========================================================================
    # File Loading
    # =========================================================================

    def _load_file(self) -> None:
        self._loader = TCFFileLoader(self.tcf_path)
        self._loader.load()
        self._loader.load_timepoint(self._initial_timepoint)
        self.s.current_timepoint = self._initial_timepoint
        if self._requested_channel is not None and self._requested_channel not in self.loader.fl_channels:
            raise ValueError(f"Unknown fluorescence channel: {self._requested_channel}")
        available = list(self.loader.fl_data) or self.loader.fl_channels
        self.s.current_fl_channel = self._requested_channel or (available[0] if available else None)
        self._fl_mapper, self.s.fl_vmin, self.s.fl_vmax = self._prepare_fl_mapper(self.loader, self.s.current_fl_channel)
        shape = self.loader.data_3d.shape
        self.s.current_z, self.s.current_y, self.s.current_x = (n // 2 for n in shape)
        self._auto_contrast_global()

    def _prepare_fl_mapper(self, loader: TCFFileLoader, channel: str | None):
        mode, translation = self.z_offset_mode, (0, 0, 0)
        if self.registration_path is not None:
            from tomocube.processing.alignment import load_alignment
            alignment = load_alignment(self.registration_path, loader, channel)
            mode, translation = alignment.z_offset_mode, alignment.translation_um
        if channel in loader.fl_data:
            mapper = FluorescenceMapper(
                loader.fl_data[channel], loader.data_3d.shape,
                loader.reg_params, channel, mode, translation_um=translation,
            )
            return mapper, *loader.get_fl_contrast(channel)
        return None, 0, 1

    def _load_timepoint(self, idx: int) -> None:
        # Prepare the acquisition AND its display mapping before publishing it.
        # A failed registration must not mix old images with new hover/save data.
        candidate = TCFFileLoader(self.tcf_path)
        try:
            candidate.load()
            candidate.load_timepoint(idx)
            mapper, fl_min, fl_max = self._prepare_fl_mapper(candidate, self.s.current_fl_channel)
            low, high = np.percentile(candidate.data_3d, [1, 99])
        except Exception:
            candidate.close()
            raise
        previous = self._loader
        self._loader = candidate
        previous.close()
        self._fl_mapper = mapper
        self.s.fl_vmin, self.s.fl_vmax = fl_min, fl_max
        self.s.vmin, self.s.vmax = float(low), float(high)
        self.s.current_timepoint = idx
        shape = self.loader.data_3d.shape
        self.s.current_z = min(self.s.current_z, shape[0] - 1)
        self.s.current_y = min(self.s.current_y, shape[1] - 1)
        self.s.current_x = min(self.s.current_x, shape[2] - 1)

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
        return plane_extent(self.loader.data_3d.shape, self.spacing, 0)

    def _get_extent_xz(self) -> list[float]:
        return plane_extent(self.loader.data_3d.shape, self.spacing, 1)

    def _get_extent_yz(self) -> list[float]:
        return plane_extent(self.loader.data_3d.shape, self.spacing, 2)







    def _setup_figure(self) -> None:
        """Create the figure and all UI elements.

        Layout is carefully organized to prevent overlaps:
        - Top: Info text (y=0.97) and buttons (y=0.92, 0.88)
        - Middle: Views and colorbars (y=0.25-0.82)
        - Bottom: Sliders (y=0.04-0.18) and colormap selector
        """
        self._fig = plt.figure(figsize=(16, 10), facecolor=self.DARK_BG)
        self._histogram_timer = self.fig.canvas.new_timer(interval=150)
        self._histogram_timer.single_shot = True
        self._histogram_timer.add_callback(self._flush_histogram)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(f"TCF Viewer - {self.tcf_path.name}")

        # Main axes for views - organized layout with proper spacing
        # XY view (main view, left side)
        self.ax_xy = self.fig.add_axes((0.055, 0.28, 0.35, 0.52), facecolor=self.DARK_FG)
        # XZ view (top right orthogonal)
        self.ax_xz = self.fig.add_axes((0.52, 0.59, 0.20, 0.21), facecolor=self.DARK_FG)
        # YZ view (bottom right orthogonal)
        self.ax_yz = self.fig.add_axes((0.52, 0.28, 0.20, 0.21), facecolor=self.DARK_FG)
        # Histogram (far right)
        self.ax_hist = self.fig.add_axes((0.835, 0.28, 0.14, 0.52), facecolor=self.DARK_FG)

        # Colorbar axes - positioned next to their respective view groups
        self.ax_cbar_ht = self.fig.add_axes((0.425, 0.28, 0.012, 0.52), facecolor=self.DARK_FG)
        if self.loader.has_fluorescence:
            self.ax_cbar_fl = self.fig.add_axes((0.735, 0.28, 0.012, 0.52), facecolor=self.DARK_FG)

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
        """Create navigation and contrast sliders with proper spacing."""
        slider_color = "#4a9eff"
        fl_color = "#50c878"
        data = self.loader.data_3d
        z_max = data.shape[0] - 1
        y_max = data.shape[1] - 1

        # Slider layout: stacked vertically in control region (y=0.04 to 0.19)
        slider_h = 0.022
        slider_gap = 0.008

        # Z slider - show in micrometers (top slider)
        z_um_max = max(z_max, 1) * self.res_z
        y_pos = 0.17
        ax_z = self.fig.add_axes((0.08, y_pos, 0.325, slider_h), facecolor=self.DARK_FG)
        self.z_slider = Slider(ax_z, "Z (μm)", 0, z_um_max,
                               valinit=self.s.current_z * self.res_z,
                               color=slider_color)
        self.z_slider.label.set_color("white")
        self.z_slider.valtext.set_color("white")
        self.z_slider.on_changed(self._on_z_change)

        # Track active slider for arrow key control (RangeSlider excluded)
        self._active_slider: Slider = self.z_slider
        self._sliders: list[Slider] = [self.z_slider]

        # Y slider - show in micrometers
        y_pos -= (slider_h + slider_gap)
        y_um_max = max(y_max, 1) * self.res_y
        ax_y = self.fig.add_axes((0.08, y_pos, 0.325, slider_h), facecolor=self.DARK_FG)
        self.y_slider = Slider(ax_y, "Y (μm)", 0, y_um_max,
                               valinit=self.s.current_y * self.res_y,
                               color=slider_color)
        self.y_slider.label.set_color("white")
        self.y_slider.valtext.set_color("white")
        self.y_slider.on_changed(self._on_y_change)
        self._sliders.append(self.y_slider)

        # Contrast slider - show RI values
        y_pos -= (slider_h + slider_gap)
        data_min, data_max = contrast_limits(data)
        ax_c = self.fig.add_axes((0.08, y_pos, 0.325, slider_h), facecolor=self.DARK_FG)
        self.contrast_slider = RangeSlider(ax_c, "RI", data_min, data_max,
                                           valinit=(self.s.vmin, self.s.vmax), color=slider_color)
        self.contrast_slider.label.set_color("white")
        self.contrast_slider.valtext.set_color("white")
        self.contrast_slider.on_changed(self._on_contrast_change)

        # FL alpha slider (if applicable)
        if self.loader.tcf_info.has_fluorescence:
            y_pos -= (slider_h + slider_gap)
            ax_fl = self.fig.add_axes((0.08, y_pos, 0.325, slider_h), facecolor=self.DARK_FG)
            self.fl_alpha_slider = Slider(ax_fl, "FL Alpha", 0, 1, valinit=0.5, color=fl_color)
            self.fl_alpha_slider.label.set_color("white")
            self.fl_alpha_slider.valtext.set_color("white")
            self.fl_alpha_slider.on_changed(self._on_fl_alpha_change)
            self._sliders.append(self.fl_alpha_slider)
            
            # FL Z offset slider - adjust FL position relative to HT
            y_pos -= (slider_h + slider_gap)
            # Range: full HT Z extent in both directions
            fov_z = data.shape[0] * self.res_z
            ax_fl_z = self.fig.add_axes((0.08, y_pos, 0.325, slider_h), facecolor=self.DARK_FG)
            self.fl_z_offset_slider = Slider(
                ax_fl_z, "FL Z (μm)", -fov_z, fov_z,
                valinit=0, color=fl_color
            )
            self.fl_z_offset_slider.label.set_color("white")
            self.fl_z_offset_slider.valtext.set_color("white")
            self.fl_z_offset_slider.on_changed(self._on_fl_z_offset_change)
            self._sliders.append(self.fl_z_offset_slider)

        # Timepoint slider (right side, only if multiple timepoints)
        if len(self.loader.timepoints) > 1:
            ax_t = self.fig.add_axes((0.52, 0.17, 0.20, slider_h), facecolor=self.DARK_FG)
            self.tp_slider = Slider(ax_t, "Time", 0, len(self.loader.timepoints) - 1,
                                    valinit=self.s.current_timepoint, valstep=1, color=slider_color)
            self.tp_slider.label.set_color("white")
            self.tp_slider.valtext.set_color("white")
            self.tp_slider.on_changed(self._on_timepoint_change)
            self._sliders.append(self.tp_slider)

        configure_position_slider(self.z_slider, data.shape[0], self.res_z, self.s.current_z)
        configure_position_slider(self.y_slider, data.shape[1], self.res_y, self.s.current_y)

    def _setup_buttons(self) -> None:
        """Create control buttons with clear, professional labels."""
        btn_w, btn_h = 0.08, 0.032
        btn_y = 0.92
        btn_spacing = 0.085

        # Contrast & display controls (top row)
        buttons = [
            (0.05, "Auto [A]", self._on_auto_contrast),
            (0.05 + btn_spacing, "Global [G]", self._on_global_contrast),
            (0.05 + btn_spacing * 2, "Reset [R]", self._on_reset),
            (0.05 + btn_spacing * 3, "Invert [I]", self._on_invert),
            (0.05 + btn_spacing * 4, "Save [S]", self._on_save_slice),
        ]

        if self.loader.tcf_info.has_fluorescence:
            buttons.append((0.05 + btn_spacing * 5, "FL [F]", self._on_toggle_fluorescence))
        if len(self.loader.fl_channels) > 1:
            buttons.append((0.05 + btn_spacing * 6, "Channel [N]", self._on_next_channel))

        # Measurement tools (second row)
        btn_y2 = 0.88
        meas_buttons = [
            (0.05, "Dist [D]", self._on_start_distance),
            (0.05 + btn_spacing, "Area [P]", self._on_start_area),
            (0.05 + btn_spacing * 2, "Clear [C]", self._on_clear_measurements),
        ]

        self._buttons = []
        for x, label, handler in buttons:
            ax = self.fig.add_axes((x, btn_y, btn_w, btn_h), facecolor=self.DARK_FG)
            btn = Button(ax, label, color=self.DARK_FG, hovercolor="#3d3d3d")
            btn.label.set_color("white")
            btn.label.set_fontsize(8)
            btn.on_clicked(handler)
            self._buttons.append(btn)

        # Measurement buttons
        for x, label, handler in meas_buttons:
            ax = self.fig.add_axes((x, btn_y2, btn_w, btn_h), facecolor=self.DARK_FG)
            btn = Button(ax, label, color=self.DARK_FG, hovercolor="#3d3d3d")
            btn.label.set_color("white")
            btn.label.set_fontsize(8)
            btn.on_clicked(handler)
            self._buttons.append(btn)

        # Colormap selector (positioned in bottom right area)
        ax_cmap = self.fig.add_axes((0.835, 0.04, 0.12, 0.16), facecolor=self.DARK_FG)
        self.cmap_radio = RadioButtons(ax_cmap, self.COLORMAPS, active=0)
        for label in self.cmap_radio.labels:
            label.set_color("white")
            label.set_fontsize(8)
        self.cmap_radio.on_clicked(self._on_cmap_change)

        # Colormap label
        self.fig.text(0.895, 0.205, "Colormap", ha="center", va="bottom",
                      color="#888888", fontsize=8)

    def _setup_info_text(self) -> None:
        """Create info text display and status bar."""
        # File info header
        self.info_text = self.fig.text(0.5, 0.97, "", ha="center", va="top",
                                       color="white", fontsize=10, family="monospace")
        # Status bar at bottom for hover info and measurement status
        self.pixel_text = self.fig.text(0.5, 0.012, "", ha="center", va="bottom",
                                        color="#aaaaaa", fontsize=9, family="monospace")
        # Keyboard hint on far right
        self.fig.text(0.98, 0.012, "Press Q to quit | Scroll to navigate",
                      ha="right", va="bottom", color="#666666", fontsize=8)

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
        y_um = s.current_y * self.res_y
        x_um = s.current_x * self.res_x

        # XY view
        extent_xy = self._get_extent_xy()
        xy_slice = data[s.current_z]
        self._im_xy = self.ax_xy.imshow(xy_slice, cmap=cmap, vmin=s.vmin, vmax=s.vmax,
                                         aspect="equal", extent=extent_xy)
        self._setup_fl_overlay(self.ax_xy, "xy", extent_xy)
        self._setup_crosshairs(self.ax_xy, "xy", x_um, y_um)
        self.ax_xy.set_xlabel("X (μm)", color="white", fontsize=9)
        self.ax_xy.set_ylabel("Y (μm)", color="white", fontsize=9)
        self._title_xy = self.ax_xy.set_title(f"XY plane at Z = {z_um:.1f} μm",
                                               color="white", fontsize=10)
        self._scale_bar = add_scale_bar(self.ax_xy, data.shape[2] * self.res_x)

        # XZ view
        extent_xz = self._get_extent_xz()
        xz_slice = data[:, s.current_y, :]
        self._im_xz = self.ax_xz.imshow(xz_slice, cmap=cmap, vmin=s.vmin, vmax=s.vmax,
                                         aspect="auto", extent=extent_xz)
        self._setup_fl_overlay(self.ax_xz, "xz", extent_xz)
        self._setup_crosshairs(self.ax_xz, "xz", x_um, z_um)
        self.ax_xz.set_xlabel("X (μm)", color="white", fontsize=8)
        self.ax_xz.set_ylabel("Z (μm)", color="white", fontsize=8)
        self._title_xz = self.ax_xz.set_title(f"XZ at Y = {y_um:.1f} μm",
                                               color="white", fontsize=9)

        # YZ view
        extent_yz = self._get_extent_yz()
        yz_slice = data[:, :, s.current_x]
        self._im_yz = self.ax_yz.imshow(yz_slice, cmap=cmap, vmin=s.vmin, vmax=s.vmax,
                                         aspect="auto", extent=extent_yz)
        self._setup_fl_overlay(self.ax_yz, "yz", extent_yz)
        self._setup_crosshairs(self.ax_yz, "yz", y_um, z_um)
        self.ax_yz.set_xlabel("Y (μm)", color="white", fontsize=8)
        self.ax_yz.set_ylabel("Z (μm)", color="white", fontsize=8)
        self._title_yz = self.ax_yz.set_title(f"YZ at X = {x_um:.1f} μm",
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

        self._update_fl_overlays()
        self._update_info_text()

    def _setup_crosshairs(self, ax, view_id: str, x: float, y: float) -> None:
        """Set up crosshair lines for an axis."""
        hline = ax.axhline(y=y, color="#ff6b6b", lw=0.8, alpha=0.7)
        vline = ax.axvline(x=x, color="#50c878", lw=0.8, alpha=0.7)
        self._crosshairs[view_id] = {"h": hline, "v": vline}

    def _setup_fl_overlay(self, ax, plane: str, ht_extent: list[float]) -> None:
        if not self.loader.has_fluorescence:
            return
        # Allocate images even if this acquisition has no data for the channel.
        axis = ("xy", "xz", "yz").index(plane)
        shape = tuple(n for i, n in enumerate(self.loader.data_3d.shape) if i != axis)
        image = ax.imshow(np.zeros((*shape, 4), dtype=np.float32), extent=ht_extent,
                          aspect="equal" if axis == 0 else "auto", interpolation="nearest")
        image.set_visible(False)
        setattr(self, f"_im_fl_{plane}", image)

    def _create_fl_rgba(self, fl_slice: np.ndarray) -> np.ndarray:
        fl_norm = normalize_with_bounds(fl_slice, self.s.fl_vmin, self.s.fl_vmax)
        rgba = np.zeros((*fl_norm.shape, 4), dtype=np.float32)
        rgba[:, :, 1] = fl_norm
        rgba[:, :, 3] = fl_norm * self.s.fl_overlay_alpha
        return rgba





    def _update_display(self) -> None:
        """Fast update using set_data() - no clearing/recreating."""
        data = self.loader.data_3d
        s = self.s

        # Calculate physical positions
        z_um = s.current_z * self.res_z
        y_um = s.current_y * self.res_y
        x_um = s.current_x * self.res_x

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
        self._title_xy.set_text(f"XY plane at Z = {z_um:.1f} μm")
        self._title_xz.set_text(f"XZ at Y = {y_um:.1f} μm")
        self._title_yz.set_text(f"YZ at X = {x_um:.1f} μm")

        self._update_histogram_debounced(xy_slice)
        self._update_info_text()

    def _update_fl_overlays(self) -> None:
        visible = self.s.show_fluorescence and self._fl_mapper is not None
        positions = (self.s.current_z, self.s.current_y, self.s.current_x)
        for axis, plane in enumerate(("xy", "xz", "yz")):
            image = getattr(self, f"_im_fl_{plane}")
            if image is None:
                continue
            image.set_visible(visible)
            if visible:
                result = self._fl_mapper.get_slice(axis, positions[axis], self.s.fl_z_offset_um)
                image.set_data(self._create_fl_rgba(result.data))
                image.set_extent(result.extent)
        if hasattr(self, "ax_cbar_fl"):
            self.ax_cbar_fl.set_visible(visible)
            self._cbar_fl.mappable.set_clim(self.s.fl_vmin, self.s.fl_vmax)

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

        # Update histogram (debounced for performance during rapid navigation)
        xy_slice = self.loader.data_3d[s.current_z]
        self._update_histogram_debounced(xy_slice)

        self._update_info_text()

    def _update_histogram_debounced(self, xy_slice: np.ndarray) -> None:
        # Matplotlib timers run on the GUI event loop, never a worker thread.
        self._histogram_pending = xy_slice
        self._histogram_timer.stop()
        self._histogram_timer.start()

    def _flush_histogram(self) -> None:
        if self._fig is not None and self._histogram_pending is not None:
            self._update_histogram(self._histogram_pending)
            self._histogram_pending = None
            self.fig.canvas.draw_idle()

    def _update_histogram(self, xy_slice: np.ndarray) -> None:
        """Update histogram display immediately (for initial display)."""
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

        fov_x = data.shape[2] * self.res_x
        fov_y = data.shape[1] * self.res_y
        fov_z = data.shape[0] * self.res_z

        parts = [
            f"Time: {self.loader.current_timepoint}",
            f"{info.magnification or '?'}x  NA {info.numerical_aperture or '?'}",
            f"FOV: {fov_x:g} × {fov_y:g} × {fov_z:g} μm",
            f"RI: {self._format_ri(self.s.vmin)} - {self._format_ri(self.s.vmax)}",
        ]

        if self.s.show_fluorescence and self.s.current_fl_channel:
            status = "unavailable at this time" if self._fl_mapper is None else f"alpha={self.s.fl_overlay_alpha:.1f}"
            parts.append(f"FL: {self.s.current_fl_channel} ({status})")

        self.info_text.set_text("  |  ".join(parts))

    def _connect_events(self) -> None:
        """Connect matplotlib events."""
        if self.fig.canvas.manager is not None:
            # Our documented shortcuts own S/P/F/etc.; avoid also firing the
            # toolbar's save dialog, pan mode or fullscreen shortcut.
            self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("close_event", lambda event: self.close())

    def _on_z_change(self, val_um: float) -> None:
        """Handle Z slider change (value is in micrometers)."""
        self.s.current_z = int(round(val_um / self.res_z))
        self.s.current_z = np.clip(self.s.current_z, 0, self.loader.data_3d.shape[0] - 1)
        self._update_display()
        self.fig.canvas.draw_idle()

    def _on_y_change(self, val_um: float) -> None:
        """Handle Y slider change (value is in micrometers)."""
        self.s.current_y = int(round(val_um / self.res_y))
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

    def _on_fl_z_offset_change(self, val: float) -> None:
        self.s.fl_z_offset_um = val
        self._update_fl_overlays()
        self.fig.canvas.draw_idle()

    def _on_timepoint_change(self, val: float) -> None:
        try:
            self._load_timepoint(int(val))
        except (TCFError, OSError, ValueError, IndexError) as error:
            self.tp_slider.eventson = False
            self.tp_slider.set_val(self.s.current_timepoint)
            self.tp_slider.eventson = True
            self.pixel_text.set_text(f"Could not load timepoint {int(val)}: {error}")
            self.fig.canvas.draw_idle()
            return
        self._on_clear_measurements()
        self._update_sliders()
        for image, extent, ax in (
            (self._im_xy, self._get_extent_xy(), self.ax_xy),
            (self._im_xz, self._get_extent_xz(), self.ax_xz),
            (self._im_yz, self._get_extent_yz(), self.ax_yz),
        ):
            image.set_extent(extent)
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
        self._scale_bar.remove()
        self._scale_bar = add_scale_bar(self.ax_xy, self.loader.data_3d.shape[2] * self.res_x)
        self._update_display()
        self._update_contrast()
        self._update_histogram(self.loader.data_3d[self.s.current_z])
        self.fig.canvas.draw_idle()

    def _update_sliders(self) -> None:
        shape = self.loader.data_3d.shape
        configure_position_slider(self.z_slider, shape[0], self.res_z, self.s.current_z)
        configure_position_slider(self.y_slider, shape[1], self.res_y, self.s.current_y)
        slider = self.contrast_slider
        slider.eventson = False
        slider.valmin, slider.valmax = contrast_limits(self.loader.data_3d)
        slider.ax.set_xlim(slider.valmin, slider.valmax)
        slider.set_val((self.s.vmin, self.s.vmax))
        slider.eventson = True
        if hasattr(self, "fl_z_offset_slider"):
            slider = self.fl_z_offset_slider
            fov = shape[0] * self.res_z
            slider.eventson = False
            slider.valmin, slider.valmax = -fov, fov
            slider.ax.set_xlim(-fov, fov)
            self.s.fl_z_offset_um = float(np.clip(self.s.fl_z_offset_um, -fov, fov))
            slider.set_val(self.s.fl_z_offset_um)
            slider.eventson = True

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
        self.y_slider.set_val(self.s.current_y * self.res_y)
        self.contrast_slider.set_val((self.s.vmin, self.s.vmax))

    def _on_invert(self, event: Event | None = None) -> None:
        self.s.invert_cmap = not self.s.invert_cmap
        self._update_contrast()
        self.fig.canvas.draw_idle()

    def _on_toggle_fluorescence(self, event: Event | None = None) -> None:
        self.s.show_fluorescence = not self.s.show_fluorescence
        self._update_fl_overlays()
        self._update_info_text()
        self.fig.canvas.draw_idle()

    def _on_next_channel(self, event: Event | None = None) -> None:
        channels = self.loader.fl_channels
        if not channels:
            return
        start = channels.index(self.s.current_fl_channel)
        failures = []
        for step in range(1, len(channels) + 1):
            channel = channels[(start + step) % len(channels)]
            try:
                mapper, low, high = self._prepare_fl_mapper(self.loader, channel)
                break
            except ValueError as error:
                failures.append(f"{channel}: {error}")
        else:
            self.pixel_text.set_text("No displayable channel: " + "; ".join(failures))
            self.fig.canvas.draw_idle()
            return
        self.s.current_fl_channel = channel
        self._fl_mapper = mapper
        self.s.fl_vmin, self.s.fl_vmax = low, high
        self.s.show_fluorescence = True
        self._update_fl_overlays()
        self._update_info_text()
        self.pixel_text.set_text("Skipped " + "; ".join(failures) if failures else "")
        self.fig.canvas.draw_idle()

    def _on_save_slice(self, event: Event | None = None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        z_um = self.s.current_z * self.res_z
        filename = f"{self.tcf_path.stem}_t{self.loader.current_timepoint}_z{self.s.current_z}_{timestamp}.png"
        filepath = self.tcf_path.parent / filename
        plt.imsave(filepath, self.loader.data_3d[self.s.current_z], cmap=self._get_cmap(),
                   vmin=self.s.vmin, vmax=self.s.vmax)
        self.pixel_text.set_text(f"Saved: {filename}")
        self.fig.canvas.draw_idle()

    def _adjust_slider(self, direction: int) -> None:
        adjust_slider(self._active_slider, direction)

    def _on_key(self, event: Event) -> None:
        key = getattr(event, 'key', None)
        if key is None:
            return

        # Arrow keys control active slider
        if key in ("up", "right"):
            self._adjust_slider(1)
        elif key in ("down", "left"):
            self._adjust_slider(-1)
        elif key == "home":
            self._active_slider.set_val(self._active_slider.valmin)
        elif key == "end":
            self._active_slider.set_val(self._active_slider.valmax)
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
        elif key == "s":
            self._on_save_slice()
        elif key == "f" and self.loader.tcf_info.has_fluorescence:
            self._on_toggle_fluorescence()
        elif key == "n":
            self._on_next_channel()
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
                self.close()
        elif key in ("1", "2", "3", "4", "5", "6"):
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
            self.y_slider.set_val(new_y * self.res_y)

    def _on_click(self, event: Event) -> None:
        xdata = getattr(event, 'xdata', None)
        ydata = getattr(event, 'ydata', None)
        inaxes = getattr(event, 'inaxes', None)
        
        # Check if click is on any slider axis (for arrow key focus)
        if inaxes is not None:
            for slider in self._sliders:
                if inaxes == slider.ax:
                    self._active_slider = slider
                    return  # Don't process as slice navigation
        
        if xdata is None or ydata is None:
            return

        if self._measurement_tool and self._measurement_tool._mode:
            return
        shape = self.loader.data_3d.shape

        if inaxes == self.ax_xy:
            self.s.current_x = int(np.clip(np.floor(xdata / self.res_x + 0.5), 0, shape[2] - 1))
            self.s.current_y = int(np.clip(np.floor(ydata / self.res_y + 0.5), 0, shape[1] - 1))
            self.y_slider.set_val(self.s.current_y * self.res_y)
        elif inaxes == self.ax_xz:
            self.s.current_x = int(np.clip(np.floor(xdata / self.res_x + 0.5), 0, shape[2] - 1))
            self.s.current_z = int(np.clip(np.floor(ydata / self.res_z + 0.5), 0, shape[0] - 1))
            self.z_slider.set_val(self.s.current_z * self.res_z)
        elif inaxes == self.ax_yz:
            self.s.current_y = int(np.clip(np.floor(xdata / self.res_y + 0.5), 0, shape[1] - 1))
            self.s.current_z = int(np.clip(np.floor(ydata / self.res_z + 0.5), 0, shape[0] - 1))
            self.z_slider.set_val(self.s.current_z * self.res_z)
            self.y_slider.set_val(self.s.current_y * self.res_y)

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
            x_px = int(np.floor(xdata / self.res_x + 0.5))
            y_px = int(np.floor(ydata / self.res_y + 0.5))
            if 0 <= x_px < shape[2] and 0 <= y_px < shape[1]:
                val = data[self.s.current_z, y_px, x_px]
                z_um = self.s.current_z * self.res_z
                self.pixel_text.set_text(
                    f"Position: ({xdata:.1f}, {ydata:.1f}, {z_um:.1f}) μm  |  RI = {self._format_ri(val)}"
                )
                self.fig.canvas.draw_idle()
        elif inaxes == self.ax_xz:
            x_px = int(np.floor(xdata / self.res_x + 0.5))
            z_px = int(np.floor(ydata / self.res_z + 0.5))
            if 0 <= x_px < shape[2] and 0 <= z_px < shape[0]:
                val = data[z_px, self.s.current_y, x_px]
                y_um = self.s.current_y * self.res_y
                self.pixel_text.set_text(
                    f"Position: ({xdata:.1f}, {y_um:.1f}, {ydata:.1f}) μm  |  RI = {self._format_ri(val)}"
                )
                self.fig.canvas.draw_idle()

    def _on_save_mip(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.tcf_path.stem}_t{self.loader.current_timepoint}_MIP_{timestamp}.png"
        filepath = self.tcf_path.parent / filename
        plt.imsave(filepath, self.loader.data_mip, cmap=self._get_cmap(),
                   vmin=self.s.vmin, vmax=self.s.vmax)
        self.pixel_text.set_text(f"Saved MIP: {filename}")
        self.fig.canvas.draw_idle()

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
        """Close the file, GUI timer and figure, including window-manager close."""
        if self._histogram_timer is not None:
            self._histogram_timer.stop()
        self._histogram_pending = None
        self._fl_mapper = None
        if self._loader is not None:
            self._loader.close()
            self._loader = None
        if self._fig is not None:
            figure, self._fig = self._fig, None
            plt.close(figure)

    def __enter__(self) -> TCFViewer:
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        self.close()


def main() -> None:
    from tomocube.__main__ import _view_2d
    sys.exit(_view_2d("view", sys.argv[1:]))


if __name__ == "__main__":
    main()
