"""
TCF File Viewer - Interactive 3D Holotomography Visualizer.

Features:
    - Navigate Z-slices with slider, keyboard, or scroll wheel
    - Adjustable contrast with auto and percentile options
    - Multiple colormaps with invert option
    - XY, XZ, YZ orthogonal views with crosshairs
    - Click to navigate in any view
    - Histogram display
    - Export current slice or MIP as PNG
    - Fluorescence overlay with registration

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
    1-6                  Switch colormap
    Escape/Q             Quit
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
from matplotlib.widgets import Button, RadioButtons, RangeSlider, Slider

from tomocube.core.file import TCFFileLoader
from tomocube.core.types import ViewerState
from tomocube.processing.image import normalize_with_bounds
from tomocube.viewer.components import FluorescenceMapper

if TYPE_CHECKING:
    from matplotlib.backend_bases import Event

logger = logging.getLogger(__name__)


class TCFViewer:
    """
    Interactive viewer for Tomocube TCF files.

    Uses component classes for separation of concerns:
        - TCFFileLoader: Handles file I/O and data access
        - ViewerState: Holds all mutable UI state
        - FluorescenceMapper: FL-to-HT coordinate mapping
    """

    COLORMAPS = ["gray", "viridis", "inferno", "turbo", "coolwarm", "bone"]
    DARK_BG = "#1e1e1e"
    DARK_FG = "#2d2d2d"

    def __init__(self, tcf_path: str):
        self.tcf_path = Path(tcf_path)

        # Component classes
        self._loader: TCFFileLoader | None = None
        self.s: ViewerState = ViewerState()  # s = state (short for frequent access)
        self._fl_mapper: FluorescenceMapper | None = None
        self._fig: Figure | None = None

        try:
            self._load_file()
            self._setup_figure()
            self._connect_events()
        except Exception:
            self.close()
            raise

    # =========================================================================
    # Loader Accessors (guaranteed non-None after init)
    # =========================================================================

    @property
    def loader(self) -> TCFFileLoader:
        assert self._loader is not None, "Loader not initialized"
        return self._loader

    @property
    def fig(self) -> Figure:
        assert self._fig is not None, "Figure not created"
        return self._fig

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

        # Initialize position to center
        shape = self.loader.data_3d.shape
        self.s.current_z = shape[0] // 2
        self.s.current_y = shape[1] // 2
        self.s.current_x = shape[2] // 2

        # Update FL contrast if available
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

    # =========================================================================
    # Figure Setup
    # =========================================================================

    def _setup_figure(self) -> None:
        """Create the figure and all UI elements."""
        self._fig = plt.figure(figsize=(14, 9), facecolor=self.DARK_BG)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(f"TCF Viewer - {self.tcf_path.name}")

        # Main axes for views
        self.ax_xy = self.fig.add_axes((0.05, 0.30, 0.42, 0.55), facecolor=self.DARK_FG)
        self.ax_xz = self.fig.add_axes((0.52, 0.55, 0.22, 0.30), facecolor=self.DARK_FG)
        self.ax_yz = self.fig.add_axes((0.52, 0.30, 0.22, 0.22), facecolor=self.DARK_FG)
        self.ax_hist = self.fig.add_axes((0.77, 0.30, 0.20, 0.55), facecolor=self.DARK_FG)

        for ax in [self.ax_xy, self.ax_xz, self.ax_yz, self.ax_hist]:
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("white")

        self._setup_sliders()
        self._setup_buttons()
        self._setup_info_text()
        self._update_display()

    def _setup_sliders(self) -> None:
        """Create navigation and contrast sliders."""
        slider_color = "#4a9eff"
        data = self.loader.data_3d
        z_max = data.shape[0] - 1
        y_max = data.shape[1] - 1

        # Z slider
        ax_z = self.fig.add_axes((0.05, 0.20, 0.42, 0.03), facecolor=self.DARK_FG)
        self.z_slider = Slider(ax_z, "Z", 0, z_max, valinit=self.s.current_z,
                               valstep=1, color=slider_color)
        self.z_slider.label.set_color("white")
        self.z_slider.valtext.set_color("white")
        self.z_slider.on_changed(self._on_z_change)

        # Y slider
        ax_y = self.fig.add_axes((0.05, 0.15, 0.42, 0.03), facecolor=self.DARK_FG)
        self.y_slider = Slider(ax_y, "Y", 0, y_max, valinit=self.s.current_y,
                               valstep=1, color=slider_color)
        self.y_slider.label.set_color("white")
        self.y_slider.valtext.set_color("white")
        self.y_slider.on_changed(self._on_y_change)

        # Contrast slider
        data_min, data_max = float(data.min()), float(data.max())
        ax_c = self.fig.add_axes((0.05, 0.10, 0.42, 0.03), facecolor=self.DARK_FG)
        self.contrast_slider = RangeSlider(ax_c, "Contrast", data_min, data_max,
                                           valinit=(self.s.vmin, self.s.vmax), color=slider_color)
        self.contrast_slider.label.set_color("white")
        self.contrast_slider.valtext.set_color("white")
        self.contrast_slider.on_changed(self._on_contrast_change)

        # FL alpha slider (if applicable)
        if self.loader.tcf_info.has_fluorescence:
            ax_fl = self.fig.add_axes((0.05, 0.05, 0.42, 0.03), facecolor=self.DARK_FG)
            self.fl_alpha_slider = Slider(ax_fl, "FL a", 0, 1, valinit=0.5, color="#50c878")
            self.fl_alpha_slider.label.set_color("white")
            self.fl_alpha_slider.valtext.set_color("white")
            self.fl_alpha_slider.on_changed(self._on_fl_alpha_change)

        # Timepoint slider
        if len(self.loader.timepoints) > 1:
            ax_t = self.fig.add_axes((0.52, 0.20, 0.22, 0.03), facecolor=self.DARK_FG)
            self.tp_slider = Slider(ax_t, "T", 0, len(self.loader.timepoints) - 1,
                                    valinit=0, valstep=1, color=slider_color)
            self.tp_slider.label.set_color("white")
            self.tp_slider.valtext.set_color("white")
            self.tp_slider.on_changed(self._on_timepoint_change)

    def _setup_buttons(self) -> None:
        """Create control buttons."""
        btn_w, btn_h = 0.08, 0.035
        btn_y = 0.91

        # Button definitions: (x, label, handler)
        buttons = [
            (0.05, "Auto", self._on_auto_contrast),
            (0.14, "Global", self._on_global_contrast),
            (0.23, "Reset", self._on_reset),
            (0.32, "Invert", self._on_invert),
            (0.41, "Save", self._on_save_slice),
        ]

        if self.loader.tcf_info.has_fluorescence:
            buttons.append((0.50, "FL +/-", self._on_toggle_fluorescence))

        self._buttons = []
        for x, label, handler in buttons:
            ax = self.fig.add_axes((x, btn_y, btn_w, btn_h), facecolor=self.DARK_FG)
            btn = Button(ax, label, color=self.DARK_FG, hovercolor="#3d3d3d")
            btn.label.set_color("white")
            btn.on_clicked(handler)
            self._buttons.append(btn)

        # Colormap selector
        ax_cmap = self.fig.add_axes((0.77, 0.05, 0.15, 0.18), facecolor=self.DARK_FG)
        self.cmap_radio = RadioButtons(ax_cmap, self.COLORMAPS, active=0)
        for label in self.cmap_radio.labels:
            label.set_color("white")
            label.set_fontsize(9)
        self.cmap_radio.on_clicked(self._on_cmap_change)

    def _setup_info_text(self) -> None:
        """Create info text display."""
        self.info_text = self.fig.text(0.5, 0.97, "", ha="center", va="top",
                                       color="white", fontsize=10, family="monospace")
        self.pixel_text = self.fig.text(0.5, 0.01, "", ha="center", va="bottom",
                                        color="#888888", fontsize=9, family="monospace")

    # =========================================================================
    # Display Update
    # =========================================================================

    def _update_display(self) -> None:
        """Update all views."""
        cmap = self._get_cmap()
        data = self.loader.data_3d
        s = self.s

        # Clear all axes
        for ax in [self.ax_xy, self.ax_xz, self.ax_yz, self.ax_hist]:
            ax.clear()

        # XY view
        xy_slice = data[s.current_z]
        self.ax_xy.imshow(xy_slice, cmap=cmap, vmin=s.vmin, vmax=s.vmax, aspect="equal")
        if s.show_fluorescence:
            self._overlay_fl(self.ax_xy, "xy")
        self._draw_crosshairs(self.ax_xy, s.current_x, s.current_y)
        self.ax_xy.set_title(f"XY (Z={s.current_z})", color="white", fontsize=10)

        # XZ view
        xz_slice = data[:, s.current_y, :]
        self.ax_xz.imshow(xz_slice, cmap=cmap, vmin=s.vmin, vmax=s.vmax, aspect="auto")
        if s.show_fluorescence:
            self._overlay_fl(self.ax_xz, "xz")
        self._draw_crosshairs(self.ax_xz, s.current_x, s.current_z)
        self.ax_xz.set_title(f"XZ (Y={s.current_y})", color="white", fontsize=10)

        # YZ view
        yz_slice = data[:, :, s.current_x]
        self.ax_yz.imshow(yz_slice, cmap=cmap, vmin=s.vmin, vmax=s.vmax, aspect="auto")
        if s.show_fluorescence:
            self._overlay_fl(self.ax_yz, "yz")
        self._draw_crosshairs(self.ax_yz, s.current_y, s.current_z)
        self.ax_yz.set_title(f"YZ (X={s.current_x})", color="white", fontsize=10)

        # Histogram
        self.ax_hist.hist(xy_slice.ravel(), bins=100, color="#4a9eff", alpha=0.7)
        self.ax_hist.axvline(s.vmin, color="#ff6b6b", ls="--", lw=1.5)
        self.ax_hist.axvline(s.vmax, color="#ff6b6b", ls="--", lw=1.5)
        self.ax_hist.set_title("RI Histogram", color="white", fontsize=10)

        # Update info text
        self._update_info_text()

    def _draw_crosshairs(self, ax, x: int, y: int) -> None:
        """Draw crosshairs at position."""
        ax.axhline(y=y, color="#ff6b6b", lw=0.8, alpha=0.7)
        ax.axvline(x=x, color="#50c878", lw=0.8, alpha=0.7)

    def _update_info_text(self) -> None:
        """Update the info text display."""
        info = self.loader.tcf_info
        data = self.loader.data_3d
        z_um = self.s.current_z * self.loader.reg_params.ht_res_z

        parts = [
            f"{info.magnification or '?'}x NA{info.numerical_aperture or '?'}",
            f"{data.shape[2]}x{data.shape[1]}x{data.shape[0]}",
            f"RI: [{self._format_ri(data.min())}-{self._format_ri(data.max())}]",
            f"Z={z_um:.1f}um",
        ]

        if self.s.show_fluorescence and self.s.current_fl_channel:
            parts.append(f"FL: {self.s.current_fl_channel}")

        self.info_text.set_text("  |  ".join(parts))

    # =========================================================================
    # Fluorescence Overlay
    # =========================================================================

    def _overlay_fl(self, ax, plane: str) -> None:
        """Overlay fluorescence on the given axis for the specified plane."""
        ch = self.s.current_fl_channel
        if ch is None or ch not in self.loader.fl_data:
            return

        fl_3d = self.loader.fl_data[ch]
        ht_shape = self.loader.data_3d.shape

        if plane == "xy":
            fl_slice = self._get_fl_xy_slice(fl_3d)
            if fl_slice is None:
                return
            extent = [0, ht_shape[2] - 1, ht_shape[1] - 1, 0]
        elif plane == "xz":
            fl_slice = self._get_fl_xz_slice(fl_3d)
            extent = [0, ht_shape[2] - 1, ht_shape[0] - 1, 0]
        else:  # yz
            fl_slice = self._get_fl_yz_slice(fl_3d)
            extent = [0, ht_shape[1] - 1, ht_shape[0] - 1, 0]

        # Normalize and create RGBA overlay
        fl_norm = normalize_with_bounds(fl_slice, self.s.fl_vmin, self.s.fl_vmax)
        fl_rgba = np.zeros((*fl_norm.shape, 4))
        fl_rgba[:, :, 1] = fl_norm  # Green channel
        fl_rgba[:, :, 3] = fl_norm * self.s.fl_overlay_alpha

        ax.imshow(fl_rgba, extent=extent, aspect="auto" if plane != "xy" else "equal")

    def _get_fl_xy_slice(self, fl_3d: np.ndarray) -> np.ndarray | None:
        """Get FL slice for XY view using FluorescenceMapper."""
        if self._fl_mapper is None:
            return None
        return self._fl_mapper.get_xy_slice(fl_3d, self.s.current_z, self.loader.data_3d.shape)

    def _get_fl_xz_slice(self, fl_3d: np.ndarray) -> np.ndarray:
        """Get FL slice for XZ view using FluorescenceMapper."""
        shape = self.loader.data_3d.shape
        if self._fl_mapper is None:
            return np.zeros((shape[0], shape[2]))
        return self._fl_mapper.get_xz_slice(fl_3d, self.s.current_y, shape)

    def _get_fl_yz_slice(self, fl_3d: np.ndarray) -> np.ndarray:
        """Get FL slice for YZ view using FluorescenceMapper."""
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

    def _on_z_change(self, val: float) -> None:
        self.s.current_z = int(val)
        self._update_display()
        self.fig.canvas.draw_idle()

    def _on_y_change(self, val: float) -> None:
        self.s.current_y = int(val)
        self._update_display()
        self.fig.canvas.draw_idle()

    def _on_contrast_change(self, val: tuple[float, float]) -> None:
        if val[0] <= val[1]:
            self.s.vmin, self.s.vmax = val
            self._update_display()
            self.fig.canvas.draw_idle()

    def _on_fl_alpha_change(self, val: float) -> None:
        self.s.fl_overlay_alpha = val
        if self.s.show_fluorescence:
            self._update_display()
            self.fig.canvas.draw_idle()

    def _on_timepoint_change(self, val: float) -> None:
        self._load_timepoint(int(val))
        self._update_sliders()
        self._update_display()
        self.fig.canvas.draw_idle()

    def _update_sliders(self) -> None:
        """Update slider ranges after loading new data."""
        shape = self.loader.data_3d.shape
        self.z_slider.valmax = shape[0] - 1
        self.y_slider.valmax = shape[1] - 1

    def _on_cmap_change(self, label: str | None) -> None:
        if label is None:
            return
        self.s.colormap = label
        self._update_display()
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
        self.z_slider.set_val(self.s.current_z)
        self.y_slider.set_val(self.s.current_y)
        self.contrast_slider.set_val((self.s.vmin, self.s.vmax))

    def _on_invert(self, event: Event | None = None) -> None:
        self.s.invert_cmap = not self.s.invert_cmap
        self._update_display()
        self.fig.canvas.draw_idle()

    def _on_toggle_fluorescence(self, event: Event | None = None) -> None:
        self.s.show_fluorescence = not self.s.show_fluorescence
        print(f"Fluorescence: {'ON' if self.s.show_fluorescence else 'OFF'}")
        self._update_display()
        self.fig.canvas.draw_idle()

    def _on_save_slice(self, event: Event | None = None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.tcf_path.stem}_z{self.s.current_z:03d}_{timestamp}.png"
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
            self.z_slider.set_val(min(self.s.current_z + 1, z_max))
        elif key in ("down", "s"):
            self.z_slider.set_val(max(self.s.current_z - 1, 0))
        elif key == "home":
            self.z_slider.set_val(0)
        elif key == "end":
            self.z_slider.set_val(z_max)
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
        elif key in ("q", "escape"):
            plt.close(self.fig)
        elif key in "123456":
            idx = int(key) - 1
            if idx < len(self.COLORMAPS):
                self.s.colormap = self.COLORMAPS[idx]
                self._update_display()
                self.fig.canvas.draw_idle()

    def _on_scroll(self, event: Event) -> None:
        button = getattr(event, 'button', None)
        inaxes = getattr(event, 'inaxes', None)
        delta = 1 if button == "up" else -1
        shape = self.loader.data_3d.shape
        z_max = shape[0] - 1
        y_max = shape[1] - 1

        if inaxes == self.ax_xy:
            self.z_slider.set_val(np.clip(self.s.current_z + delta, 0, z_max))
        elif inaxes == self.ax_xz:
            self.y_slider.set_val(np.clip(self.s.current_y + delta, 0, y_max))

    def _on_click(self, event: Event) -> None:
        xdata = getattr(event, 'xdata', None)
        ydata = getattr(event, 'ydata', None)
        inaxes = getattr(event, 'inaxes', None)
        if not xdata or not ydata:
            return

        x, y = int(xdata), int(ydata)
        shape = self.loader.data_3d.shape

        if inaxes == self.ax_xy:
            self.s.current_x = np.clip(x, 0, shape[2] - 1)
            self.s.current_y = np.clip(y, 0, shape[1] - 1)
            self.y_slider.set_val(self.s.current_y)
        elif inaxes == self.ax_xz:
            self.s.current_x = np.clip(x, 0, shape[2] - 1)
            self.s.current_z = np.clip(y, 0, shape[0] - 1)
            self.z_slider.set_val(self.s.current_z)
        elif inaxes == self.ax_yz:
            self.s.current_y = np.clip(x, 0, shape[1] - 1)
            self.s.current_z = np.clip(y, 0, shape[0] - 1)
            self.z_slider.set_val(self.s.current_z)
            self.y_slider.set_val(self.s.current_y)

        self._update_display()
        self.fig.canvas.draw_idle()

    def _on_motion(self, event: Event) -> None:
        xdata = getattr(event, 'xdata', None)
        ydata = getattr(event, 'ydata', None)
        inaxes = getattr(event, 'inaxes', None)
        if not xdata or not ydata:
            return

        x, y = int(xdata), int(ydata)
        data = self.loader.data_3d
        shape = data.shape

        if inaxes == self.ax_xy:
            if 0 <= x < shape[2] and 0 <= y < shape[1]:
                val = data[self.s.current_z, y, x]
                self.pixel_text.set_text(f"XY({x}, {y}) Z={self.s.current_z} -> RI={self._format_ri(val)}")
                self.fig.canvas.draw_idle()
        elif inaxes == self.ax_xz:
            if 0 <= x < shape[2] and 0 <= y < shape[0]:
                val = data[y, self.s.current_y, x]
                self.pixel_text.set_text(f"XZ({x}, {y}) Y={self.s.current_y} -> RI={self._format_ri(val)}")
                self.fig.canvas.draw_idle()

    def _on_save_mip(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.tcf_path.stem}_MIP_{timestamp}.png"
        filepath = self.tcf_path.parent / filename
        plt.imsave(filepath, self.loader.data_mip, cmap=self._get_cmap(),
                   vmin=self.s.vmin, vmax=self.s.vmax)
        print(f"Saved: {filepath}")

    # =========================================================================
    # Public API
    # =========================================================================

    def show(self) -> None:
        """Display the viewer."""
        plt.show()

    def close(self) -> None:
        """Close file and cleanup resources."""
        # Close HDF5 file via loader (this also clears data arrays)
        if self._loader is not None:
            try:
                self._loader.close()
            except Exception:
                logger.debug("Error closing HDF5 file during cleanup", exc_info=True)
            finally:
                self._loader = None

        # Close matplotlib figure
        if self._fig is not None:
            try:
                plt.close(self._fig)
            except Exception:
                logger.debug("Error closing matplotlib figure during cleanup", exc_info=True)
            finally:
                self._fig = None

    def __enter__(self) -> TCFViewer:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        """Context manager exit - ensures file is closed."""
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
            # Try to find test files in data directory
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
