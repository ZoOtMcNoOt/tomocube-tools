"""
Interactive Slice Viewer for TCF Files.

Compare HT and FL slices side-by-side with overlay, showing proper
physical units (micrometers) and scientific visualization.

Optimized for responsive navigation using set_data() updates.

Usage:
    python -m tomocube slice path/to/file.TCF   # View specific file

Controls:
    - Slider or arrow keys: Navigate Z slices
    - Home/End: Jump to first/last slice
    - Q/Escape: Quit
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.widgets import Button, Slider

from tomocube.core.file import TCFFileLoader
from tomocube.processing.registration import Z_OFFSET_MODES
from tomocube.viewer.components import (
    FluorescenceMapper, add_scale_bar, adjust_slider, configure_position_slider, plane_extent,
)
from tomocube.processing.image import normalize_with_bounds


class SliceViewer:
    """Interactive viewer for comparing HT and FL slices with physical units."""

    DARK_BG = "#1e1e1e"
    DARK_FG = "#2d2d2d"

    def __init__(self, tcf_path: str | Path, z_offset_mode: str = "start", *,
                 timepoint: int = 0, fl_channel: str | None = None) -> None:
        """Initialize the slice viewer.
        
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
        self.timepoint = timepoint
        self.fl_channel = fl_channel
        self.fl_z_offset_um = 0.0
        self._fl_mapper = None
        self.fig = None

        # Image references for fast updates
        self._im_ht: AxesImage | None = None
        self._im_fl: AxesImage | None = None
        self._im_overlay: AxesImage | None = None
        self._title_ht = None
        self._title_fl = None

        try:
            self._load_data()
            self._setup_figure()
        except Exception:
            self.close()
            raise

    def _load_data(self) -> None:
        """Use the common loader for numeric timepoints and calibrated channels."""
        with TCFFileLoader(self.tcf_path) as loader:
            loader.load_timepoint(self.timepoint)
            self.timepoint_key = loader.current_timepoint
            self.ht_3d = loader.data_3d
            self.params = loader.reg_params
            if self.fl_channel is not None and self.fl_channel not in loader.fl_data:
                raise ValueError(f"Fluorescence channel {self.fl_channel} is unavailable at timepoint {self.timepoint_key}")
            self.fl_channel = self.fl_channel or next(iter(loader.fl_data), None)
            self.fl_3d = loader.fl_data.get(self.fl_channel)
            self.has_fl = self.fl_3d is not None
            self.fl_vmin, self.fl_vmax = loader.get_fl_contrast(self.fl_channel) if self.has_fl else (0, 1)
        self.spacing = (self.params.ht_res_z, self.params.ht_res_y, self.params.ht_res_x)
        self.res_z = self.params.ht_res_z
        self.ht_fov_z, self.ht_fov_y, self.ht_fov_x = (
            n * d for n, d in zip(self.ht_3d.shape, self.spacing)
        )
        self.fov_x, self.fov_y, self.fov_z = self.ht_fov_x, self.ht_fov_y, self.ht_fov_z
        self.ht_vmin, self.ht_vmax = np.percentile(self.ht_3d, [1, 99])
        if self.has_fl:
            self._fl_mapper = FluorescenceMapper(
                self.fl_3d, self.ht_3d.shape, self.params, self.fl_channel, self.z_offset_mode,
            )

    def _get_ht_extent(self) -> list[float]:
        return plane_extent(self.ht_3d.shape, self.spacing)



    def _get_fl_slice_at_z_um(self, z_um: float) -> tuple[np.ndarray | None, bool]:
        if self._fl_mapper is None:
            return None, False
        index = int(np.clip(round(z_um / self.res_z), 0, self.ht_3d.shape[0] - 1))
        result = self._fl_mapper.get_slice(0, index, self.fl_z_offset_um)
        return result.data, result.in_range



    def _setup_figure(self) -> None:
        """Setup the matplotlib figure with sliders and colorbars."""
        ncols = 3 if self.has_fl else 1
        self.fig = plt.figure(figsize=(5 * ncols + 1, 6), facecolor=self.DARK_BG)

        # Initial slice
        self.current_z = self.ht_3d.shape[0] // 2
        z_um = self.current_z * self.res_z
        ht_extent = self._get_ht_extent()

        # Create HT axes and colorbar
        self.ax_ht = self.fig.add_axes((0.05, 0.25, 0.25 if self.has_fl else 0.8, 0.55),
                                        facecolor=self.DARK_FG)
        self.ax_ht_cbar = self.fig.add_axes((0.31 if self.has_fl else 0.87, 0.25, 0.015, 0.55))

        ht_slice = self.ht_3d[self.current_z]
        self._im_ht = self.ax_ht.imshow(ht_slice, cmap="gray", vmin=self.ht_vmin,
                                         vmax=self.ht_vmax, extent=ht_extent)
        self.ax_ht.set_xlabel("X (μm)", color="white", fontsize=9)
        self.ax_ht.set_ylabel("Y (μm)", color="white", fontsize=9)
        self._title_ht = self.ax_ht.set_title(f"HT at Z = {z_um:.1f} μm",
                                               color="white", fontsize=10)
        self.ax_ht.tick_params(colors="white", labelsize=8)
        add_scale_bar(self.ax_ht, self.fov_x)

        # HT colorbar
        cbar_ht = self.fig.colorbar(self._im_ht, cax=self.ax_ht_cbar)
        cbar_ht.set_label("RI", color="white", fontsize=9)
        cbar_ht.ax.yaxis.set_label_position("left")
        cbar_ht.ax.tick_params(colors="white", labelsize=8)

        if self.has_fl:
            assert self.fl_3d is not None
            fl_extent = self._get_ht_extent()

            # FL axes and colorbar - use same physical coordinate space as HT
            self.ax_fl = self.fig.add_axes((0.37, 0.25, 0.25, 0.55), facecolor=self.DARK_FG)
            self.ax_fl_cbar = self.fig.add_axes((0.63, 0.25, 0.015, 0.55))

            fl_slice, has_data = self._get_fl_slice_at_z_um(z_um)
            self._im_fl = self.ax_fl.imshow(fl_slice, cmap="Greens", vmin=self.fl_vmin,
                                             vmax=self.fl_vmax, extent=fl_extent)
            # Set axis limits to match HT coordinate space
            self.ax_fl.set_xlim(ht_extent[:2])
            self.ax_fl.set_ylim(ht_extent[2:])
            self.ax_fl.set_xlabel("X (μm)", color="white", fontsize=9)
            self.ax_fl.set_ylabel("Y (μm)", color="white", fontsize=9)
            status = "(in range)" if has_data else "(no data)"
            self._title_fl = self.ax_fl.set_title(f"FL {self.fl_channel} {status}", color="white", fontsize=10)
            self.ax_fl.tick_params(colors="white", labelsize=8)
            add_scale_bar(self.ax_fl, self.fov_x)

            # FL colorbar
            cbar_fl = self.fig.colorbar(self._im_fl, cax=self.ax_fl_cbar)
            cbar_fl.set_label("Intensity", color="white", fontsize=9)
            cbar_fl.ax.tick_params(colors="white", labelsize=8)

            # Overlay axes - show both in same physical space
            self.ax_overlay = self.fig.add_axes((0.70, 0.25, 0.25, 0.55), facecolor=self.DARK_FG)

            # HT as background (red channel)
            ht_norm = normalize_with_bounds(ht_slice, self.ht_vmin, self.ht_vmax)
            rgb = np.zeros((*ht_norm.shape, 3), dtype=np.float32)
            rgb[:, :, 0] = ht_norm
            self._im_overlay_ht = self.ax_overlay.imshow(rgb, extent=ht_extent)
            
            # FL sampled on the HT plane (green channel)
            fl_norm = normalize_with_bounds(fl_slice if fl_slice is not None else np.zeros((1,1)), 
                                           self.fl_vmin, self.fl_vmax)
            fl_rgba = np.zeros((*fl_norm.shape, 4), dtype=np.float32)
            fl_rgba[:, :, 1] = fl_norm  # Green channel
            fl_rgba[:, :, 3] = fl_norm * 0.7  # Alpha based on intensity
            self._im_overlay_fl = self.ax_overlay.imshow(fl_rgba, extent=fl_extent)
            
            self.ax_overlay.set_xlim(ht_extent[:2])
            self.ax_overlay.set_ylim(ht_extent[2:])
            self.ax_overlay.set_xlabel("X (μm)", color="white", fontsize=9)
            self.ax_overlay.set_ylabel("Y (μm)", color="white", fontsize=9)
            self.ax_overlay.set_title("Overlay (R=HT, G=FL)", color="white", fontsize=10)
            self.ax_overlay.tick_params(colors="white", labelsize=8)
            add_scale_bar(self.ax_overlay, self.fov_x)

        # Z slider - show in micrometers
        z_um_max = max(self.ht_3d.shape[0] - 1, 1) * self.res_z
        ax_slider = self.fig.add_axes((0.15, 0.08, 0.7, 0.03), facecolor=self.DARK_FG)
        self.slider = Slider(
            ax_slider, "Z (μm)", 0, z_um_max,
            valinit=z_um, color="#4a9eff"
        )
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")
        self.slider.on_changed(self._update_slice)
        configure_position_slider(self.slider, self.ht_3d.shape[0], self.res_z, self.current_z)

        # Track active slider for arrow key control
        self._active_slider = self.slider
        self._sliders: list[Slider] = [self.slider]

        # FL Z offset slider (if FL data available)
        if self.has_fl:
            # Use HT FOV as offset range
            fov_z = self.ht_fov_z
            ax_fl_z = self.fig.add_axes((0.15, 0.03, 0.7, 0.03), facecolor=self.DARK_FG)
            self.fl_z_offset_slider = Slider(
                ax_fl_z, "FL Z offset", -fov_z, fov_z,
                valinit=0, color="#50c878"
            )
            self.fl_z_offset_slider.label.set_color("white")
            self.fl_z_offset_slider.valtext.set_color("white")
            self.fl_z_offset_slider.on_changed(self._on_fl_z_offset_change)
            self.fl_z_offset_um = 0.0
            self._sliders.append(self.fl_z_offset_slider)

        # Connect click event for slider focus
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Navigation buttons
        ax_prev = self.fig.add_axes((0.15, 0.14, 0.08, 0.04))
        ax_next = self.fig.add_axes((0.77, 0.14, 0.08, 0.04))
        self.btn_prev = Button(ax_prev, "< Prev", color=self.DARK_FG, hovercolor="#3d3d3d")
        self.btn_next = Button(ax_next, "Next >", color=self.DARK_FG, hovercolor="#3d3d3d")
        self.btn_prev.label.set_color("white")
        self.btn_next.label.set_color("white")
        self.btn_prev.on_clicked(self._prev_slice)
        self.btn_next.on_clicked(self._next_slice)

        # Keyboard navigation
        if self.fig.canvas.manager is not None:
            self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", lambda event: self.close())

        # Title with physical info
        self.fig.suptitle(
            f"{self.tcf_path.stem}  |  Time: {self.timepoint_key}  |  FOV: {self.fov_x:g} × {self.fov_y:g} × {self.fov_z:g} μm",
            fontsize=10, color="white"
        )

    def _update_slice(self, z_um: float) -> None:
        """Update displayed slices using set_data() for speed."""
        self.current_z = int(round(z_um / self.res_z))
        self.current_z = np.clip(self.current_z, 0, self.ht_3d.shape[0] - 1)
        z_um_actual = self.current_z * self.res_z

        # Fast update - use set_data instead of recreating images
        ht_slice = self.ht_3d[self.current_z]
        self._im_ht.set_data(ht_slice)
        self._title_ht.set_text(f"HT at Z = {z_um_actual:.1f} μm")

        if self.has_fl:
            assert self.fl_3d is not None
            fl_slice, has_data = self._get_fl_slice_at_z_um(z_um_actual)
            self._im_fl.set_data(fl_slice)

            status = "(in range)" if has_data else "(no data)"
            self._title_fl.set_text(f"FL {self.fl_channel} {status} at Z = {z_um_actual:.1f} μm")

            # Update overlay - HT layer
            ht_norm = normalize_with_bounds(ht_slice, self.ht_vmin, self.ht_vmax)
            rgb = np.zeros((*ht_norm.shape, 3), dtype=np.float32)
            rgb[:, :, 0] = ht_norm
            self._im_overlay_ht.set_data(rgb)
            
            # Update overlay - FL layer
            fl_norm = normalize_with_bounds(fl_slice if fl_slice is not None else np.zeros((1,1)), 
                                           self.fl_vmin, self.fl_vmax)
            fl_rgba = np.zeros((*fl_norm.shape, 4), dtype=np.float32)
            fl_rgba[:, :, 1] = fl_norm  # Green channel
            fl_rgba[:, :, 3] = fl_norm * 0.7  # Alpha based on intensity
            self._im_overlay_fl.set_data(fl_rgba)

        self.fig.canvas.draw_idle()

    def _on_fl_z_offset_change(self, val: float) -> None:
        """Handle FL Z offset slider changes."""
        self.fl_z_offset_um = val
        
        # Update the current slice at the current Z position
        z_um_actual = self.current_z * self.res_z
        self._update_slice(z_um_actual)

    def _prev_slice(self, event: object = None) -> None:
        """Go to previous slice."""
        if self.current_z > 0:
            self.slider.set_val((self.current_z - 1) * self.res_z)

    def _next_slice(self, event: object = None) -> None:
        """Go to next slice."""
        if self.current_z < self.ht_3d.shape[0] - 1:
            self.slider.set_val((self.current_z + 1) * self.res_z)

    def _on_click(self, event: object) -> None:
        """Handle mouse clicks to set active slider."""
        if event is None or getattr(event, 'inaxes', None) is None:
            return
        # Check if click is on any slider axis
        for slider in self._sliders:
            if event.inaxes == slider.ax:
                self._active_slider = slider
                break

    def _adjust_slider(self, direction: int) -> None:
        adjust_slider(self._active_slider, direction)

    def _on_key(self, event: object) -> None:
        """Handle keyboard events."""
        key = getattr(event, "key", "")

        if key in ("left", "down"):
            self._adjust_slider(-1)
        elif key in ("right", "up"):
            self._adjust_slider(1)
        elif key == "home":
            self._active_slider.set_val(self._active_slider.valmin)
        elif key == "end":
            self._active_slider.set_val(self._active_slider.valmax)
        elif key in ("q", "escape"):
            self.close()

    def show(self) -> None:
        plt.show()

    def close(self) -> None:
        if self.fig is not None:
            figure, self.fig = self.fig, None
            plt.close(figure)
        self._fl_mapper = None

    def __enter__(self) -> SliceViewer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

def main() -> None:
    from tomocube.__main__ import _view_2d
    sys.exit(_view_2d("slice", sys.argv[1:]))


if __name__ == "__main__":
    main()
