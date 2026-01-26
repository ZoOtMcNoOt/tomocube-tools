"""
Interactive Slice Viewer for TCF Files.

Compare HT and FL slices side-by-side with overlay, showing proper
physical units (micrometers) and scientific visualization.

Optimized for responsive navigation using set_data() updates.

Usage:
    python -m tomocube slice                    # Use default test file
    python -m tomocube slice path/to/file.TCF   # View specific file

Controls:
    - Slider or arrow keys: Navigate Z slices
    - Home/End: Jump to first/last slice
    - Q/Escape: Quit
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.widgets import Button, Slider

from tomocube.core.file import load_registration_params
from tomocube.processing.image import normalize_with_bounds
from tomocube.processing.registration import register_fl_to_ht


class SliceViewer:
    """Interactive viewer for comparing HT and FL slices with physical units."""

    DARK_BG = "#1e1e1e"
    DARK_FG = "#2d2d2d"

    def __init__(self, tcf_path: str | Path) -> None:
        self.tcf_path = Path(tcf_path)

        # Image references for fast updates
        self._im_ht: AxesImage | None = None
        self._im_fl: AxesImage | None = None
        self._im_overlay: AxesImage | None = None
        self._title_ht = None
        self._title_fl = None

        self._load_data()
        self._setup_figure()

    def _load_data(self) -> None:
        """Load HT and FL data from TCF file."""
        print(f"Loading: {self.tcf_path.name}")

        with h5py.File(self.tcf_path, "r") as f:
            # Load HT
            self.ht_3d: np.ndarray = np.asarray(f["Data/3D/000000"]).astype(np.float32)
            print(f"  HT shape: {self.ht_3d.shape}")

            # Get resolution info
            data_3d = f["Data/3D"]
            res_x = data_3d.attrs.get("ResolutionX", 0.196)
            res_z = data_3d.attrs.get("ResolutionZ", 0.839)
            # Handle array attributes (extract first element if array)
            self.res_xy = float(res_x[0]) if hasattr(res_x, '__len__') else float(res_x)
            self.res_z = float(res_z[0]) if hasattr(res_z, '__len__') else float(res_z)

            # Load FL if available
            self.has_fl = "Data/3DFL/CH0/000000" in f

            if self.has_fl:
                fl_raw: np.ndarray = np.asarray(f["Data/3DFL/CH0/000000"]).astype(np.float32)
                params = load_registration_params(f)
                print(f"  FL shape: {fl_raw.shape}")

                # Register FL to HT coordinate space
                self.fl_3d: np.ndarray | None = register_fl_to_ht(fl_raw, self.ht_3d.shape, params)

                # Calculate FL Z coverage
                self.fl_z_start = int(params.fl_offset_z / params.ht_res_z)
                self.fl_z_end = self.fl_z_start + int(
                    fl_raw.shape[0] * params.fl_res_z / params.ht_res_z
                )
                print(f"  FL covers HT slices: {self.fl_z_start} to {self.fl_z_end}")
            else:
                print("  No FL data found")
                self.fl_3d = None
                self.fl_z_start = 0
                self.fl_z_end = 0

        # Compute normalization bounds
        self.ht_vmin, self.ht_vmax = np.percentile(self.ht_3d, [1, 99])

        if self.has_fl and self.fl_3d is not None:
            fl_nonzero = self.fl_3d[self.fl_3d > 0]
            if len(fl_nonzero) > 0:
                self.fl_vmin, self.fl_vmax = np.percentile(fl_nonzero, [1, 99])
            else:
                self.fl_vmin, self.fl_vmax = 0.0, 1.0
        else:
            self.fl_vmin, self.fl_vmax = 0.0, 1.0

        # Calculate physical dimensions
        self.fov_x = self.ht_3d.shape[2] * self.res_xy
        self.fov_y = self.ht_3d.shape[1] * self.res_xy
        self.fov_z = self.ht_3d.shape[0] * self.res_z

    def _get_extent(self) -> list[float]:
        """Get extent for images in micrometers."""
        return [0, self.fov_x, self.fov_y, 0]

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

    def _setup_figure(self) -> None:
        """Setup the matplotlib figure with sliders and colorbars."""
        ncols = 3 if self.has_fl else 1
        self.fig = plt.figure(figsize=(5 * ncols + 1, 6), facecolor=self.DARK_BG)

        # Initial slice
        self.current_z = self.ht_3d.shape[0] // 2
        z_um = self.current_z * self.res_z
        extent = self._get_extent()

        # Create HT axes and colorbar
        self.ax_ht = self.fig.add_axes((0.05, 0.25, 0.25 if self.has_fl else 0.8, 0.55),
                                        facecolor=self.DARK_FG)
        self.ax_ht_cbar = self.fig.add_axes((0.31 if self.has_fl else 0.87, 0.25, 0.015, 0.55))

        ht_slice = self.ht_3d[self.current_z]
        self._im_ht = self.ax_ht.imshow(ht_slice, cmap="gray", vmin=self.ht_vmin,
                                         vmax=self.ht_vmax, extent=extent)
        self.ax_ht.set_xlabel("X (um)", color="white", fontsize=9)
        self.ax_ht.set_ylabel("Y (um)", color="white", fontsize=9)
        self._title_ht = self.ax_ht.set_title(f"HT at Z = {z_um:.1f} um",
                                               color="white", fontsize=10)
        self.ax_ht.tick_params(colors="white", labelsize=8)
        self._add_scale_bar(self.ax_ht, self.fov_x)

        # HT colorbar
        cbar_ht = self.fig.colorbar(self._im_ht, cax=self.ax_ht_cbar)
        cbar_ht.set_label("RI", color="white", fontsize=9)
        cbar_ht.ax.tick_params(colors="white", labelsize=8)

        if self.has_fl:
            assert self.fl_3d is not None

            # FL axes and colorbar
            self.ax_fl = self.fig.add_axes((0.37, 0.25, 0.25, 0.55), facecolor=self.DARK_FG)
            self.ax_fl_cbar = self.fig.add_axes((0.63, 0.25, 0.015, 0.55))

            fl_slice = self.fl_3d[self.current_z]
            self._im_fl = self.ax_fl.imshow(fl_slice, cmap="Greens", vmin=self.fl_vmin,
                                             vmax=self.fl_vmax, extent=extent)
            self.ax_fl.set_xlabel("X (um)", color="white", fontsize=9)
            self.ax_fl.set_ylabel("Y (um)", color="white", fontsize=9)
            self._title_fl = self.ax_fl.set_title("FL (registered)", color="white", fontsize=10)
            self.ax_fl.tick_params(colors="white", labelsize=8)
            self._add_scale_bar(self.ax_fl, self.fov_x)

            # FL colorbar
            cbar_fl = self.fig.colorbar(self._im_fl, cax=self.ax_fl_cbar)
            cbar_fl.set_label("Intensity", color="white", fontsize=9)
            cbar_fl.ax.tick_params(colors="white", labelsize=8)

            # Overlay axes
            self.ax_overlay = self.fig.add_axes((0.70, 0.25, 0.25, 0.55), facecolor=self.DARK_FG)

            ht_norm = normalize_with_bounds(ht_slice, self.ht_vmin, self.ht_vmax)
            fl_norm = normalize_with_bounds(fl_slice, self.fl_vmin, self.fl_vmax)
            rgb = np.zeros((*ht_norm.shape, 3), dtype=np.float32)
            rgb[:, :, 0] = ht_norm  # Red = HT
            rgb[:, :, 1] = fl_norm  # Green = FL
            self._im_overlay = self.ax_overlay.imshow(rgb, extent=extent)
            self.ax_overlay.set_xlabel("X (um)", color="white", fontsize=9)
            self.ax_overlay.set_ylabel("Y (um)", color="white", fontsize=9)
            self.ax_overlay.set_title("Overlay (R=HT, G=FL)", color="white", fontsize=10)
            self.ax_overlay.tick_params(colors="white", labelsize=8)
            self._add_scale_bar(self.ax_overlay, self.fov_x)

        # Z slider - show in micrometers
        z_um_max = (self.ht_3d.shape[0] - 1) * self.res_z
        ax_slider = self.fig.add_axes((0.15, 0.08, 0.7, 0.03), facecolor=self.DARK_FG)
        self.slider = Slider(
            ax_slider, "Z (um)", 0, z_um_max,
            valinit=z_um, color="#4a9eff"
        )
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")
        self.slider.on_changed(self._update_slice)

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
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Title with physical info
        self.fig.suptitle(
            f"{self.tcf_path.stem}  |  FOV: {self.fov_x:.0f} x {self.fov_y:.0f} x {self.fov_z:.0f} um",
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
        self._title_ht.set_text(f"HT at Z = {z_um_actual:.1f} um")

        if self.has_fl:
            assert self.fl_3d is not None
            fl_slice = self.fl_3d[self.current_z]
            self._im_fl.set_data(fl_slice)

            has_data = self.fl_z_start <= self.current_z < self.fl_z_end
            self._title_fl.set_text(
                f"FL {'(registered)' if has_data else '(no data)'} at Z = {z_um_actual:.1f} um"
            )

            # Update overlay
            ht_norm = normalize_with_bounds(ht_slice, self.ht_vmin, self.ht_vmax)
            fl_norm = normalize_with_bounds(fl_slice, self.fl_vmin, self.fl_vmax)
            rgb = np.zeros((*ht_norm.shape, 3), dtype=np.float32)
            rgb[:, :, 0] = ht_norm
            rgb[:, :, 1] = fl_norm
            self._im_overlay.set_data(rgb)

        self.fig.canvas.draw_idle()

    def _prev_slice(self, event: object = None) -> None:
        """Go to previous slice."""
        if self.current_z > 0:
            self.slider.set_val((self.current_z - 1) * self.res_z)

    def _next_slice(self, event: object = None) -> None:
        """Go to next slice."""
        if self.current_z < self.ht_3d.shape[0] - 1:
            self.slider.set_val((self.current_z + 1) * self.res_z)

    def _on_key(self, event: object) -> None:
        """Handle keyboard events."""
        key = getattr(event, "key", "")

        if key in ("left", "down"):
            self._prev_slice()
        elif key in ("right", "up"):
            self._next_slice()
        elif key == "home":
            self.slider.set_val(0)
        elif key == "end":
            self.slider.set_val((self.ht_3d.shape[0] - 1) * self.res_z)
        elif key in ("q", "escape"):
            plt.close(self.fig)

    def show(self) -> None:
        """Display the viewer."""
        plt.show()


def main() -> None:
    """Run slice viewer."""
    if len(sys.argv) > 1:
        tcf_path = sys.argv[1]
    else:
        # Try to find test files in data directory
        from pathlib import Path
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        if data_dir.exists():
            tcf_files = list(data_dir.rglob("*.TCF"))
            # Prefer S008 for good FL data
            s008_files = [f for f in tcf_files if "S008" in f.name]
            if s008_files:
                tcf_path = str(s008_files[0])
            elif tcf_files:
                tcf_path = str(tcf_files[0])
            else:
                print("Error: No TCF files found.")
                print(f"Usage: python -m tomocube slice <path_to_file.TCF>")
                sys.exit(1)
        else:
            print("Error: No TCF files found.")
            print(f"Usage: python -m tomocube slice <path_to_file.TCF>")
            sys.exit(1)

    viewer = SliceViewer(tcf_path)
    viewer.show()


if __name__ == "__main__":
    main()
