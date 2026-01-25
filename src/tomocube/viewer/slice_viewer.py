"""
Interactive Slice Viewer for TCF Files.

Compare HT and FL slices side-by-side with overlay.

Usage:
    python -m tomocube slice                    # Use default test file
    python -m tomocube slice path/to/file.TCF   # View specific file

Controls:
    - Slider or arrow keys: Navigate Z slices
    - Home/End: Jump to first/last slice
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from tomocube.core.file import load_registration_params
from tomocube.processing.image import normalize_with_bounds
from tomocube.processing.registration import register_fl_to_ht


class SliceViewer:
    """Interactive viewer for comparing HT and FL slices."""

    def __init__(self, tcf_path: str | Path) -> None:
        self.tcf_path = Path(tcf_path)
        self._load_data()
        self._setup_figure()

    def _load_data(self) -> None:
        """Load HT and FL data from TCF file."""
        print(f"Loading: {self.tcf_path.name}")

        with h5py.File(self.tcf_path, "r") as f:
            # Load HT
            self.ht_3d: np.ndarray = np.asarray(f["Data/3D/000000"]).astype(float)
            print(f"  HT shape: {self.ht_3d.shape}")

            # Load FL if available
            self.has_fl = "Data/3DFL/CH0/000000" in f

            if self.has_fl:
                fl_raw: np.ndarray = np.asarray(f["Data/3DFL/CH0/000000"]).astype(float)
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

    def _setup_figure(self) -> None:
        """Setup the matplotlib figure with sliders."""
        ncols = 3 if self.has_fl else 1
        self.fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        self.axes = [axes] if ncols == 1 else list(axes)

        plt.subplots_adjust(bottom=0.2)

        # Initial slice
        self.current_z = self.ht_3d.shape[0] // 2

        # Create HT image
        ht_slice = normalize_with_bounds(self.ht_3d[self.current_z], self.ht_vmin, self.ht_vmax)
        self.im_ht = self.axes[0].imshow(ht_slice, cmap="gray", vmin=0, vmax=1)
        self.axes[0].set_title(f"HT (z={self.current_z})")
        self.axes[0].axis("off")

        if self.has_fl:
            # FL image
            assert self.fl_3d is not None  # guaranteed when has_fl is True
            fl_slice = normalize_with_bounds(
                self.fl_3d[self.current_z], self.fl_vmin, self.fl_vmax
            )
            self.im_fl = self.axes[1].imshow(fl_slice, cmap="Greens", vmin=0, vmax=1)
            self.axes[1].set_title("FL (registered)")
            self.axes[1].axis("off")

            # Overlay
            rgb = np.zeros((*ht_slice.shape, 3))
            rgb[:, :, 0] = ht_slice  # Red = HT
            rgb[:, :, 1] = fl_slice  # Green = FL
            self.im_overlay = self.axes[2].imshow(rgb)
            self.axes[2].set_title("Overlay (R=HT, G=FL)")
            self.axes[2].axis("off")

        # Z slider
        ax_slider = plt.axes((0.2, 0.05, 0.6, 0.03))
        self.slider = Slider(
            ax_slider, "Z Slice", 0, self.ht_3d.shape[0] - 1,
            valinit=self.current_z, valstep=1
        )
        self.slider.on_changed(self._update_slice)

        # Navigation buttons
        ax_prev = plt.axes((0.2, 0.1, 0.1, 0.04))
        ax_next = plt.axes((0.7, 0.1, 0.1, 0.04))
        self.btn_prev = Button(ax_prev, "< Prev")
        self.btn_next = Button(ax_next, "Next >")
        self.btn_prev.on_clicked(self._prev_slice)
        self.btn_next.on_clicked(self._next_slice)

        # Keyboard navigation
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.fig.suptitle(self.tcf_path.stem, fontsize=10)

    def _update_slice(self, z: float) -> None:
        """Update displayed slices."""
        self.current_z = int(z)

        ht_slice = normalize_with_bounds(self.ht_3d[self.current_z], self.ht_vmin, self.ht_vmax)
        self.im_ht.set_data(ht_slice)
        self.axes[0].set_title(f"HT (z={self.current_z})")

        if self.has_fl:
            assert self.fl_3d is not None  # guaranteed when has_fl is True
            fl_slice = normalize_with_bounds(
                self.fl_3d[self.current_z], self.fl_vmin, self.fl_vmax
            )
            self.im_fl.set_data(fl_slice)

            has_data = self.fl_z_start <= self.current_z < self.fl_z_end
            self.axes[1].set_title(f"FL {'(registered)' if has_data else '(no data)'}")

            # Update overlay
            rgb = np.zeros((*ht_slice.shape, 3))
            rgb[:, :, 0] = ht_slice
            rgb[:, :, 1] = fl_slice
            self.im_overlay.set_data(rgb)

        self.fig.canvas.draw_idle()

    def _prev_slice(self, event: object = None) -> None:
        """Go to previous slice."""
        if self.current_z > 0:
            self.slider.set_val(self.current_z - 1)

    def _next_slice(self, event: object = None) -> None:
        """Go to next slice."""
        if self.current_z < self.ht_3d.shape[0] - 1:
            self.slider.set_val(self.current_z + 1)

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
            self.slider.set_val(self.ht_3d.shape[0] - 1)

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
