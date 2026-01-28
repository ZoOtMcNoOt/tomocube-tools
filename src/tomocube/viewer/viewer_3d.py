"""
3D Volume Viewer for Tomocube TCF files.

Provides interactive 3D visualization using napari with:
- Volume rendering (MIP, attenuated, etc.)
- Multi-channel fluorescence overlay
- XYZ range sliders for sub-volume cropping
- Layer controls with background removal
- Camera presets for different viewing angles
- Scale bar with physical units
- Screenshot and animation export (GIF/MP4)
- Performance optimization for large volumes (512^3+)

Usage:
    python -m tomocube view3d sample.TCF
    python -m tomocube view3d sample.TCF --slices

Keyboard shortcuts:
    1-6     Camera presets (Top, Bottom, Front, Back, Left, Right)
    0       Isometric view
    R       Reset camera
    F       Fit view to data
    +/-     Zoom in/out
    T       Start turntable animation export
    2/3     Toggle 2D/3D view
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from tomocube.core.config import vprint, is_verbose

if TYPE_CHECKING:
    from tomocube.core.file import TCFFileLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Animation Export
# =============================================================================

class AnimationExporter:
    """
    Export turntable and slice sweep animations as GIF or MP4.

    IMPORTANT: All methods must be called from the main Qt thread since
    napari's screenshot() requires OpenGL context on the main thread.
    Use capture_frame() with QTimer for animation loops.
    """

    def __init__(self, viewer, output_dir: Path):
        self.viewer = viewer
        self.output_dir = output_dir
        self._is_exporting = False
        self._frames: list = []
        self._export_config: dict = {}
        self._original_state: dict = {}

    def start_turntable_export(
        self,
        filename: str,
        n_frames: int,
        duration_ms: int,
    ) -> None:
        """
        Initialize turntable export state. Call capture_turntable_frame() repeatedly.
        Must be called from main thread.
        """
        self._is_exporting = True
        self._frames = []
        self._export_config = {
            "mode": "turntable",
            "filename": filename,
            "n_frames": n_frames,
            "duration_ms": duration_ms,
            "current_frame": 0,
        }
        self._original_state = {
            "angles": self.viewer.camera.angles,
        }
        # Ensure 3D mode
        self.viewer.dims.ndisplay = 3

    def capture_turntable_frame(self) -> tuple[int, int, bool]:
        """
        Capture one frame of turntable animation. Returns (current, total, done).
        Must be called from main thread.
        """
        from qtpy.QtWidgets import QApplication

        cfg = self._export_config
        current = cfg["current_frame"]
        n_frames = cfg["n_frames"]

        if current >= n_frames:
            return (current, n_frames, True)

        # Calculate rotation angle
        roll, pitch, yaw = self._original_state["angles"]
        angle = yaw + (360 * current / n_frames)
        self.viewer.camera.angles = (roll, pitch, angle)

        # Process events to ensure render completes
        QApplication.processEvents()

        # Capture frame
        frame = self.viewer.screenshot(canvas_only=True)
        self._frames.append(frame)

        cfg["current_frame"] = current + 1
        done = cfg["current_frame"] >= n_frames

        return (current + 1, n_frames, done)

    def start_slice_sweep_export(
        self,
        filename: str,
        axis: int,
        duration_ms: int,
    ) -> int:
        """
        Initialize slice sweep export. Returns total number of slices.
        Must be called from main thread.
        
        Args:
            axis: 0=Z, 1=Y, 2=X (napari dim order for 3D volume)
        """
        self._is_exporting = True
        self._frames = []

        n_slices = int(self.viewer.dims.range[axis][1])

        self._export_config = {
            "mode": "sweep",
            "filename": filename,
            "axis": axis,
            "n_slices": n_slices,
            "duration_ms": duration_ms,
            "current_frame": 0,
        }
        self._original_state = {
            "ndisplay": self.viewer.dims.ndisplay,
            "point": list(self.viewer.dims.point),
            "order": list(self.viewer.dims.order),
        }
        
        # Switch to 2D mode and set the correct axis to be the sliced dimension
        # In napari 2D mode, the first axis in `order` is the one being sliced
        # Default order is (0, 1, 2) meaning Z is sliced. 
        # For Y sweep, order should be (1, 0, 2) - Y is sliced, display ZX
        # For X sweep, order should be (2, 0, 1) - X is sliced, display ZY
        if axis == 0:  # Z sweep
            new_order = (0, 1, 2)
        elif axis == 1:  # Y sweep
            new_order = (1, 0, 2)
        else:  # X sweep (axis == 2)
            new_order = (2, 0, 1)
        
        self.viewer.dims.order = new_order
        self.viewer.dims.ndisplay = 2

        return n_slices

    def capture_sweep_frame(self) -> tuple[int, int, bool]:
        """
        Capture one frame of slice sweep animation. Returns (current, total, done).
        Must be called from main thread.
        """
        from qtpy.QtWidgets import QApplication

        cfg = self._export_config
        current = cfg["current_frame"]
        n_slices = cfg["n_slices"]
        axis = cfg["axis"]

        if current >= n_slices:
            return (current, n_slices, True)

        # Set slice position
        self.viewer.dims.set_point(axis, current)

        # Process events to ensure render completes
        QApplication.processEvents()

        # Capture frame
        frame = self.viewer.screenshot(canvas_only=True)
        self._frames.append(frame)

        cfg["current_frame"] = current + 1
        done = cfg["current_frame"] >= n_slices

        return (current + 1, n_slices, done)

    def finish_export(self) -> Path:
        """
        Save captured frames to file and restore viewer state.
        Must be called from main thread.
        """
        import imageio.v3 as iio

        cfg = self._export_config
        filename = cfg["filename"]
        duration_ms = cfg["duration_ms"]

        # Save animation
        output_path = self.output_dir / filename

        # Ensure all frames have the same shape (window might resize during capture)
        if self._frames:
            # First, normalize all frames to RGB (3 channels)
            normalized_frames = []
            for frame in self._frames:
                if frame.ndim == 3 and frame.shape[2] == 4:
                    # Convert RGBA to RGB
                    frame = frame[:, :, :3].copy()
                elif frame.ndim == 2:
                    # Grayscale to RGB
                    frame = np.stack([frame, frame, frame], axis=2)
                else:
                    frame = frame.copy()
                normalized_frames.append(frame)
            
            # Find minimum dimensions across all frames
            min_h = min(f.shape[0] for f in normalized_frames)
            min_w = min(f.shape[1] for f in normalized_frames)
            
            # Ensure dimensions are divisible by 16 (required by h264 macro blocks)
            min_h = (min_h // 16) * 16
            min_w = (min_w // 16) * 16
            
            # Crop all frames to minimum dimensions
            final_frames = []
            for frame in normalized_frames:
                cropped = frame[:min_h, :min_w, :3]
                final_frames.append(cropped)
            
            # Stack into single array for imageio
            self._frames = np.stack(final_frames, axis=0)

        if filename.endswith('.mp4'):
            # Use imageio-ffmpeg for MP4 export
            fps = max(1, 1000 // duration_ms)
            try:
                iio.imwrite(
                    str(output_path),
                    self._frames,
                    fps=fps,
                    plugin="FFMPEG",
                )
            except Exception as e:
                # Fallback to GIF if ffmpeg fails
                print(f"Warning: MP4 encoding failed ({e})")
                print("Saving as GIF instead...")
                output_path = output_path.with_suffix('.gif')
                cfg["filename"] = output_path.name
                duration_sec = duration_ms / 1000.0
                iio.imwrite(
                    str(output_path),
                    self._frames,
                    duration=duration_sec,
                    loop=0,
                    plugin="pillow",
                )
        else:
            # GIF export
            duration_sec = duration_ms / 1000.0
            iio.imwrite(
                str(output_path),
                self._frames,
                duration=duration_sec,
                loop=0,
                plugin="pillow",
            )

        # Restore state
        if cfg["mode"] == "turntable":
            self.viewer.camera.angles = self._original_state["angles"]
        else:
            # Restore dims order first, then ndisplay and points
            if "order" in self._original_state:
                self.viewer.dims.order = tuple(self._original_state["order"])
            self.viewer.dims.ndisplay = self._original_state["ndisplay"]
            for ax, pt in enumerate(self._original_state["point"]):
                self.viewer.dims.set_point(ax, pt)

        self._is_exporting = False
        self._frames = []

        return output_path

    def cancel_export(self) -> None:
        """Cancel export and restore viewer state."""
        if not self._is_exporting:
            return

        cfg = self._export_config
        if cfg.get("mode") == "turntable":
            self.viewer.camera.angles = self._original_state.get("angles", (0, -30, 45))
        elif cfg.get("mode") == "sweep":
            if "order" in self._original_state:
                self.viewer.dims.order = tuple(self._original_state["order"])
            self.viewer.dims.ndisplay = self._original_state.get("ndisplay", 3)
            for ax, pt in enumerate(self._original_state.get("point", [])):
                self.viewer.dims.set_point(ax, pt)

        self._is_exporting = False
        self._frames = []


def _get_voxel_scale(loader: TCFFileLoader) -> tuple[float, float, float]:
    """Extract voxel scale from TCF metadata. Returns (z, y, x) in µm."""
    try:
        # Use registration params which contain the actual HT resolution
        if hasattr(loader, 'reg_params') and loader.reg_params is not None:
            params = loader.reg_params
            return (
                float(params.ht_res_z),
                float(params.ht_res_y),
                float(params.ht_res_x)
            )
    except Exception:
        pass
    # Fallback to Tomocube HT-2H defaults (from constants.py)
    from tomocube.core.constants import DEFAULT_HT_RES_X, DEFAULT_HT_RES_Y, DEFAULT_HT_RES_Z
    return (DEFAULT_HT_RES_Z, DEFAULT_HT_RES_Y, DEFAULT_HT_RES_X)


def _create_layer_controls(viewer):
    """Create layer controls with opacity, threshold, colormap, and auto-contrast."""
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QSlider, QComboBox, QCheckBox, QGroupBox, QScrollArea, QSizePolicy
    )
    from qtpy.QtCore import Qt
    
    # Available colormaps
    COLORMAPS = ["gray", "viridis", "plasma", "magma", "inferno", "turbo",
                 "green", "magenta", "cyan", "yellow", "red", "blue"]
    
    class LayerControl(QWidget):
        """Controls for a single layer."""
        def __init__(self, layer, parent=None):
            super().__init__(parent)
            self.layer = layer
            self.original_data = layer.data.copy() if hasattr(layer, 'data') else None
            self.original_contrast = layer.contrast_limits if hasattr(layer, 'contrast_limits') else None
            self.original_colormap = str(layer.colormap.name) if hasattr(layer, 'colormap') else "gray"
            self.original_opacity = layer.opacity
            self.original_blending = layer.blending
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(2, 2, 2, 4)
            layout.setSpacing(3)
            
            # Layer name and visibility
            header = QHBoxLayout()
            self.visible_cb = QCheckBox(layer.name)
            self.visible_cb.setChecked(layer.visible)
            self.visible_cb.setStyleSheet("font-weight: bold;")
            self.visible_cb.toggled.connect(lambda v: setattr(layer, 'visible', v))
            header.addWidget(self.visible_cb)
            header.addStretch()
            layout.addLayout(header)
            
            # Colormap selector
            cmap_row = QHBoxLayout()
            cmap_lbl = QLabel("Color")
            cmap_lbl.setFixedWidth(55)
            cmap_row.addWidget(cmap_lbl)
            self.cmap_combo = QComboBox()
            self.cmap_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.cmap_combo.addItems(COLORMAPS)
            current_cmap = str(layer.colormap.name) if hasattr(layer, 'colormap') else "gray"
            if current_cmap in COLORMAPS:
                self.cmap_combo.setCurrentText(current_cmap)
            self.cmap_combo.currentTextChanged.connect(
                lambda v: setattr(layer, 'colormap', v)
            )
            cmap_row.addWidget(self.cmap_combo)
            layout.addLayout(cmap_row)
            
            # Opacity slider
            opacity_row = QHBoxLayout()
            opacity_lbl = QLabel("Opacity")
            opacity_lbl.setFixedWidth(55)
            opacity_row.addWidget(opacity_lbl)
            self.opacity_slider = QSlider(Qt.Horizontal)
            self.opacity_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.opacity_slider.setRange(0, 100)
            self.opacity_slider.setValue(int(layer.opacity * 100))
            self.opacity_slider.valueChanged.connect(
                lambda v: setattr(layer, 'opacity', v / 100)
            )
            opacity_row.addWidget(self.opacity_slider)
            self.opacity_label = QLabel(f"{int(layer.opacity * 100)}%")
            self.opacity_label.setFixedWidth(35)
            self.opacity_slider.valueChanged.connect(
                lambda v: self.opacity_label.setText(f"{v}%")
            )
            opacity_row.addWidget(self.opacity_label)
            layout.addLayout(opacity_row)
            
            # Background threshold (removes values below threshold)
            if hasattr(layer, 'contrast_limits'):
                thresh_row = QHBoxLayout()
                thresh_lbl = QLabel("Threshold")
                thresh_lbl.setFixedWidth(55)
                thresh_row.addWidget(thresh_lbl)
                self.thresh_slider = QSlider(Qt.Horizontal)
                self.thresh_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                self.thresh_slider.setRange(0, 100)
                self.thresh_slider.setValue(0)
                self.thresh_slider.setToolTip("Remove background below this percentile")
                self.thresh_slider.valueChanged.connect(self._apply_threshold)
                thresh_row.addWidget(self.thresh_slider)
                self.thresh_label = QLabel("0%")
                self.thresh_label.setFixedWidth(35)
                thresh_row.addWidget(self.thresh_label)
                layout.addLayout(thresh_row)
            
            # Blending mode
            blend_row = QHBoxLayout()
            blend_lbl = QLabel("Blend")
            blend_lbl.setFixedWidth(55)
            blend_row.addWidget(blend_lbl)
            self.blend_combo = QComboBox()
            self.blend_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.blend_combo.addItems(["translucent", "additive", "minimum", "opaque"])
            self.blend_combo.setCurrentText(layer.blending)
            self.blend_combo.currentTextChanged.connect(
                lambda v: setattr(layer, 'blending', v)
            )
            blend_row.addWidget(self.blend_combo)
            layout.addLayout(blend_row)
        
        def _apply_threshold(self, value):
            """Apply background threshold by adjusting contrast limits."""
            self.thresh_label.setText(f"{value}%")
            if self.original_data is not None and self.original_contrast is not None:
                if value == 0:
                    self.layer.contrast_limits = self.original_contrast
                else:
                    data = self.original_data
                    low = np.percentile(data, value)
                    high = self.original_contrast[1]
                    self.layer.contrast_limits = (low, high)
        
        def auto_contrast(self):
            """Apply auto contrast based on data percentiles."""
            if hasattr(self.layer, 'data') and hasattr(self.layer, 'contrast_limits'):
                data = self.layer.data
                if data.size > 0:
                    p1, p99 = np.percentile(data, [1, 99])
                    self.layer.contrast_limits = (p1, p99)
                    self.original_contrast = (p1, p99)
        
        def reset(self):
            """Reset layer to original settings."""
            # Reset opacity
            self.layer.opacity = self.original_opacity
            self.opacity_slider.setValue(int(self.layer.opacity * 100))
            
            # Reset threshold
            if hasattr(self, 'thresh_slider'):
                self.thresh_slider.setValue(0)
            
            # Reset contrast
            if self.original_contrast:
                self.layer.contrast_limits = self.original_contrast
            
            # Reset colormap
            self.layer.colormap = self.original_colormap
            if self.original_colormap in COLORMAPS:
                self.cmap_combo.setCurrentText(self.original_colormap)
            
            # Reset blending
            self.layer.blending = self.original_blending
            self.blend_combo.setCurrentText(self.layer.blending)
    
    class LayerControlsWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.layer_controls = []
            
            main_layout = QVBoxLayout(self)
            main_layout.setSpacing(4)
            main_layout.setContentsMargins(4, 4, 4, 4)
            
            # Title
            title = QLabel("<b>Layers</b>")
            title.setStyleSheet("font-size: 12px;")
            main_layout.addWidget(title)

            # Global buttons
            btn_row = QHBoxLayout()
            
            auto_btn = QPushButton("Auto Contrast")
            auto_btn.setToolTip("Recalculate contrast for all layers")
            auto_btn.clicked.connect(self._auto_all)
            btn_row.addWidget(auto_btn)
            
            reset_btn = QPushButton("Reset All")
            reset_btn.setToolTip("Reset all layers to defaults")
            reset_btn.clicked.connect(self._reset_all)
            btn_row.addWidget(reset_btn)
            
            main_layout.addLayout(btn_row)
            
            main_layout.addSpacing(4)
            
            # Scrollable area for layer controls
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            
            scroll_content = QWidget()
            self.layers_layout = QVBoxLayout(scroll_content)
            self.layers_layout.setSpacing(4)
            self.layers_layout.setContentsMargins(0, 0, 0, 0)
            
            # Add controls for each image layer
            for layer in viewer.layers:
                if hasattr(layer, 'data') and layer.data.ndim == 3:
                    ctrl = LayerControl(layer)
                    self.layer_controls.append(ctrl)
                    
                    group = QGroupBox()
                    group_layout = QVBoxLayout(group)
                    group_layout.setContentsMargins(4, 4, 4, 4)
                    group_layout.addWidget(ctrl)
                    self.layers_layout.addWidget(group)
            
            self.layers_layout.addStretch()
            scroll.setWidget(scroll_content)
            main_layout.addWidget(scroll)
        
        def _auto_all(self):
            for ctrl in self.layer_controls:
                ctrl.auto_contrast()
        
        def _reset_all(self):
            for ctrl in self.layer_controls:
                ctrl.reset()
    
    widget = LayerControlsWidget()
    dock = viewer.window.add_dock_widget(widget, name="Layers", area="right")
    return dock


def _create_histogram_widget(viewer):
    """Create histogram widget for visualizing and adjusting layer contrast."""
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QComboBox, QSizePolicy, QGroupBox
    )
    from qtpy.QtCore import Qt
    
    try:
        import pyqtgraph as pg
        HAS_PYQTGRAPH = True
    except ImportError:
        HAS_PYQTGRAPH = False
    
    class HistogramWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.current_layer = None
            
            layout = QVBoxLayout(self)
            layout.setSpacing(4)
            layout.setContentsMargins(4, 4, 4, 4)
            
            # Title
            title = QLabel("<b>Histogram</b>")
            title.setStyleSheet("font-size: 13px;")
            layout.addWidget(title)
            
            desc = QLabel("View intensity distribution and adjust contrast.")
            desc.setStyleSheet("color: #888; font-size: 10px;")
            desc.setWordWrap(True)
            layout.addWidget(desc)
            
            # Layer selector
            layer_row = QHBoxLayout()
            layer_lbl = QLabel("Layer:")
            layer_lbl.setFixedWidth(40)
            layer_row.addWidget(layer_lbl)
            self.layer_combo = QComboBox()
            self.layer_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            for layer in viewer.layers:
                if hasattr(layer, 'data') and layer.data.ndim == 3:
                    self.layer_combo.addItem(layer.name)
            self.layer_combo.currentTextChanged.connect(self._on_layer_change)
            layer_row.addWidget(self.layer_combo)
            layout.addLayout(layer_row)
            
            layout.addSpacing(4)
            
            if HAS_PYQTGRAPH:
                # Create pyqtgraph histogram widget
                self.hist_widget = pg.PlotWidget()
                self.hist_widget.setMinimumHeight(150)
                self.hist_widget.setMaximumHeight(200)
                self.hist_widget.setBackground('w')
                self.hist_widget.showGrid(x=True, y=True, alpha=0.3)
                self.hist_widget.setLabel('bottom', 'Intensity')
                self.hist_widget.setLabel('left', 'Count')
                self.hist_plot = self.hist_widget.plot(pen=pg.mkPen('b', width=1), fillLevel=0, brush=(100, 100, 255, 80))
                
                # Vertical lines for contrast limits
                self.low_line = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen('r', width=2))
                self.high_line = pg.InfiniteLine(pos=1, angle=90, movable=True, pen=pg.mkPen('g', width=2))
                self.low_line.sigPositionChanged.connect(self._on_limit_change)
                self.high_line.sigPositionChanged.connect(self._on_limit_change)
                self.hist_widget.addItem(self.low_line)
                self.hist_widget.addItem(self.high_line)
                
                layout.addWidget(self.hist_widget)
            else:
                # Fallback: text-based stats
                self.stats_label = QLabel("Install pyqtgraph for histogram view:\npip install pyqtgraph")
                self.stats_label.setStyleSheet("color: #888; font-family: monospace;")
                self.stats_label.setWordWrap(True)
                layout.addWidget(self.stats_label)
            
            # Stats display
            self.info_label = QLabel("")
            self.info_label.setStyleSheet("font-family: monospace; font-size: 10px;")
            self.info_label.setWordWrap(True)
            layout.addWidget(self.info_label)
            
            # Contrast limit display
            limit_row = QHBoxLayout()
            limit_lbl = QLabel("Limits:")
            limit_lbl.setFixedWidth(40)
            limit_row.addWidget(limit_lbl)
            self.limit_label = QLabel("-- to --")
            self.limit_label.setStyleSheet("font-family: monospace;")
            limit_row.addWidget(self.limit_label)
            limit_row.addStretch()
            layout.addLayout(limit_row)
            
            # Preset buttons
            preset_row = QHBoxLayout()
            
            auto_btn = QPushButton("Auto (1-99%)")
            auto_btn.setToolTip("Set limits to 1st-99th percentile")
            auto_btn.clicked.connect(lambda: self._apply_percentile(1, 99))
            preset_row.addWidget(auto_btn)
            
            wide_btn = QPushButton("Wide (0.1-99.9%)")
            wide_btn.setToolTip("Set limits to 0.1-99.9 percentile")
            wide_btn.clicked.connect(lambda: self._apply_percentile(0.1, 99.9))
            preset_row.addWidget(wide_btn)
            
            layout.addLayout(preset_row)
            
            preset_row2 = QHBoxLayout()
            
            boost_btn = QPushButton("Boost Weak (5-99.5%)")
            boost_btn.setToolTip("For weak FL: set lower bound higher to see signal")
            boost_btn.clicked.connect(lambda: self._apply_percentile(5, 99.5))
            preset_row2.addWidget(boost_btn)
            
            full_btn = QPushButton("Full Range")
            full_btn.setToolTip("Use full data range (min to max)")
            full_btn.clicked.connect(self._apply_full_range)
            preset_row2.addWidget(full_btn)
            
            layout.addLayout(preset_row2)
            
            layout.addStretch()
            
            # Initialize with first layer
            if self.layer_combo.count() > 0:
                self._on_layer_change(self.layer_combo.currentText())
        
        def _on_layer_change(self, layer_name: str):
            """Update histogram for selected layer."""
            for layer in viewer.layers:
                if layer.name == layer_name:
                    self.current_layer = layer
                    self._update_histogram()
                    break
        
        def _update_histogram(self):
            """Recalculate and display histogram."""
            if self.current_layer is None:
                return
            
            data = self.current_layer.data
            if data is None or data.size == 0:
                return
            
            # Flatten and sample for performance (max 1M points)
            flat = data.ravel()
            if len(flat) > 1_000_000:
                flat = np.random.choice(flat, 1_000_000, replace=False)
            
            # Calculate stats
            d_min, d_max = float(np.min(data)), float(np.max(data))
            d_mean = float(np.mean(flat))
            d_std = float(np.std(flat))
            nonzero = flat[flat > 0]
            nonzero_pct = len(nonzero) / len(flat) * 100 if len(flat) > 0 else 0
            
            # Percentiles
            p1, p50, p99 = np.percentile(flat, [1, 50, 99])
            
            self.info_label.setText(
                f"Min: {d_min:.2f}  Max: {d_max:.2f}\n"
                f"Mean: {d_mean:.2f}  Std: {d_std:.2f}\n"
                f"1%: {p1:.2f}  50%: {p50:.2f}  99%: {p99:.2f}\n"
                f"Non-zero: {nonzero_pct:.1f}%"
            )
            
            if HAS_PYQTGRAPH:
                # Calculate histogram
                # Use log-scale friendly binning for FL data
                if d_max > 0:
                    hist, bin_edges = np.histogram(flat, bins=200, range=(d_min, d_max))
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    # Use log scale for counts (add 1 to avoid log(0))
                    hist_log = np.log1p(hist)
                    self.hist_plot.setData(bin_centers, hist_log)
                    
                    # Update limit lines
                    if hasattr(self.current_layer, 'contrast_limits'):
                        low, high = self.current_layer.contrast_limits
                        self.low_line.blockSignals(True)
                        self.high_line.blockSignals(True)
                        self.low_line.setValue(low)
                        self.high_line.setValue(high)
                        self.low_line.blockSignals(False)
                        self.high_line.blockSignals(False)
                        self.limit_label.setText(f"{low:.2f} to {high:.2f}")
                    
                    # Set X range to data range
                    self.hist_widget.setXRange(d_min, d_max)
        
        def _on_limit_change(self):
            """Apply contrast limits from draggable lines."""
            if self.current_layer is None or not hasattr(self.current_layer, 'contrast_limits'):
                return
            
            low = self.low_line.value()
            high = self.high_line.value()
            
            # Ensure low < high
            if low >= high:
                return
            
            self.current_layer.contrast_limits = (low, high)
            self.limit_label.setText(f"{low:.2f} to {high:.2f}")
        
        def _apply_percentile(self, low_pct: float, high_pct: float):
            """Apply percentile-based contrast limits."""
            if self.current_layer is None or not hasattr(self.current_layer, 'contrast_limits'):
                return
            
            data = self.current_layer.data
            low, high = np.percentile(data, [low_pct, high_pct])
            self.current_layer.contrast_limits = (low, high)
            self._update_histogram()
        
        def _apply_full_range(self):
            """Apply full data range as contrast limits."""
            if self.current_layer is None or not hasattr(self.current_layer, 'contrast_limits'):
                return
            
            data = self.current_layer.data
            self.current_layer.contrast_limits = (float(np.min(data)), float(np.max(data)))
            self._update_histogram()
    
    widget = HistogramWidget()
    dock = viewer.window.add_dock_widget(widget, name="Histogram", area="right")
    return dock


def _create_camera_controls(viewer):
    """Create camera preset controls with keyboard shortcuts."""
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout, QSizePolicy
    )
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QKeySequence
    from qtpy.QtWidgets import QShortcut

    # Camera presets: (name, angles, key)
    CAMERA_PRESETS = [
        ("Top", (0, 0, 90), "1"),
        ("Bottom", (0, 180, 90), "2"),
        ("Front", (0, -90, 0), "3"),
        ("Back", (0, 90, 0), "4"),
        ("Left", (90, -90, 0), "5"),
        ("Right", (-90, -90, 0), "6"),
    ]

    class CameraWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.default_zoom = 0.8
            self.default_angles = (0, -30, 45)

            layout = QVBoxLayout(self)
            layout.setSpacing(4)
            layout.setContentsMargins(4, 4, 4, 4)

            title = QLabel("<b>Camera</b>")
            title.setStyleSheet("font-size: 12px;")
            layout.addWidget(title)

            # View presets in grid
            grid = QGridLayout()
            grid.setSpacing(4)

            for i, (name, angles, key) in enumerate(CAMERA_PRESETS):
                btn = QPushButton(f"{name} [{key}]")
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                btn.setToolTip(f"View from {name.lower()} (press {key})")
                btn.clicked.connect(lambda checked, a=angles: self._set_view_animated(a))
                grid.addWidget(btn, i // 3, i % 3)

            layout.addLayout(grid)

            # Isometric and reset
            iso_row = QHBoxLayout()

            iso_btn = QPushButton("Isometric [0]")
            iso_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            iso_btn.setToolTip("45-degree isometric view (press 0)")
            iso_btn.clicked.connect(lambda: self._set_view_animated((0, -30, 45)))
            iso_row.addWidget(iso_btn)

            reset_btn = QPushButton("Reset [R]")
            reset_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            reset_btn.setToolTip("Reset camera to default (press R)")
            reset_btn.clicked.connect(self._reset_camera)
            iso_row.addWidget(reset_btn)

            layout.addLayout(iso_row)

            # Zoom controls
            zoom_row = QHBoxLayout()
            zoom_lbl = QLabel("Zoom")
            zoom_lbl.setFixedWidth(40)
            zoom_row.addWidget(zoom_lbl)

            zoom_out = QPushButton("- [-]")
            zoom_out.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            zoom_out.clicked.connect(lambda: self._zoom(0.8))
            zoom_row.addWidget(zoom_out)

            zoom_in = QPushButton("+ [=]")
            zoom_in.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            zoom_in.clicked.connect(lambda: self._zoom(1.25))
            zoom_row.addWidget(zoom_in)

            zoom_fit = QPushButton("Fit [F]")
            zoom_fit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            zoom_fit.clicked.connect(lambda: viewer.reset_view())
            zoom_row.addWidget(zoom_fit)

            layout.addLayout(zoom_row)

            layout.addStretch()

            # Register keyboard shortcuts
            self._setup_shortcuts()

        def _setup_shortcuts(self):
            """Register keyboard shortcuts for camera control."""
            window = viewer.window._qt_window

            # Camera presets 1-6
            for name, angles, key in CAMERA_PRESETS:
                shortcut = QShortcut(QKeySequence(key), window)
                shortcut.activated.connect(lambda a=angles: self._set_view_animated(a))

            # Isometric (0)
            iso_shortcut = QShortcut(QKeySequence("0"), window)
            iso_shortcut.activated.connect(lambda: self._set_view_animated((0, -30, 45)))

            # Reset (R)
            reset_shortcut = QShortcut(QKeySequence("R"), window)
            reset_shortcut.activated.connect(self._reset_camera)

            # Fit (F)
            fit_shortcut = QShortcut(QKeySequence("F"), window)
            fit_shortcut.activated.connect(lambda: viewer.reset_view())

            # Zoom in/out
            zoom_in = QShortcut(QKeySequence("="), window)
            zoom_in.activated.connect(lambda: self._zoom(1.25))
            zoom_in2 = QShortcut(QKeySequence("+"), window)
            zoom_in2.activated.connect(lambda: self._zoom(1.25))

            zoom_out = QShortcut(QKeySequence("-"), window)
            zoom_out.activated.connect(lambda: self._zoom(0.8))

        def _set_view_animated(self, target_angles, steps: int = 10):
            """Animate camera transition to target angles."""
            from qtpy.QtWidgets import QApplication

            viewer.dims.ndisplay = 3

            # Get current angles
            current = viewer.camera.angles
            target = target_angles

            # Simple linear interpolation over steps
            def lerp(a, b, t):
                return a + (b - a) * t

            for i in range(1, steps + 1):
                t = i / steps
                new_angles = tuple(lerp(current[j], target[j], t) for j in range(3))
                viewer.camera.angles = new_angles
                QApplication.processEvents()  # Process Qt events to update display
                time.sleep(0.02)  # ~50fps animation

        def _set_view(self, angles):
            viewer.dims.ndisplay = 3
            viewer.camera.angles = angles

        def _zoom(self, factor):
            viewer.camera.zoom *= factor

        def _reset_camera(self):
            viewer.dims.ndisplay = 3
            self._set_view_animated(self.default_angles)
            viewer.camera.zoom = self.default_zoom
            viewer.camera.center = (0, 0, 0)

    widget = CameraWidget()
    dock = viewer.window.add_dock_widget(widget, name="Camera", area="left")
    return dock


def _create_crop_widget(viewer, ht_data: np.ndarray, scale: tuple):
    """Create a docked widget with XYZ range sliders for cropping."""
    from superqt import QRangeSlider
    from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy
    from qtpy.QtCore import Qt
    
    class AxisRangeSlider(QWidget):
        """Single range slider with two handles for one axis."""
        def __init__(self, label: str, max_val: int, unit_scale: float, on_change, parent=None):
            super().__init__(parent)
            self.unit_scale = unit_scale
            self.max_val = max_val
            self.on_change = on_change
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 4, 0, 4)
            layout.setSpacing(2)
            
            # Header with label and range display
            header = QHBoxLayout()
            self.axis_label = QLabel(f"<b>{label}</b>")
            self.range_label = QLabel(f"0 - {max_val}")
            self.range_label.setStyleSheet("font-family: monospace;")
            header.addWidget(self.axis_label)
            header.addStretch()
            header.addWidget(self.range_label)
            layout.addLayout(header)
            
            # Range slider (single slider with 2 handles)
            self.slider = QRangeSlider(Qt.Horizontal)
            self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.slider.setRange(0, max_val)
            self.slider.setValue((0, max_val))
            self.slider.valueChanged.connect(self._on_changed)
            layout.addWidget(self.slider)
            
            # Size in um
            self.size_label = QLabel(f"{max_val * unit_scale:.1f} um")
            self.size_label.setStyleSheet("color: #888; font-size: 10px;")
            layout.addWidget(self.size_label)
        
        def _on_changed(self, value):
            min_v, max_v = value
            self.range_label.setText(f"{min_v} - {max_v}")
            size = (max_v - min_v) * self.unit_scale
            self.size_label.setText(f"{size:.1f} um")
            self.on_change()  # Live update
        
        def get_range(self) -> tuple[int, int]:
            return self.slider.value()
        
        def reset(self):
            self.slider.setValue((0, self.max_val))
    
    class CropWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.full_data = {}  # {layer_name: data}
            self.layer_info = {}  # {layer_name: {'scale': tuple, 'translate': tuple, 'is_fl': bool}}
            
            layout = QVBoxLayout(self)
            layout.setSpacing(4)
            layout.setContentsMargins(4, 4, 4, 4)
            
            # Title
            title = QLabel("<b>Volume Crop</b>")
            title.setStyleSheet("font-size: 12px;")
            layout.addWidget(title)
            
            z, y, x = ht_data.shape
            
            # Store all layers with their original data, scale, and translate
            for layer in viewer.layers:
                if hasattr(layer, 'data') and isinstance(layer.data, np.ndarray) and layer.data.ndim == 3:
                    self.full_data[layer.name] = layer.data.copy()
                    self.layer_info[layer.name] = {
                        'scale': tuple(layer.scale),
                        'translate': tuple(layer.translate),
                        'is_fl': layer.name != "RI"  # FL channels are anything that's not RI
                    }
            
            # Create range sliders (based on HT/RI dimensions)
            self.z_slider = AxisRangeSlider("Z (depth)", z - 1, scale[0], self._apply_crop)
            self.y_slider = AxisRangeSlider("Y (height)", y - 1, scale[1], self._apply_crop)
            self.x_slider = AxisRangeSlider("X (width)", x - 1, scale[2], self._apply_crop)
            
            layout.addWidget(self.z_slider)
            layout.addWidget(self.y_slider)
            layout.addWidget(self.x_slider)
            
            layout.addSpacing(8)
            
            # Reset button
            self.reset_btn = QPushButton("Reset")
            self.reset_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.reset_btn.clicked.connect(self._reset)
            layout.addWidget(self.reset_btn)
            
            layout.addStretch()
        
        def _apply_crop(self):
            """Called automatically when any slider changes."""
            z_min, z_max = self.z_slider.get_range()
            y_min, y_max = self.y_slider.get_range()
            x_min, x_max = self.x_slider.get_range()
            
            # Calculate physical crop bounds in µm (from HT coordinates)
            phys_z_min = z_min * scale[0]
            phys_z_max = (z_max + 1) * scale[0]
            phys_y_min = y_min * scale[1]
            phys_y_max = (y_max + 1) * scale[1]
            phys_x_min = x_min * scale[2]
            phys_x_max = (x_max + 1) * scale[2]
            
            for layer in viewer.layers:
                if layer.name not in self.full_data:
                    continue
                    
                full = self.full_data[layer.name]
                info = self.layer_info[layer.name]
                layer_scale = info['scale']
                orig_translate = info['translate']
                
                if not info['is_fl']:
                    # HT/RI layer - direct pixel crop
                    cropped = full[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
                    layer.data = cropped
                    layer.translate = (phys_z_min, phys_y_min, phys_x_min)
                else:
                    # FL layer - convert physical bounds to FL pixel coordinates
                    # Account for original translate offset
                    fl_z, fl_y, fl_x = full.shape
                    
                    # FL physical bounds (with original translate)
                    fl_phys_z_start = orig_translate[0]
                    fl_phys_y_start = orig_translate[1]
                    fl_phys_x_start = orig_translate[2]
                    fl_phys_z_end = fl_phys_z_start + fl_z * layer_scale[0]
                    fl_phys_y_end = fl_phys_y_start + fl_y * layer_scale[1]
                    fl_phys_x_end = fl_phys_x_start + fl_x * layer_scale[2]
                    
                    # Find intersection of crop region with FL physical bounds
                    crop_z_start = max(phys_z_min, fl_phys_z_start)
                    crop_z_end = min(phys_z_max, fl_phys_z_end)
                    crop_y_start = max(phys_y_min, fl_phys_y_start)
                    crop_y_end = min(phys_y_max, fl_phys_y_end)
                    crop_x_start = max(phys_x_min, fl_phys_x_start)
                    crop_x_end = min(phys_x_max, fl_phys_x_end)
                    
                    # Check if there's any intersection
                    if crop_z_end <= crop_z_start or crop_y_end <= crop_y_start or crop_x_end <= crop_x_start:
                        # No intersection - hide layer with empty data
                        layer.data = np.zeros((1, 1, 1), dtype=full.dtype)
                        layer.visible = False
                        continue
                    
                    layer.visible = True
                    
                    # Convert physical intersection to FL pixel coordinates
                    fl_pix_z_min = int((crop_z_start - fl_phys_z_start) / layer_scale[0])
                    fl_pix_z_max = int(np.ceil((crop_z_end - fl_phys_z_start) / layer_scale[0]))
                    fl_pix_y_min = int((crop_y_start - fl_phys_y_start) / layer_scale[1])
                    fl_pix_y_max = int(np.ceil((crop_y_end - fl_phys_y_start) / layer_scale[1]))
                    fl_pix_x_min = int((crop_x_start - fl_phys_x_start) / layer_scale[2])
                    fl_pix_x_max = int(np.ceil((crop_x_end - fl_phys_x_start) / layer_scale[2]))
                    
                    # Clamp to valid range
                    fl_pix_z_min = max(0, min(fl_pix_z_min, fl_z))
                    fl_pix_z_max = max(0, min(fl_pix_z_max, fl_z))
                    fl_pix_y_min = max(0, min(fl_pix_y_min, fl_y))
                    fl_pix_y_max = max(0, min(fl_pix_y_max, fl_y))
                    fl_pix_x_min = max(0, min(fl_pix_x_min, fl_x))
                    fl_pix_x_max = max(0, min(fl_pix_x_max, fl_x))
                    
                    cropped = full[fl_pix_z_min:fl_pix_z_max, fl_pix_y_min:fl_pix_y_max, fl_pix_x_min:fl_pix_x_max]
                    layer.data = cropped
                    # New translate: where the cropped FL starts in physical space
                    layer.translate = (crop_z_start, crop_y_start, crop_x_start)
        
        def _reset(self):
            # Block signals to prevent multiple updates
            self.z_slider.slider.blockSignals(True)
            self.y_slider.slider.blockSignals(True)
            self.x_slider.slider.blockSignals(True)
            
            self.z_slider.reset()
            self.y_slider.reset()
            self.x_slider.reset()
            
            self.z_slider.slider.blockSignals(False)
            self.y_slider.slider.blockSignals(False)
            self.x_slider.slider.blockSignals(False)
            
            # Restore full data with original scale and translate
            for layer in viewer.layers:
                if layer.name in self.full_data:
                    layer.data = self.full_data[layer.name].copy()
                    layer.translate = self.layer_info[layer.name]['translate']
                    layer.visible = True
    
    crop_widget = CropWidget()
    dock = viewer.window.add_dock_widget(crop_widget, name="Crop", area="left")
    return dock


def _create_clipping_widget(viewer, ht_data: np.ndarray, scale: tuple):
    """Create clipping planes widget for volume sectioning with range sliders."""
    from superqt import QRangeSlider
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QCheckBox, QSizePolicy, QGroupBox
    )
    from qtpy.QtCore import Qt

    class ClippingWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.full_data = {}
            self.clip_enabled = {"x": False, "y": False, "z": False}
            # Store ranges as (min, max) tuples
            self.clip_ranges = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}

            # Only store HT layer (RI) - FL layers have different dimensions
            for layer in viewer.layers:
                if layer.name == "RI" and hasattr(layer, 'data') and isinstance(layer.data, np.ndarray):
                    self.full_data[layer.name] = layer.data.copy()

            z, y, x = ht_data.shape
            # Store max values for initialization
            self.max_vals = {"x": x - 1, "y": y - 1, "z": z - 1}
            # Initialize ranges to full extent
            self.clip_ranges = {"x": (0, x - 1), "y": (0, y - 1), "z": (0, z - 1)}

            layout = QVBoxLayout(self)
            layout.setSpacing(4)
            layout.setContentsMargins(4, 4, 4, 4)

            # Title
            title = QLabel("<b>Clipping Planes</b>")
            title.setStyleSheet("font-size: 13px;")
            layout.addWidget(title)

            desc = QLabel("Use range sliders to clip the volume on both sides of each axis.")
            desc.setStyleSheet("color: #888; font-size: 10px;")
            desc.setWordWrap(True)
            layout.addWidget(desc)

            # X clipping
            x_group = QGroupBox("X Axis")
            x_layout = QVBoxLayout(x_group)

            x_header = QHBoxLayout()
            self.x_enabled = QCheckBox("Enable")
            self.x_enabled.toggled.connect(lambda v: self._toggle_clip("x", v))
            x_header.addWidget(self.x_enabled)
            x_header.addStretch()
            self.x_label = QLabel(f"0 - {x-1}")
            self.x_label.setStyleSheet("font-family: monospace;")
            x_header.addWidget(self.x_label)
            x_layout.addLayout(x_header)

            self.x_slider = QRangeSlider(Qt.Horizontal)
            self.x_slider.setRange(0, x - 1)
            self.x_slider.setValue((0, x - 1))
            self.x_slider.valueChanged.connect(lambda v: self._on_range_change("x", v))
            x_layout.addWidget(self.x_slider)
            layout.addWidget(x_group)

            # Y clipping
            y_group = QGroupBox("Y Axis")
            y_layout = QVBoxLayout(y_group)

            y_header = QHBoxLayout()
            self.y_enabled = QCheckBox("Enable")
            self.y_enabled.toggled.connect(lambda v: self._toggle_clip("y", v))
            y_header.addWidget(self.y_enabled)
            y_header.addStretch()
            self.y_label = QLabel(f"0 - {y-1}")
            self.y_label.setStyleSheet("font-family: monospace;")
            y_header.addWidget(self.y_label)
            y_layout.addLayout(y_header)

            self.y_slider = QRangeSlider(Qt.Horizontal)
            self.y_slider.setRange(0, y - 1)
            self.y_slider.setValue((0, y - 1))
            self.y_slider.valueChanged.connect(lambda v: self._on_range_change("y", v))
            y_layout.addWidget(self.y_slider)
            layout.addWidget(y_group)

            # Z clipping
            z_group = QGroupBox("Z Axis (Depth)")
            z_layout = QVBoxLayout(z_group)

            z_header = QHBoxLayout()
            self.z_enabled = QCheckBox("Enable")
            self.z_enabled.toggled.connect(lambda v: self._toggle_clip("z", v))
            z_header.addWidget(self.z_enabled)
            z_header.addStretch()
            self.z_label = QLabel(f"0 - {z-1}")
            self.z_label.setStyleSheet("font-family: monospace;")
            z_header.addWidget(self.z_label)
            z_layout.addLayout(z_header)

            self.z_slider = QRangeSlider(Qt.Horizontal)
            self.z_slider.setRange(0, z - 1)
            self.z_slider.setValue((0, z - 1))
            self.z_slider.valueChanged.connect(lambda v: self._on_range_change("z", v))
            z_layout.addWidget(self.z_slider)
            layout.addWidget(z_group)

            # Reset button
            reset_btn = QPushButton("Reset All")
            reset_btn.clicked.connect(self._reset)
            layout.addWidget(reset_btn)

            layout.addStretch()

        def _toggle_clip(self, axis: str, enabled: bool):
            self.clip_enabled[axis] = enabled
            self._apply_clipping()

        def _on_range_change(self, axis: str, value: tuple):
            min_v, max_v = value
            self.clip_ranges[axis] = (min_v, max_v)
            label = getattr(self, f"{axis}_label")
            label.setText(f"{min_v} - {max_v}")
            if self.clip_enabled[axis]:
                self._apply_clipping()

        def _apply_clipping(self):
            """Apply clipping by masking data with NaN outside the range."""
            for layer in viewer.layers:
                if layer.name not in self.full_data:
                    continue

                full = self.full_data[layer.name].copy()
                z, y, x = full.shape

                # Apply each enabled clip - mask values OUTSIDE the range
                if self.clip_enabled["x"]:
                    x_min, x_max = self.clip_ranges["x"]
                    full[:, :, :x_min] = np.nan
                    full[:, :, x_max + 1:] = np.nan

                if self.clip_enabled["y"]:
                    y_min, y_max = self.clip_ranges["y"]
                    full[:, :y_min, :] = np.nan
                    full[:, y_max + 1:, :] = np.nan

                if self.clip_enabled["z"]:
                    z_min, z_max = self.clip_ranges["z"]
                    full[:z_min, :, :] = np.nan
                    full[z_max + 1:, :, :] = np.nan

                layer.data = full

        def _reset(self):
            """Reset all clipping."""
            # Block signals during reset
            self.x_slider.blockSignals(True)
            self.y_slider.blockSignals(True)
            self.z_slider.blockSignals(True)

            self.x_enabled.setChecked(False)
            self.y_enabled.setChecked(False)
            self.z_enabled.setChecked(False)

            # Reset sliders to full range
            self.x_slider.setValue((0, self.max_vals["x"]))
            self.y_slider.setValue((0, self.max_vals["y"]))
            self.z_slider.setValue((0, self.max_vals["z"]))

            # Reset labels
            self.x_label.setText(f"0 - {self.max_vals['x']}")
            self.y_label.setText(f"0 - {self.max_vals['y']}")
            self.z_label.setText(f"0 - {self.max_vals['z']}")

            # Reset ranges
            self.clip_ranges = {
                "x": (0, self.max_vals["x"]),
                "y": (0, self.max_vals["y"]),
                "z": (0, self.max_vals["z"]),
            }

            self.x_slider.blockSignals(False)
            self.y_slider.blockSignals(False)
            self.z_slider.blockSignals(False)

            # Restore full data
            for layer in viewer.layers:
                if layer.name in self.full_data:
                    layer.data = self.full_data[layer.name].copy()

    widget = ClippingWidget()
    viewer.window.add_dock_widget(widget, name="Clipping", area="left")
    return widget


def _create_fl_z_offset_widget(viewer, loader, ht_data: np.ndarray, scale: tuple, initial_mode: str, crop_widget=None):
    """Create widget for adjusting FL Z offset interactively.
    
    Uses napari's translate parameter instead of resampling data.
    """
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QSlider, QComboBox, QSizePolicy
    )
    from qtpy.QtCore import Qt

    class FLZOffsetWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.fl_layers = {}
            self.current_offset_um = 0.0

            # Get registration params
            self.reg_params = loader.reg_params
            self.ht_z_total = ht_data.shape[0] * self.reg_params.ht_res_z

            # Find FL layers
            for ch_name in loader.fl_data.keys():
                for layer in viewer.layers:
                    if layer.name == ch_name:
                        self.fl_layers[ch_name] = layer
                        # Store initial translate
                        layer._initial_translate = layer.translate
                        break

            if not self.fl_layers:
                layout = QVBoxLayout(self)
                layout.addWidget(QLabel("No fluorescence data available"))
                layout.addStretch()
                return

            layout = QVBoxLayout(self)
            layout.setSpacing(4)
            layout.setContentsMargins(4, 4, 4, 4)

            # Title
            title = QLabel("<b>FL Z Offset</b>")
            title.setStyleSheet("font-size: 12px;")
            layout.addWidget(title)

            # Offset slider
            offset_row = QHBoxLayout()
            offset_lbl = QLabel("Z Offset:")
            offset_lbl.setFixedWidth(60)
            offset_row.addWidget(offset_lbl)
            self.offset_label = QLabel("0.0 µm")
            self.offset_label.setStyleSheet("font-family: monospace; font-weight: bold;")
            self.offset_label.setFixedWidth(80)
            offset_row.addWidget(self.offset_label)
            layout.addLayout(offset_row)

            # Slider (range: full HT Z extent in both directions)
            self.offset_range = self.ht_z_total
            self.offset_slider = QSlider(Qt.Horizontal)
            self.offset_slider.setRange(-1000, 1000)  # -100.0% to +100.0%
            self.offset_slider.setValue(0)
            self.offset_slider.valueChanged.connect(self._on_slider_change)
            layout.addWidget(self.offset_slider)

            # Range info
            range_label = QLabel(f"Range: ±{self.offset_range:.1f} µm")
            range_label.setStyleSheet("color: #888; font-size: 10px;")
            layout.addWidget(range_label)

            layout.addSpacing(8)

            # Reset button
            reset_btn = QPushButton("Reset")
            reset_btn.clicked.connect(self._reset)
            layout.addWidget(reset_btn)

            layout.addStretch()

        def _on_slider_change(self, value: int):
            # Convert slider value to µm offset
            self.current_offset_um = (value / 1000.0) * self.offset_range
            self.offset_label.setText(f"{self.current_offset_um:+.1f} µm")
            self._apply_offset()

        def _apply_offset(self):
            """Update FL layer translate to apply Z offset."""
            for ch_name, layer in self.fl_layers.items():
                initial = layer._initial_translate
                # Only modify Z (first component)
                new_translate = (
                    initial[0] + self.current_offset_um,
                    initial[1],
                    initial[2]
                )
                layer.translate = new_translate

        def _reset(self):
            """Reset to initial position."""
            self.current_offset_um = 0.0
            self.offset_slider.setValue(0)
            self.offset_label.setText("0.0 µm")
            for ch_name, layer in self.fl_layers.items():
                layer.translate = layer._initial_translate

    widget = FLZOffsetWidget()
    dock = viewer.window.add_dock_widget(widget, name="FL Z", area="right")
    return dock


def _create_animation_widget(viewer, output_dir: Path):
    """Create animation export widget with turntable and slice sweep options."""
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QSpinBox, QComboBox, QProgressBar, QSizePolicy, QGroupBox
    )
    from qtpy.QtCore import Qt, QTimer

    class AnimationWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.exporter = AnimationExporter(viewer, output_dir)
            self._export_timer: QTimer | None = None
            self._export_mode: str = ""

            layout = QVBoxLayout(self)
            layout.setSpacing(4)
            layout.setContentsMargins(4, 4, 4, 4)

            # Title
            title = QLabel("<b>Animation Export</b>")
            title.setStyleSheet("font-size: 13px;")
            layout.addWidget(title)

            desc = QLabel("Export turntable or slice animations as GIF/MP4.")
            desc.setStyleSheet("color: #888; font-size: 10px;")
            desc.setWordWrap(True)
            layout.addWidget(desc)

            # Shared speed control at top
            speed_row = QHBoxLayout()
            speed_row.addWidget(QLabel("Speed:"))
            self.speed_combo = QComboBox()
            self.speed_combo.addItems(["Slow (150ms)", "Normal (100ms)", "Fast (50ms)"])
            self.speed_combo.setCurrentIndex(1)
            speed_row.addWidget(self.speed_combo)
            speed_row.addStretch()
            layout.addLayout(speed_row)

            layout.addSpacing(4)

            # Turntable section
            turntable_group = QGroupBox("Turntable (360°)")
            turntable_layout = QVBoxLayout(turntable_group)

            # Frames control
            frames_row = QHBoxLayout()
            frames_row.addWidget(QLabel("Frames:"))
            self.frames_spin = QSpinBox()
            self.frames_spin.setRange(12, 360)
            self.frames_spin.setValue(36)
            self.frames_spin.setToolTip("Number of frames (36 = 10° per frame)")
            frames_row.addWidget(self.frames_spin)
            frames_row.addStretch()
            turntable_layout.addLayout(frames_row)

            # Export buttons
            btn_row = QHBoxLayout()
            self.gif_btn = QPushButton("Export GIF")
            self.gif_btn.clicked.connect(lambda: self._export_turntable("gif"))
            btn_row.addWidget(self.gif_btn)
            self.mp4_btn = QPushButton("Export MP4")
            self.mp4_btn.clicked.connect(lambda: self._export_turntable("mp4"))
            btn_row.addWidget(self.mp4_btn)
            turntable_layout.addLayout(btn_row)

            layout.addWidget(turntable_group)

            # Slice sweep section
            sweep_group = QGroupBox("Slice Sweep")
            sweep_layout = QVBoxLayout(sweep_group)

            axis_row = QHBoxLayout()
            axis_row.addWidget(QLabel("Axis:"))
            self.axis_combo = QComboBox()
            self.axis_combo.addItems(["Z (depth)", "Y (height)", "X (width)"])
            axis_row.addWidget(self.axis_combo)
            axis_row.addStretch()
            sweep_layout.addLayout(axis_row)

            sweep_btn_row = QHBoxLayout()
            self.sweep_gif_btn = QPushButton("Export GIF")
            self.sweep_gif_btn.clicked.connect(lambda: self._export_sweep("gif"))
            sweep_btn_row.addWidget(self.sweep_gif_btn)
            self.sweep_mp4_btn = QPushButton("Export MP4")
            self.sweep_mp4_btn.clicked.connect(lambda: self._export_sweep("mp4"))
            sweep_btn_row.addWidget(self.sweep_mp4_btn)
            sweep_layout.addLayout(sweep_btn_row)

            layout.addWidget(sweep_group)

            # Progress bar
            self.progress = QProgressBar()
            self.progress.setVisible(False)
            layout.addWidget(self.progress)

            # Status
            self.status = QLabel("")
            self.status.setStyleSheet("color: #888; font-size: 10px;")
            self.status.setWordWrap(True)
            layout.addWidget(self.status)

            layout.addStretch()

        def _get_duration_ms(self) -> int:
            idx = self.speed_combo.currentIndex()
            return [150, 100, 50][idx]

        def _export_turntable(self, fmt: str):
            """Start turntable export using QTimer on main thread."""
            if self._export_timer and self._export_timer.isActive():
                return

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"turntable_{timestamp}.{fmt}"

            self.progress.setVisible(True)
            self.progress.setValue(0)
            self.progress.setMaximum(self.frames_spin.value())
            self.status.setText("Exporting turntable animation...")
            self._set_buttons_enabled(False)

            # Initialize export
            self.exporter.start_turntable_export(
                filename, self.frames_spin.value(), self._get_duration_ms()
            )
            self._export_mode = "turntable"

            # Use QTimer to capture frames on main thread
            self._export_timer = QTimer()
            self._export_timer.timeout.connect(self._capture_frame)
            self._export_timer.start(50)  # Capture at ~20fps

        def _export_sweep(self, fmt: str):
            """Start slice sweep export using QTimer on main thread."""
            if self._export_timer and self._export_timer.isActive():
                return

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            axis_name = ["z", "y", "x"][self.axis_combo.currentIndex()]
            filename = f"sweep_{axis_name}_{timestamp}.{fmt}"

            self.progress.setVisible(True)
            self.progress.setValue(0)
            self.status.setText("Exporting slice sweep animation...")
            self._set_buttons_enabled(False)

            # Initialize export - use shared speed control
            n_slices = self.exporter.start_slice_sweep_export(
                filename, self.axis_combo.currentIndex(), self._get_duration_ms()
            )
            self.progress.setMaximum(n_slices)
            self._export_mode = "sweep"

            # Use QTimer to capture frames on main thread
            self._export_timer = QTimer()
            self._export_timer.timeout.connect(self._capture_frame)
            self._export_timer.start(30)  # Faster for slice sweeps

        def _capture_frame(self):
            """Capture one frame - called by QTimer on main thread."""
            try:
                if self._export_mode == "turntable":
                    current, total, done = self.exporter.capture_turntable_frame()
                else:
                    current, total, done = self.exporter.capture_sweep_frame()

                self.progress.setValue(current)

                if done:
                    self._export_timer.stop()
                    try:
                        path = self.exporter.finish_export()
                        self._on_finished(str(path))
                    except Exception as e:
                        self._on_error(str(e))

            except Exception as e:
                self._export_timer.stop()
                self.exporter.cancel_export()
                self._on_error(str(e))

        def _on_finished(self, path: str):
            self.progress.setVisible(False)
            self.status.setText(f"Saved: {path}")
            self._set_buttons_enabled(True)

        def _on_error(self, msg: str):
            self.progress.setVisible(False)
            self.status.setText(f"Error: {msg}")
            self._set_buttons_enabled(True)

        def _set_buttons_enabled(self, enabled: bool):
            self.gif_btn.setEnabled(enabled)
            self.mp4_btn.setEnabled(enabled)
            self.sweep_gif_btn.setEnabled(enabled)
            self.sweep_mp4_btn.setEnabled(enabled)

    widget = AnimationWidget()
    dock = viewer.window.add_dock_widget(widget, name="Animation", area="right")
    return dock


def view_3d(
    tcf_path: str | Path,
    show_slices: bool = False,
    rendering: str = "mip",
    screenshot: str | None = None,
    enable_downsampling: bool = True,
    z_offset_mode: str = "auto",
) -> None:
    """
    Open interactive 3D viewer for a TCF file.

    Args:
        tcf_path: Path to TCF file
        show_slices: Start in 2D slice mode (default: 3D volume)
        rendering: Volume rendering - "mip", "attenuated_mip", "minip", "average"
        screenshot: Path to save screenshot
        enable_downsampling: Auto-downsample during interaction for large volumes
        z_offset_mode: FL Z alignment mode - "auto", "start", or "center"
    """
    try:
        import napari
    except ImportError:
        raise ImportError(
            "napari is required for 3D viewing. Install with:\n"
            "  pip install 'tomocube-tools[3d]'"
        )

    from tomocube.core.file import TCFFileLoader

    tcf_path = Path(tcf_path)

    with TCFFileLoader(str(tcf_path)) as loader:
        loader.load_timepoint(0)

        ht_data = loader.data_3d.copy()
        scale = _get_voxel_scale(loader)
        title = f"TCF 3D: {tcf_path.stem}"

        # Track all layer data for downsampling
        layers_data = {}

        # Create viewer
        viewer = napari.Viewer(title=title)

        # Contrast limits
        p1, p99 = np.percentile(ht_data, [1, 99])

        # Add HT volume
        ht_layer = viewer.add_image(
            ht_data,
            name="RI",
            colormap="gray",
            contrast_limits=(p1, p99),
            scale=scale,
            blending="translucent",
            opacity=0.9,
            rendering=rendering,
        )
        layers_data["RI"] = ht_data

        if rendering == "attenuated_mip":
            ht_layer.attenuation = 0.5

        # Add FL channels using napari's native coordinate system
        # FL and HT cover the same physical XY area (e.g., 230×230 µm) at different resolutions
        # We use napari's scale/translate to overlay them without resampling
        if loader.has_fluorescence:
            colormaps = ["green", "magenta", "cyan", "yellow", "red", "blue"]

            rp = loader.reg_params
            ht_z, ht_y, ht_x = ht_data.shape

            # FL scale uses its own resolution (different from HT)
            fl_scale = (rp.fl_res_z, rp.fl_res_y, rp.fl_res_x)

            # Calculate physical FOV to verify alignment
            ht_fov_x = ht_x * rp.ht_res_x
            ht_fov_y = ht_y * rp.ht_res_y
            ht_fov_z = ht_z * rp.ht_res_z

            vprint(f"\n{'='*60}")
            vprint(f"FL Overlay (native resolution, no resampling)")
            vprint(f"{'='*60}")
            vprint(f"  HT: {ht_x}×{ht_y}×{ht_z} px @ {rp.ht_res_x:.4f} µm/px = {ht_fov_x:.1f}×{ht_fov_y:.1f}×{ht_fov_z:.1f} µm")

            for idx, (ch_name, fl_data) in enumerate(loader.fl_data.items()):
                fl_z, fl_y, fl_x = fl_data.shape
                fl_fov_x = fl_x * rp.fl_res_x
                fl_fov_y = fl_y * rp.fl_res_y
                fl_fov_z = fl_z * rp.fl_res_z

                vprint(f"  {ch_name}: {fl_x}×{fl_y}×{fl_z} px @ {rp.fl_res_x:.4f} µm/px = {fl_fov_x:.1f}×{fl_fov_y:.1f}×{fl_fov_z:.1f} µm")

                # Get Z offset from file (physical µm offset)
                ch_offset_z = rp.get_offset_z(ch_name)

                # Determine Z translation based on mode
                if z_offset_mode == "auto":
                    # Center FL on HT volume
                    z_translate = (ht_fov_z - fl_fov_z) / 2
                    vprint(f"    Z: centered at {z_translate:.1f} µm (auto)")
                elif z_offset_mode == "center":
                    # OffsetZ is center of FL in HT space
                    z_translate = ch_offset_z - fl_fov_z / 2
                    vprint(f"    Z: center at {ch_offset_z:.1f} µm => translate {z_translate:.1f} µm")
                else:  # "start"
                    # OffsetZ is where FL starts in HT space
                    z_translate = ch_offset_z
                    vprint(f"    Z: starts at {ch_offset_z:.1f} µm")

                # XY centering: both FOVs should match, but center anyway
                # (handles any small FOV differences)
                y_translate = (ht_fov_y - fl_fov_y) / 2
                x_translate = (ht_fov_x - fl_fov_x) / 2

                fl_translate = (z_translate, y_translate, x_translate)

                fl_nonzero = fl_data[fl_data > 0]
                if len(fl_nonzero) > 0:
                    fl_p1, fl_p99 = np.percentile(fl_nonzero, [5, 99.5])
                else:
                    fl_p1, fl_p99 = 0, 1

                fl_copy = fl_data.astype(np.float32).copy()
                viewer.add_image(
                    fl_copy,
                    name=ch_name,
                    colormap=colormaps[idx % len(colormaps)],
                    contrast_limits=(fl_p1, fl_p99),
                    scale=fl_scale,
                    translate=fl_translate,
                    blending="additive",
                    opacity=0.8,
                    rendering=rendering,
                )
                layers_data[ch_name] = fl_copy

            print(f"{'='*60}\n", flush=True)

        # Scale bar
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "µm"
        viewer.scale_bar.font_size = 14

        # Hide napari's built-in layer list - we have our own
        viewer.window._qt_viewer.dockLayerList.setVisible(False)
        viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        # Add control widgets and tabify them for cleaner UI
        # Left side: Camera and Crop in tabs
        camera_dock = _create_camera_controls(viewer)  # left
        crop_dock = _create_crop_widget(viewer, ht_data, scale)  # left
        
        # Right side: Layers, Histogram, FL, Animation in tabs
        layers_dock = _create_layer_controls(viewer)  # right
        histogram_dock = _create_histogram_widget(viewer)  # right
        fl_dock = None
        if loader.has_fluorescence:
            fl_dock = _create_fl_z_offset_widget(viewer, loader, ht_data, scale, z_offset_mode, crop_dock)  # right
        animation_dock = _create_animation_widget(viewer, tcf_path.parent)  # right
        
        # Tabify widgets: group related controls into tabs
        # Left side: Camera + Crop
        main_window = viewer.window._qt_window
        main_window.tabifyDockWidget(camera_dock, crop_dock)
        camera_dock.raise_()  # Show Camera tab by default
        
        # Right side: Layers + Histogram + FL + Animation
        main_window.tabifyDockWidget(layers_dock, histogram_dock)
        if fl_dock is not None:
            main_window.tabifyDockWidget(histogram_dock, fl_dock)
            main_window.tabifyDockWidget(fl_dock, animation_dock)
        else:
            main_window.tabifyDockWidget(histogram_dock, animation_dock)
        layers_dock.raise_()  # Show Layers tab by default

        # Set view mode
        if show_slices:
            viewer.dims.ndisplay = 2
            viewer.dims.set_point(0, ht_data.shape[0] // 2)
        else:
            viewer.dims.ndisplay = 3
            viewer.camera.angles = (0, -30, 45)
            viewer.camera.zoom = 0.8

        # Print info
        z, y, x = ht_data.shape
        volume_size = ht_data.size
        print(f"\nVolume: {x} x {y} x {z} voxels ({volume_size / 1e6:.1f}M voxels)")
        print(f"Physical size: {x*scale[2]:.1f} x {y*scale[1]:.1f} x {z*scale[0]:.1f} µm")
        print(f"\nKeyboard shortcuts:")
        print(f"  1-6: Camera presets (Top/Bottom/Front/Back/Left/Right)")
        print(f"  0: Isometric view  |  R: Reset  |  F: Fit")
        print(f"  +/-: Zoom  |  2/3: Toggle slice/3D view", flush=True)

        if screenshot:
            time.sleep(0.5)
            viewer.screenshot(screenshot)
            print(f"\n  Screenshot saved: {screenshot}")

        napari.run()
