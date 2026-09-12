"""
3D Volume Viewer for Tomocube TCF files.

Provides interactive 3D visualization using napari with:
- Volume rendering (MIP, attenuated, etc.)
- Multi-channel fluorescence overlay
- XYZ range sliders for render-only 3D clipping
- Layer controls with background removal
- Camera presets for different viewing angles
- Scale bar with physical units
- Screenshot and animation export (GIF/MP4)

Usage:
    python -m tomocube view3d sample.TCF
    python -m tomocube view3d sample.TCF --slices

Keyboard shortcuts:
    1-6     Camera presets (Top, Bottom, Front, Back, Left, Right)
    0       Isometric view
    R       Reset camera
    F       Fit view to data
    +/-     Zoom in/out
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tomocube.processing.registration import FluorescenceRegistration, Z_OFFSET_MODES
from tomocube.viewer.volume_geometry import VolumeGeometry, constrain_native_slicing, native_slice_conflicts

if TYPE_CHECKING:
    from tomocube.core.file import TCFFileLoader

# =============================================================================
# Animation Export
# =============================================================================

class AnimationExporter:
    """Capture on the Qt thread and publish GIF/MP4 only when encoding succeeds."""

    def __init__(self, viewer, output_dir: Path):
        self.viewer = viewer
        self.output_dir = Path(output_dir)
        self._is_exporting = False
        self._frames = []
        self._export_config = {}
        self._original_state = {}

    def _start(self, filename, duration_ms):
        if self._is_exporting:
            raise RuntimeError("An animation export is already running")
        if Path(filename).suffix.lower() not in (".gif", ".mp4"):
            raise ValueError("Animation filename must end in .gif or .mp4")
        if not np.isscalar(duration_ms) or not np.isfinite(duration_ms) or duration_ms < 10:
            raise ValueError("Frame duration must be at least 10 milliseconds")
        self._export_config = {"filename": filename, "duration_ms": float(duration_ms),
                               "current_frame": 0}
        self._original_state = {
            "angles": self.viewer.scene.camera.angles,
            "center": self.viewer.scene.camera.center,
            "zoom": self.viewer.scene.camera.zoom,
            "ndisplay": self.viewer.dims.ndisplay,
            "point": tuple(self.viewer.dims.point),
            "order": tuple(self.viewer.dims.order),
        }
        self._frames = []
        self._is_exporting = True

    def start_turntable_export(self, filename: str, n_frames: int, duration_ms: int) -> None:
        if isinstance(n_frames, (bool, np.bool_)) or not isinstance(n_frames, (int, np.integer)) or n_frames <= 0:
            raise ValueError("n_frames must be a positive integer")
        self._start(filename, duration_ms)
        self._export_config.update(mode="turntable", n_frames=n_frames)
        self.viewer.dims.ndisplay = 3

    def start_slice_sweep_export(self, filename: str, axis: int, duration_ms: int) -> int:
        """Sweep actual napari world positions, including final/singleton samples."""
        if isinstance(axis, (bool, np.bool_)) or axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        conflicts = native_slice_conflicts(self.viewer, axis)
        if conflicts:
            raise ValueError(
                "Native napari slicing cannot represent this plane for " + ", ".join(conflicts)
                + ". Use the Tomocube slice viewer or GIF export for calibrated orthogonal planes."
            )
        n_slices = self.viewer.dims.nsteps[axis]
        start, _, step = self.viewer.dims.range[axis]
        self._start(filename, duration_ms)
        self._export_config.update(mode="sweep", axis=axis, n_frames=n_slices,
                                   start=start, step=step)
        self.viewer.dims.order = (axis, *(i for i in range(3) if i != axis))
        self.viewer.dims.ndisplay = 2
        return n_slices

    def capture_turntable_frame(self) -> tuple[int, int, bool]:
        return self._capture_frame("turntable")

    def capture_sweep_frame(self) -> tuple[int, int, bool]:
        return self._capture_frame("sweep")

    def _capture_frame(self, mode):
        if not self._is_exporting or self._export_config["mode"] != mode:
            raise RuntimeError(f"No {mode} export is running")
        cfg = self._export_config
        current, count = cfg["current_frame"], cfg["n_frames"]
        if current >= count:
            return current, count, True
        if mode == "turntable":
            roll, pitch, yaw = self._original_state["angles"]
            self.viewer.scene.camera.angles = (roll, pitch, yaw + 360 * current / count)
        else:
            self.viewer.dims.set_point(cfg["axis"], cfg["start"] + current * cfg["step"])
        from qtpy.QtWidgets import QApplication
        QApplication.processEvents()
        self._frames.append(self.viewer.screenshot(canvas_only=True))
        cfg["current_frame"] += 1
        return current + 1, count, current + 1 == count

    def finish_export(self) -> Path:
        """Retain the previous output and restore the view even on encoding failure."""
        from tomocube.processing.outputs import atomic_output

        cfg = self._export_config
        if not self._is_exporting:
            raise RuntimeError("No animation export is running")
        try:
            if len(self._frames) != cfg["n_frames"]:
                raise ValueError("Animation capture is incomplete")
            shape = self._frames[0].shape
            if any(frame.shape != shape for frame in self._frames):
                raise ValueError("Canvas size changed during capture; keep it fixed and export again")
            frames = [frame[..., :3] for frame in self._frames]
            output_path = self.output_dir / cfg["filename"]
            sources = [layer.metadata[key] for layer in self.viewer.layers
                       for key in ("source", "registration_path") if key in layer.metadata]
            with atomic_output(output_path, sources=sources) as temporary:
                if output_path.suffix.lower() == ".mp4":
                    import imageio.v2 as iio
                    # H.264 yuv420 needs even dimensions. Pad only the final
                    # row/column, preserving screenshot pixels and scale bars.
                    with iio.get_writer(temporary, format="FFMPEG",
                                        fps=1000.0 / cfg["duration_ms"],
                                        macro_block_size=1) as writer:
                        for frame in frames:
                            pad = ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0))
                            writer.append_data(np.pad(frame, pad))
                else:
                    import imageio.v3 as iio
                    iio.imwrite(temporary, frames, duration=cfg["duration_ms"],
                                loop=0, plugin="pillow")
            return output_path
        finally:
            self.cancel_export()

    def cancel_export(self) -> None:
        if not self._is_exporting:
            return
        try:
            # Restore the axis order in 3D to avoid an unsupported intermediate
            # rotated slice while switching back from a sweep.
            self.viewer.dims.ndisplay = 3
            self.viewer.dims.order = self._original_state["order"]
            self.viewer.dims.ndisplay = self._original_state["ndisplay"]
            for axis, point in enumerate(self._original_state["point"]):
                self.viewer.dims.set_point(axis, point)
            self.viewer.scene.camera.angles = self._original_state["angles"]
            self.viewer.scene.camera.center = self._original_state["center"]
            self.viewer.scene.camera.zoom = self._original_state["zoom"]
        finally:
            self._is_exporting = False
            self._frames = []


def _get_voxel_scale(loader: TCFFileLoader) -> tuple[float, float, float]:
    """Measured HT spacing in ZYX micrometers; invalid calibration is an error."""
    params = loader.reg_params
    scale = (float(params.ht_res_z), float(params.ht_res_y), float(params.ht_res_x))
    if not np.isfinite(scale).all() or min(scale) <= 0:
        raise ValueError("HT resolutions must be finite and positive")
    return scale


def _display_sample(data: np.ndarray) -> np.ndarray:
    """Bound display-only statistics to one million deterministic sample values."""
    step = max(1, (data.size + 999_999) // 1_000_000)
    # flat slicing bounds allocation even when a native array is non-contiguous.
    return data.flat[::step]


def _display_limits(data, percentiles=(1, 99), *, positive=False):
    sample = _display_sample(data)
    if positive:
        sample = sample[sample > 0]
    if sample.size == 0:
        return (0.0, 1.0)
    low, high = (float(value) for value in np.percentile(sample, percentiles))
    if low == high:
        padding = max(abs(low) * 1e-4, 1e-6)
        low, high = low - padding, high + padding
    return low, high


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
            self.original_data = layer.data
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
                    low = float(np.percentile(_display_sample(data), value))
                    high = self.original_contrast[1]
                    if low < high:
                        self.layer.contrast_limits = (low, high)
        
        def auto_contrast(self):
            """Apply auto contrast based on data percentiles."""
            if hasattr(self.layer, 'data') and hasattr(self.layer, 'contrast_limits'):
                data = self.layer.data
                if data.size > 0:
                    p1, p99 = _display_limits(data)
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
            flat = _display_sample(data)
            
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
            low, high = _display_limits(data, (low_pct, high_pct))
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
                btn.clicked.connect(lambda checked, a=angles: self._set_view(a))
                grid.addWidget(btn, i // 3, i % 3)

            layout.addLayout(grid)

            # Isometric and reset
            iso_row = QHBoxLayout()

            iso_btn = QPushButton("Isometric [0]")
            iso_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            iso_btn.setToolTip("45-degree isometric view (press 0)")
            iso_btn.clicked.connect(lambda: self._set_view((0, -30, 45)))
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
            zoom_fit.clicked.connect(lambda: viewer.fit_to_view())
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
                shortcut.activated.connect(lambda a=angles: self._set_view(a))

            # Isometric (0)
            iso_shortcut = QShortcut(QKeySequence("0"), window)
            iso_shortcut.activated.connect(lambda: self._set_view((0, -30, 45)))

            # Reset (R)
            reset_shortcut = QShortcut(QKeySequence("R"), window)
            reset_shortcut.activated.connect(self._reset_camera)

            # Fit (F)
            fit_shortcut = QShortcut(QKeySequence("F"), window)
            fit_shortcut.activated.connect(lambda: viewer.fit_to_view())

            # Zoom in/out
            zoom_in = QShortcut(QKeySequence("="), window)
            zoom_in.activated.connect(lambda: self._zoom(1.25))
            zoom_in2 = QShortcut(QKeySequence("+"), window)
            zoom_in2.activated.connect(lambda: self._zoom(1.25))

            zoom_out = QShortcut(QKeySequence("-"), window)
            zoom_out.activated.connect(lambda: self._zoom(0.8))

        def _set_view(self, angles):
            viewer.dims.ndisplay = 3
            viewer.scene.camera.angles = angles

        def _zoom(self, factor):
            viewer.scene.camera.zoom *= factor

        def _reset_camera(self):
            viewer.dims.ndisplay = 3
            self._set_view(self.default_angles)
            viewer.scene.camera.zoom = self.default_zoom
            viewer.fit_to_view()

    widget = CameraWidget()
    dock = viewer.window.add_dock_widget(widget, name="Camera", area="left")
    return dock


def _create_crop_widget(viewer, geometry: VolumeGeometry):
    """Clip a calibrated HT box in the renderer without copying or slicing data."""
    from superqt import QRangeSlider
    from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
    from qtpy.QtCore import Qt

    class CropWidget(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            description = QLabel(
                "Crop the 3D rendering in HT coordinates. Slice mode shows full "
                "planes. Source data and layer visibility are preserved."
            )
            description.setWordWrap(True)
            layout.addWidget(description)
            self.sliders = []
            self.labels = []
            for axis, name in enumerate(("Z (depth)", "Y (height)", "X (width)")):
                label = QLabel()
                self.labels.append(label)
                layout.addWidget(label)
                slider = QRangeSlider(Qt.Horizontal)
                slider.setRange(0, geometry.ht_shape[axis] - 1)
                slider.setValue(geometry.ranges[axis])
                slider.setEnabled(geometry.ht_shape[axis] > 1)
                slider.setAccessibleName(f"Crop {name}")
                slider.setToolTip("Inclusive HT sample indices")
                slider.valueChanged.connect(self._apply_crop)
                self.sliders.append(slider)
                layout.addWidget(slider)
            reset = QPushButton("Reset crop")
            reset.clicked.connect(self._reset)
            layout.addWidget(reset)
            layout.addStretch()
            self._update_labels()
            # Clipping is a volume-rendering operation; avoid presenting it as
            # an effective 2D crop while slice mode is active.
            viewer.dims.events.ndisplay.connect(self._update_enabled)
            self._update_enabled()

        def _update_enabled(self, event=None):
            for axis, slider in enumerate(self.sliders):
                slider.setEnabled(viewer.dims.ndisplay == 3 and geometry.ht_shape[axis] > 1)

        def _update_labels(self):
            for axis, label in enumerate(self.labels):
                start, stop = geometry.ranges[axis]
                low = (start - 0.5) * geometry.ht_spacing[axis]
                high = (stop + 0.5) * geometry.ht_spacing[axis]
                label.setText(f"{'ZYX'[axis]}: {start}–{stop}  |  {low:.3g}–{high:.3g} µm")

        def _apply_crop(self, value=None):
            geometry.set_crop(tuple(slider.value() for slider in self.sliders))
            self._update_labels()

        def _reset(self):
            for axis, slider in enumerate(self.sliders):
                slider.blockSignals(True)
                slider.setValue((0, geometry.ht_shape[axis] - 1))
                slider.blockSignals(False)
            self._apply_crop()

    return viewer.window.add_dock_widget(CropWidget(), name="Crop", area="left")


def _create_fl_z_offset_widget(viewer, geometry: VolumeGeometry):
    """Adjust fluorescence in HT-world Z while retaining its affine and crop."""
    from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QDoubleSpinBox

    class FLZOffsetWidget(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            description = QLabel(
                "Additional FL shift in HT Z (µm). Positive values move toward "
                "larger HT Z. This display adjustment does not change saved registration."
            )
            description.setWordWrap(True)
            layout.addWidget(description)
            self.offset = QDoubleSpinBox()
            extent = geometry.ht_shape[0] * geometry.ht_spacing[0]
            self.offset.setRange(-extent, extent)
            self.offset.setDecimals(4)
            self.offset.setSingleStep(float(geometry.ht_spacing[0]))
            self.offset.setSuffix(" µm")
            self.offset.setAccessibleName("Additional fluorescence Z offset")
            self.offset.valueChanged.connect(geometry.set_fl_z_offset)
            layout.addWidget(self.offset)
            reset = QPushButton("Reset FL shift")
            reset.clicked.connect(lambda: self.offset.setValue(0))
            layout.addWidget(reset)
            layout.addStretch()

    return viewer.window.add_dock_widget(FLZOffsetWidget(), name="FL Z", area="right")


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
            try:
                self.exporter.start_turntable_export(
                    filename, self.frames_spin.value(), self._get_duration_ms()
                )
            except Exception as exc:
                self._on_error(str(exc))
                return
            self._export_mode = "turntable"

            # Use QTimer to capture frames on main thread
            self._export_timer = QTimer(self)
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
            try:
                n_slices = self.exporter.start_slice_sweep_export(
                    filename, self.axis_combo.currentIndex(), self._get_duration_ms()
                )
            except Exception as exc:
                self._on_error(str(exc))
                return
            self.progress.setMaximum(n_slices)
            self._export_mode = "sweep"

            # Use QTimer to capture frames on main thread
            self._export_timer = QTimer(self)
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
            self.exporter.cancel_export()
            self.progress.setVisible(False)
            self.status.setText(f"Error: {msg}")
            self._set_buttons_enabled(True)

        def _set_buttons_enabled(self, enabled: bool):
            self.gif_btn.setEnabled(enabled)
            self.mp4_btn.setEnabled(enabled)
            self.sweep_gif_btn.setEnabled(enabled)
            self.sweep_mp4_btn.setEnabled(enabled)

        def stop_export(self):
            if self._export_timer is not None:
                self._export_timer.stop()
            self.exporter.cancel_export()

        def closeEvent(self, event):
            self.stop_export()
            super().closeEvent(event)

    widget = AnimationWidget()
    dock = viewer.window.add_dock_widget(widget, name="Animation", area="right")
    return dock


def _add_volume_layers(viewer, ht_data, scale, registrations, rendering, metadata):
    """Attach native arrays with complete shared affines, never resampling FL."""
    geometry = VolumeGeometry(ht_data.shape, scale)
    ht_affine = np.diag((*scale, 1.0))
    ht_layer = viewer.add_image(
        ht_data, name="RI", rgb=False, colormap="gray",
        contrast_limits=_display_limits(ht_data), affine=ht_affine,
        blending="translucent", opacity=0.9, rendering=rendering,
        interpolation2d="linear", interpolation3d="linear", metadata=dict(metadata),
    )
    geometry.add_layer(ht_layer, ht_affine)
    if rendering == "attenuated_mip":
        ht_layer.attenuation = 0.5
    colormaps = ("green", "magenta", "cyan", "yellow", "red", "blue")
    for index, (channel, registration) in enumerate(registrations.items()):
        affine = registration.voxel_to_world
        layer = viewer.add_image(
            registration.data, name=channel, rgb=False,
            colormap=colormaps[index % len(colormaps)],
            contrast_limits=_display_limits(registration.data, (5, 99.5), positive=True),
            affine=affine, blending="additive", opacity=0.8, rendering=rendering,
            interpolation2d="linear", interpolation3d="linear",
            metadata={**metadata, "channel": channel,
                      "voxel_to_world_um": affine.tolist()},
        )
        geometry.add_layer(layer, affine, fluorescence=True)
    return geometry


def view_3d(
    tcf_path: str | Path,
    show_slices: bool = False,
    rendering: str = "mip",
    screenshot: str | Path | None = None,
    z_offset_mode: str = "auto",
    *,
    timepoint: int = 0,
    fl_channel: str | None = None,
    registration_path: str | Path | None = None,
) -> None:
    """Open one acquisition with calibrated native HT/FL layers in napari.

    None for fl_channel shows every available channel; an explicit name selects
    only that channel. Saved alignment requires an explicit channel and supplies
    its base Z mode and residual translation. Manual shifts are display-only.
    Input selection and registration are validated before GUI creation.
    """
    if isinstance(timepoint, (bool, np.bool_)) or not isinstance(timepoint, (int, np.integer)) or timepoint < 0:
        raise ValueError("timepoint must be a non-negative integer")
    if rendering not in ("mip", "attenuated_mip", "minip", "average"):
        raise ValueError("rendering must be mip, attenuated_mip, minip or average")
    if z_offset_mode not in Z_OFFSET_MODES:
        raise ValueError(f"z_offset_mode must be one of {Z_OFFSET_MODES}")
    if fl_channel is not None and (not isinstance(fl_channel, str) or not fl_channel.strip()):
        raise ValueError("fl_channel must be a non-empty channel name")
    if registration_path is not None and fl_channel is None:
        raise ValueError("registration_path requires an explicit fl_channel")

    from tomocube.core.file import TCFFileLoader

    tcf_path = Path(tcf_path)
    with TCFFileLoader(tcf_path) as loader:
        loader.load_timepoint(timepoint, fl_channels=[fl_channel] if fl_channel is not None else None)
        ht_data = loader.data_3d
        scale = _get_voxel_scale(loader)
        channels = [fl_channel] if fl_channel is not None else list(loader.fl_data)
        translation_um = (0.0, 0.0, 0.0)
        if registration_path is not None:
            from tomocube.processing.alignment import load_alignment
            alignment = load_alignment(registration_path, loader, fl_channel)
            z_offset_mode = alignment.z_offset_mode
            translation_um = alignment.translation_um
        registrations = {
            channel: FluorescenceRegistration(
                loader.fl_data[channel], ht_data.shape, loader.reg_params,
                channel=channel, z_offset_mode=z_offset_mode,
                translation_um=translation_um,
            )
            for channel in channels
        }
        metadata = {"source": str(tcf_path.resolve()), "timepoint": loader.current_timepoint,
                    "z_offset_mode": z_offset_mode}
        if registration_path is not None:
            metadata["registration_path"] = str(Path(registration_path).resolve())
        acquisition = loader.current_timepoint
    # All native arrays are now resident. Release the source HDF5 file before
    # entering the GUI, including when optional imports or window setup fail.
    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "napari is required for 3D viewing. Install with:\n"
            "  pip install 'tomocube-tools[3d]'"
        ) from exc

    viewer = napari.Viewer(title=f"TCF 3D: {tcf_path.stem} | acquisition {acquisition}")
    animation_widget = None
    try:
        geometry = _add_volume_layers(viewer, ht_data, scale, registrations, rendering, metadata)
        constrain_native_slicing(viewer)
        viewer.canvas.overlays.scale_bar.visible = True
        viewer.canvas.overlays.scale_bar.unit = "µm"
        viewer.canvas.overlays.scale_bar.font_size = 14
        viewer.dims.axis_labels = ("Z (µm)", "Y (µm)", "X (µm)")

        viewer.window._qt_viewer.dockLayerList.setVisible(False)
        viewer.window._qt_viewer.dockLayerControls.setVisible(False)
        camera_dock = _create_camera_controls(viewer)
        crop_dock = _create_crop_widget(viewer, geometry)
        layers_dock = _create_layer_controls(viewer)
        histogram_dock = _create_histogram_widget(viewer)
        fl_dock = _create_fl_z_offset_widget(viewer, geometry) if registrations else None
        animation_dock = _create_animation_widget(viewer, tcf_path.parent)
        animation_widget = animation_dock.widget()
        main_window = viewer.window._qt_window
        main_window.tabifyDockWidget(camera_dock, crop_dock)
        camera_dock.raise_()
        docks = [layers_dock, histogram_dock]
        if fl_dock is not None:
            docks.append(fl_dock)
        docks.append(animation_dock)
        for previous, current in zip(docks, docks[1:]):
            main_window.tabifyDockWidget(previous, current)
        layers_dock.raise_()

        viewer.dims.ndisplay = 2 if show_slices else 3
        if show_slices:
            viewer.dims.set_point(0, (ht_data.shape[0] - 1) * scale[0] / 2)
        else:
            viewer.scene.camera.angles = (0, -30, 45)
            viewer.fit_to_view()
        if screenshot is not None:
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()
            from tomocube.processing.outputs import atomic_output
            sources = [tcf_path]
            if registration_path is not None:
                sources.append(registration_path)
            with atomic_output(screenshot, sources=sources) as temporary:
                viewer.screenshot(str(temporary))
        napari.run()
    finally:
        try:
            if animation_widget is not None:
                animation_widget.stop_export()
        finally:
            viewer.close()
