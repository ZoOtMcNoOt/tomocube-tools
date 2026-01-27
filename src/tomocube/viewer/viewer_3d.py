"""
3D Volume Viewer for Tomocube TCF files.

Provides interactive 3D visualization using napari with:
- Volume rendering (MIP, attenuated, etc.)
- Multi-channel fluorescence overlay
- XYZ range sliders for sub-volume cropping
- Layer controls with background removal
- Camera presets for different viewing angles
- Scale bar with physical units
- Screenshot export

Usage:
    python -m tomocube view3d sample.TCF
    python -m tomocube view3d sample.TCF --slices
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tomocube.core.file import TCFFileLoader

logger = logging.getLogger(__name__)


def _get_voxel_scale(loader: TCFFileLoader) -> tuple[float, float, float]:
    """Extract voxel scale from TCF metadata. Returns (z, y, x) in µm."""
    try:
        if hasattr(loader, 'metadata') and loader.metadata:
            meta = loader.metadata
            z_um = meta.get('z_step_um', meta.get('zStep', 0.5))
            xy_um = meta.get('xy_pixel_um', meta.get('pixelSize', 0.1))
            return (float(z_um), float(xy_um), float(xy_um))
    except Exception:
        pass
    return (0.5, 0.1, 0.1)  # Default for Tomocube HT-2H


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
            layout.setContentsMargins(4, 4, 4, 8)
            layout.setSpacing(6)
            
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
            main_layout.setSpacing(8)
            main_layout.setContentsMargins(8, 8, 8, 8)
            
            # Title
            title = QLabel("<b>Layers</b>")
            title.setStyleSheet("font-size: 13px;")
            main_layout.addWidget(title)
            
            desc = QLabel("Visibility, color, opacity, threshold, and blending.")
            desc.setStyleSheet("color: #888; font-size: 10px;")
            desc.setWordWrap(True)
            main_layout.addWidget(desc)
            
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
            self.layers_layout.setSpacing(8)
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
    viewer.window.add_dock_widget(widget, name="Layers", area="right")
    return widget


def _create_camera_controls(viewer):
    """Create camera preset controls."""
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout, QSizePolicy
    )
    
    class CameraWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.default_zoom = 0.8
            self.default_angles = (0, -30, 45)
            
            layout = QVBoxLayout(self)
            layout.setSpacing(8)
            layout.setContentsMargins(8, 8, 8, 8)
            
            title = QLabel("<b>Camera</b>")
            title.setStyleSheet("font-size: 13px;")
            layout.addWidget(title)
            
            desc = QLabel("View angle presets and zoom controls.")
            desc.setStyleSheet("color: #888; font-size: 10px;")
            layout.addWidget(desc)
            
            # View presets in grid
            grid = QGridLayout()
            grid.setSpacing(4)
            
            presets = [
                ("Top", (0, 0, 90)),
                ("Bottom", (0, 180, 90)),
                ("Front", (0, -90, 0)),
                ("Back", (0, 90, 0)),
                ("Left", (90, -90, 0)),
                ("Right", (-90, -90, 0)),
            ]
            
            for i, (name, angles) in enumerate(presets):
                btn = QPushButton(name)
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                btn.setToolTip(f"View from {name.lower()}")
                btn.clicked.connect(lambda checked, a=angles: self._set_view(a))
                grid.addWidget(btn, i // 3, i % 3)
            
            layout.addLayout(grid)
            
            # Isometric and reset
            iso_row = QHBoxLayout()
            
            iso_btn = QPushButton("Isometric")
            iso_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            iso_btn.setToolTip("45-degree isometric view")
            iso_btn.clicked.connect(lambda: self._set_view((0, -30, 45)))
            iso_row.addWidget(iso_btn)
            
            reset_btn = QPushButton("Reset")
            reset_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            reset_btn.setToolTip("Reset camera to default")
            reset_btn.clicked.connect(self._reset_camera)
            iso_row.addWidget(reset_btn)
            
            layout.addLayout(iso_row)
            
            # Zoom controls
            zoom_row = QHBoxLayout()
            zoom_lbl = QLabel("Zoom")
            zoom_lbl.setFixedWidth(40)
            zoom_row.addWidget(zoom_lbl)
            
            zoom_out = QPushButton("-")
            zoom_out.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            zoom_out.clicked.connect(lambda: self._zoom(0.8))
            zoom_row.addWidget(zoom_out)
            
            zoom_in = QPushButton("+")
            zoom_in.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            zoom_in.clicked.connect(lambda: self._zoom(1.25))
            zoom_row.addWidget(zoom_in)
            
            zoom_fit = QPushButton("Fit")
            zoom_fit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            zoom_fit.clicked.connect(lambda: viewer.reset_view())
            zoom_row.addWidget(zoom_fit)
            
            layout.addLayout(zoom_row)
            layout.addStretch()
        
        def _set_view(self, angles):
            viewer.dims.ndisplay = 3
            viewer.camera.angles = angles
        
        def _zoom(self, factor):
            viewer.camera.zoom *= factor
        
        def _reset_camera(self):
            viewer.dims.ndisplay = 3
            viewer.camera.angles = self.default_angles
            viewer.camera.zoom = self.default_zoom
            viewer.camera.center = (0, 0, 0)
    
    widget = CameraWidget()
    viewer.window.add_dock_widget(widget, name="Camera", area="right")
    return widget


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
            self.full_data = {}
            
            layout = QVBoxLayout(self)
            layout.setSpacing(8)
            layout.setContentsMargins(8, 8, 8, 8)
            
            # Title
            title = QLabel("<b>Volume Crop</b>")
            title.setStyleSheet("font-size: 13px;")
            layout.addWidget(title)
            
            # Description
            desc = QLabel("Drag slider handles to crop each axis.")
            desc.setStyleSheet("color: #888; font-size: 10px;")
            layout.addWidget(desc)
            
            layout.addSpacing(4)
            
            z, y, x = ht_data.shape
            
            # Store original data first
            for layer in viewer.layers:
                if hasattr(layer, 'data') and isinstance(layer.data, np.ndarray):
                    if layer.data.ndim == 3:
                        self.full_data[layer.name] = layer.data.copy()
            
            # Create range sliders
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
            
            for layer in viewer.layers:
                if layer.name in self.full_data:
                    full = self.full_data[layer.name]
                    cropped = full[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
                    layer.data = cropped
                    layer.translate = (z_min * scale[0], y_min * scale[1], x_min * scale[2])
        
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
            
            # Restore full data
            for layer in viewer.layers:
                if layer.name in self.full_data:
                    layer.data = self.full_data[layer.name].copy()
                    layer.translate = (0, 0, 0)
    
    crop_widget = CropWidget()
    viewer.window.add_dock_widget(crop_widget, name="Volume Crop", area="right")
    return crop_widget


def view_3d(
    tcf_path: str | Path,
    show_slices: bool = False,
    rendering: str = "mip",
    screenshot: str | None = None,
) -> None:
    """
    Open interactive 3D viewer for a TCF file.
    
    Args:
        tcf_path: Path to TCF file
        show_slices: Start in 2D slice mode (default: 3D volume)
        rendering: Volume rendering - "mip", "attenuated_mip", "minip", "average"
        screenshot: Path to save screenshot
    """
    try:
        import napari
    except ImportError:
        raise ImportError(
            "napari is required for 3D viewing. Install with:\n"
            "  pip install 'tomocube-tools[3d]'"
        )
    
    from tomocube.core.file import TCFFileLoader
    from tomocube.processing.registration import register_fl_to_ht
    
    tcf_path = Path(tcf_path)
    
    with TCFFileLoader(str(tcf_path)) as loader:
        loader.load_timepoint(0)
        
        ht_data = loader.data_3d.copy()
        scale = _get_voxel_scale(loader)
        title = f"TCF 3D: {tcf_path.stem}"
        
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
        
        if rendering == "attenuated_mip":
            ht_layer.attenuation = 0.5
        
        # Add FL channels
        if loader.has_fluorescence:
            colormaps = ["green", "magenta", "cyan", "yellow", "red", "blue"]
            
            for idx, (ch_name, fl_data) in enumerate(loader.fl_data.items()):
                fl_registered = register_fl_to_ht(fl_data, ht_data.shape, loader.reg_params)
                
                fl_nonzero = fl_registered[fl_registered > 0]
                if len(fl_nonzero) > 0:
                    fl_p1, fl_p99 = np.percentile(fl_nonzero, [5, 99.5])
                else:
                    fl_p1, fl_p99 = 0, 1
                
                viewer.add_image(
                    fl_registered.copy(),
                    name=ch_name,
                    colormap=colormaps[idx % len(colormaps)],
                    contrast_limits=(fl_p1, fl_p99),
                    scale=scale,
                    blending="additive",
                    opacity=0.8,
                    rendering=rendering,
                )
        
        # Scale bar
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "µm"
        viewer.scale_bar.font_size = 14
        
        # Hide napari's built-in layer list - we have our own
        viewer.window._qt_viewer.dockLayerList.setVisible(False)
        viewer.window._qt_viewer.dockLayerControls.setVisible(False)
        
        # Add control widgets (order determines stacking in right panel)
        _create_camera_controls(viewer)
        _create_layer_controls(viewer)
        _create_crop_widget(viewer, ht_data, scale)
        
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
        print(f"\nVolume: {x} x {y} x {z} voxels")
        print(f"Physical size: {x*scale[2]:.1f} x {y*scale[1]:.1f} x {z*scale[0]:.1f} um")
        print(f"\nPress 2/3 to toggle between slice and 3D view.")
        
        if screenshot:
            import time
            time.sleep(0.5)
            viewer.screenshot(screenshot)
            print(f"  Screenshot saved: {screenshot}")
        
        napari.run()
