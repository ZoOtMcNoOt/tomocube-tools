"""Optional real Qt dock tests; these do not stand in for OpenGL rendering QA."""
import os

import h5py
import numpy as np
import pytest

from tomocube.core.file import TCFFileLoader
from tomocube.processing.registration import FluorescenceRegistration
from tomocube.viewer import viewer_3d


# napari 0.9.1 still uses this deprecated pydantic serialization hook during
# class construction. Keep every other warning an error, including our Qt/UI.
pytestmark = pytest.mark.filterwarnings(
    "ignore:`json_encoders` is deprecated.*:DeprecationWarning:pydantic\\._internal\\._generate_schema"
)


@pytest.fixture
def qt_scene(monkeypatch, make_tcf):
    pytest.importorskip("PyQt6")
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("QT_API", "pyqt6")
    if os.name == "nt":
        monkeypatch.setenv("QT_QPA_FONTDIR", "C:/Windows/Fonts")
    from qtpy.QtCore import Qt, QEvent
    from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow
    from napari.components import ViewerModel

    app = QApplication.instance() or QApplication([])
    qt_errors = []
    monkeypatch.setattr("sys.excepthook", lambda typ, exc, traceback: qt_errors.append(exc))

    class DockWindow:
        def __init__(self):
            self._qt_window = QMainWindow()
            self.docks = {}

        def add_dock_widget(self, widget, *, name, area):
            dock = QDockWidget(name, self._qt_window)
            dock.setWidget(widget)
            self._qt_window.addDockWidget(
                Qt.LeftDockWidgetArea if area == "left" else Qt.RightDockWidgetArea, dock)
            self.docks[name] = dock
            return dock

    class Scene:
        def __init__(self):
            self.model = ViewerModel(ndisplay=3)
            self.layers = self.model.layers
            self.dims = self.model.dims
            self.scene = self.model.scene
            self.canvas = self.model.canvas
            self.window = DockWindow()

        def add_image(self, data, **kwargs):
            return self.model.add_image(data, **kwargs)

        def fit_to_view(self):
            self.model.fit_to_view()

    path = make_tcf(fluorescence=True)
    with h5py.File(path, "a") as file:
        file["Info/MetaData/FL/Registration"].attrs["Rotation"] = [0.3]
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        data = loader.data_3d
        scale = viewer_3d._get_voxel_scale(loader)
        registrations = {
            channel: FluorescenceRegistration(fl, data.shape, loader.reg_params, channel)
            for channel, fl in loader.fl_data.items()
        }
    scene = Scene()
    geometry = viewer_3d._add_volume_layers(scene, data, scale, registrations, "mip", {})
    yield scene, geometry, app
    animation = scene.window.docks.get("Animation")
    if animation is not None:
        animation.widget().stop_export()
    scene.window._qt_window.close()
    scene.window._qt_window.deleteLater()
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    assert not qt_errors, f"Qt callback errors: {qt_errors}"


def test_real_qt_docks_crop_shift_histogram_and_reset(qt_scene, tmp_path):
    scene, geometry, app = qt_scene
    camera_dock = viewer_3d._create_camera_controls(scene)
    crop_dock = viewer_3d._create_crop_widget(scene, geometry)
    layers_dock = viewer_3d._create_layer_controls(scene)
    histogram_dock = viewer_3d._create_histogram_widget(scene)
    offset_dock = viewer_3d._create_fl_z_offset_widget(scene, geometry)
    animation_dock = viewer_3d._create_animation_widget(scene, tmp_path)
    scene.window._qt_window.tabifyDockWidget(camera_dock, crop_dock)
    scene.window._qt_window.tabifyDockWidget(layers_dock, histogram_dock)
    scene.window._qt_window.tabifyDockWidget(histogram_dock, offset_dock)
    scene.window._qt_window.tabifyDockWidget(offset_dock, animation_dock)
    app.processEvents()

    crop, controls = crop_dock.widget(), layers_dock.widget()
    histogram, offset = histogram_dock.widget(), offset_dock.widget()
    layer = scene.layers["CH1"]
    native_data = layer.data
    base_affine = layer.affine.affine_matrix.copy()
    assert len(controls.layer_controls) == 3
    assert controls.layer_controls[2].original_data is native_data
    controls.layer_controls[2].visible_cb.setChecked(False)
    crop.sliders[0].setValue((1, 2))
    crop.sliders[1].setValue((1, 3))
    offset.offset.setValue(0.75)
    app.processEvents()
    assert len(layer.experimental_clipping_planes) == 6
    assert layer.data is native_data
    assert not layer.visible
    expected = base_affine.copy()
    expected[0, 3] += 0.75
    np.testing.assert_allclose(layer.affine.affine_matrix, expected)

    histogram.layer_combo.setCurrentText("CH1")
    histogram._apply_percentile(5, 99.5)
    assert histogram.current_layer is layer
    assert histogram.hist_plot.getData()[0].size == 200
    low, high = layer.contrast_limits
    histogram.low_line.setValue(low + (high - low) / 10)
    app.processEvents()
    assert layer.contrast_limits[0] > low

    controls.layer_controls[2].opacity_slider.setValue(40)
    controls.layer_controls[2].cmap_combo.setCurrentText("magenta")
    assert layer.opacity == 0.4
    assert layer.colormap.name == "magenta"
    controls.layer_controls[2].reset()
    assert layer.opacity == 0.8
    crop._reset()
    assert len(layer.experimental_clipping_planes) == 0
    np.testing.assert_allclose(layer.affine.affine_matrix, expected)
    assert not layer.visible
    offset.offset.setValue(0)
    np.testing.assert_allclose(layer.affine.affine_matrix, base_affine)

    scene.dims.ndisplay = 2
    assert all(not slider.isEnabled() for slider in crop.sliders)
    scene.dims.ndisplay = 3
    assert all(slider.isEnabled() for slider in crop.sliders)
    camera_dock.widget()._set_view((0, 0, 90))
    np.testing.assert_allclose(scene.scene.camera.angles, (0, 0, 90))
    assert not scene.window._qt_window.isVisible()


def test_real_qt_animation_guard_keeps_buttons_usable_and_stops_timer(qt_scene, tmp_path):
    scene, geometry, app = qt_scene
    widget = viewer_3d._create_animation_widget(scene, tmp_path).widget()
    widget.axis_combo.setCurrentIndex(1)
    widget.sweep_gif_btn.click()
    app.processEvents()
    assert "calibrated orthogonal planes" in widget.status.text()
    assert widget.sweep_gif_btn.isEnabled()
    assert widget.gif_btn.isEnabled()
    assert not widget.exporter._is_exporting
    assert not list(tmp_path.glob("*.gif"))

    # Start and close before a capture timer fires. No OpenGL screenshot is
    # mocked: this covers the real Qt timer and cancellation boundary only.
    widget.gif_btn.click()
    assert widget._export_timer.isActive()
    assert widget.exporter._is_exporting
    widget.close()
    app.processEvents()
    assert not widget._export_timer.isActive()
    assert not widget.exporter._is_exporting
    assert not widget.exporter._frames
    assert not list(tmp_path.glob("*.gif"))
