"""Native 3D placement, clipping, selection and animation without an OpenGL GUI."""
from types import SimpleNamespace
import sys

import h5py
import numpy as np
from PIL import Image as PillowImage
import pytest

from tomocube.core.file import TCFFileLoader
from tomocube.core.types import RegistrationParams
from tomocube.processing.registration import FluorescenceRegistration
from tomocube.viewer import viewer_3d
from tomocube.viewer.volume_geometry import (
    VolumeGeometry, constrain_native_slicing, native_slice_conflicts, world_clip_planes,
)


# Only the optional napari dependency's known pydantic deprecation is exempt.
pytestmark = pytest.mark.filterwarnings(
    "ignore:`json_encoders` is deprecated.*:DeprecationWarning:pydantic\\._internal\\._generate_schema"
)


class Layer:
    def __init__(self, data, **kwargs):
        self.data = data
        self.visible = True
        self.metadata = {}
        self.__dict__.update(kwargs)
        self.affine = np.asarray(self.affine)
        self.experimental_clipping_planes = []

    def data_to_world(self, point):
        return (self.affine @ np.r_[point, 1])[:3]

    def world_to_data(self, point):
        return np.linalg.solve(self.affine, np.r_[point, 1])[:3]


class Dims:
    def __init__(self):
        self.ndisplay = 2
        self.order = (0, 1, 2)
        self.point = (0, 0, 0)
        self.range = ((-1, 3, 2), (0, 0, 0.5), (0, 2, 0.25))
        self.nsteps = (3, 1, 9)

    def set_point(self, axis, point):
        points = list(self.point)
        points[axis] = point
        self.point = tuple(points)


class Viewer:
    def __init__(self, **kwargs):
        self.layers = []
        self.dims = Dims()
        self.scene = SimpleNamespace(camera=SimpleNamespace(angles=(10, 20, 30), zoom=1, center=(0, 0, 0)))
        self.canvas = SimpleNamespace(overlays=SimpleNamespace(scale_bar=SimpleNamespace()))
        dock = SimpleNamespace(setVisible=lambda visible: None)
        self.window = SimpleNamespace(
            _qt_viewer=SimpleNamespace(dockLayerList=dock, dockLayerControls=dock),
            _qt_window=SimpleNamespace(tabifyDockWidget=lambda a, b: None),
        )
        self.closed = 0
        self.screenshots = []

    def add_image(self, data, **kwargs):
        layer = Layer(data, **kwargs)
        self.layers.append(layer)
        return layer

    def fit_to_view(self):
        pass

    def close(self):
        self.closed += 1

    def screenshot(self, path=None, **kwargs):
        self.screenshots.append(self.dims.point)
        frame = np.full((5, 7, 4), len(self.screenshots) * 30 % 256, dtype=np.uint8)
        frame[..., 3] = 255
        if path is not None:
            PillowImage.fromarray(frame).save(path)
        return frame


@pytest.fixture
def fake_gui(monkeypatch):
    created = []

    def create(**kwargs):
        viewer = Viewer(**kwargs)
        created.append(viewer)
        return viewer

    napari = SimpleNamespace(Viewer=create, run=lambda: None)
    monkeypatch.setitem(sys.modules, "napari", napari)
    monkeypatch.setitem(sys.modules, "qtpy.QtWidgets", SimpleNamespace(
        QApplication=SimpleNamespace(processEvents=lambda: None)))
    for name in ("_create_camera_controls", "_create_crop_widget", "_create_layer_controls",
                 "_create_histogram_widget", "_create_fl_z_offset_widget", "_create_animation_widget"):
        monkeypatch.setattr(viewer_3d, name, lambda *args: SimpleNamespace(
            raise_=lambda: None, widget=lambda: SimpleNamespace(stop_export=lambda: None)))
    monkeypatch.setattr(viewer_3d, "constrain_native_slicing", lambda viewer: None)
    return created, napari


def parameters():
    return RegistrationParams(
        ht_res_z=1, ht_res_y=0.25, ht_res_x=0.5,
        fl_res_z=2, fl_res_y=0.5, fl_res_x=0.25,
        rotation=np.pi / 2, translation_y=0.25, translation_x=0.75,
        channel_offsets_z={"CH0": 1, "CH1": 2},
    )


@pytest.mark.parametrize("mode", ["start", "center", "auto"])
@pytest.mark.parametrize("shape", [(3, 4, 5), (1, 1, 1), (1, 4, 3)])
def test_native_placement_uses_physical_centers_and_channel_offsets(mode, shape):
    params = parameters()
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    ht_shape = (7, 9, 11)
    ht_spacing = np.array((1, 0.25, 0.5))
    fl_spacing = np.array((2, 0.5, 0.25))
    residual = np.array((1, 0.25, -0.5))
    registrations = {channel: FluorescenceRegistration(
        data, ht_shape, params, channel, mode, translation_um=residual)
        for channel in ("CH0", "CH1")}
    viewer = Viewer()
    viewer_3d._add_volume_layers(viewer, np.zeros(ht_shape), ht_spacing, registrations, "mip", {})
    for layer, (channel, registration) in zip(viewer.layers[1:], registrations.items()):
        assert layer.data is data
        assert layer.data.dtype == np.uint16
        assert layer.rgb is False
        assert layer.interpolation3d == "linear"
        for point in (np.zeros(3), np.array(shape) - 1, (np.array(shape) - 1) / 2):
            # Independent forward physical-coordinate calculation.
            centered = (point[1:] - (np.array(shape[1:]) - 1) / 2) * fl_spacing[1:]
            expected_xy = np.array((-centered[1], centered[0]))
            expected_xy += (np.array(ht_shape[1:]) - 1) * ht_spacing[1:] / 2
            expected_xy += (params.translation_y, params.translation_x)
            if mode == "start":
                first_z = params.get_offset_z(channel)
            elif mode == "center":
                first_z = params.get_offset_z(channel) - (shape[0] - 1) * fl_spacing[0] / 2
            else:
                profile = data.sum(axis=(1, 2), dtype=float)
                center_z = (np.arange(shape[0]) @ profile / profile.sum()
                            if profile.sum() else (shape[0] - 1) / 2)
                first_z = (ht_shape[0] - 1) / 2 - center_z * fl_spacing[0]
            expected = np.r_[first_z + point[0] * fl_spacing[0], expected_xy] + residual
            np.testing.assert_allclose(layer.data_to_world(point), expected, atol=1e-12)


def test_3d_fiducial_agrees_with_export_sampling():
    data = np.arange(60, dtype=np.uint16).reshape(3, 4, 5)
    registration = FluorescenceRegistration(
        data, (7, 9, 11), parameters(), "CH0", "start", translation_um=(1, 0.25, -0.5))
    viewer = Viewer()
    viewer_3d._add_volume_layers(viewer, np.zeros((7, 9, 11)), (1, 0.25, 0.5),
                                 {"CH0": registration}, "mip", {})
    point = (1, 1, 3)
    expected_ht_index = np.array((4, 5, 5))
    np.testing.assert_allclose(viewer.layers[1].data_to_world(point),
                               expected_ht_index * (1, 0.25, 0.5))
    sample, overlap = registration.sample_plane(0, 4)
    assert overlap
    assert sample[5, 5] == data[point]


def inside_planes(point, planes):
    return all(np.dot(np.asarray(point) - p["position"], p["normal"]) >= -1e-10 for p in planes)


def test_rotated_crop_planes_match_world_box():
    registration = FluorescenceRegistration(
        np.zeros((3, 4, 5), dtype=np.uint16), (7, 9, 11), parameters(), "CH1", "start")
    affine = registration.voxel_to_world
    bounds = np.array(((1.5, 4.5), (0.375, 1.625), (0.75, 4.25)))
    planes = world_clip_planes(affine, bounds)
    points = np.random.default_rng(53).uniform(-2, 10, (100, 3))
    for point in points:
        world = (affine @ np.r_[point, 1])[:3]
        assert inside_planes(point, planes) == bool(
            ((world >= bounds[:, 0]) & (world <= bounds[:, 1])).all())


def test_crop_and_z_adjustment_commute_without_copying_or_changing_visibility():
    data = np.arange(60, dtype=np.uint16).reshape(3, 4, 5)
    registration = FluorescenceRegistration(data, (7, 9, 11), parameters(), "CH1", "start")
    viewer = Viewer()
    geometry = viewer_3d._add_volume_layers(
        viewer, np.zeros((7, 9, 11)), (1, 0.25, 0.5), {"CH1": registration}, "mip", {})
    layer = viewer.layers[1]
    layer.visible = False
    initial = layer.affine.copy()
    geometry.set_crop(((1, 5), (2, 6), (2, 8)))
    geometry.set_fl_z_offset(0.75)
    expected = initial.copy()
    expected[0, 3] += 0.75
    np.testing.assert_allclose(layer.affine, expected)
    first_planes = layer.experimental_clipping_planes
    geometry.set_fl_z_offset(0)
    geometry.set_fl_z_offset(0.75)
    geometry.set_crop(((1, 5), (2, 6), (2, 8)))
    for before, after in zip(first_planes, layer.experimental_clipping_planes):
        np.testing.assert_allclose(before["position"], after["position"])
        np.testing.assert_allclose(before["normal"], after["normal"])
    geometry.set_crop(((0, 6), (0, 8), (0, 10)))
    assert layer.experimental_clipping_planes == []
    np.testing.assert_allclose(layer.affine, expected)
    assert layer.data is data
    assert layer.visible is False
    np.testing.assert_array_equal(data.ravel(), np.arange(60))
    geometry.set_fl_z_offset(0)
    np.testing.assert_allclose(layer.affine, initial)


def test_crop_half_voxel_edges_retain_final_and_singleton_centers():
    layer = Layer(np.zeros((1, 3, 2)), affine=np.diag((2, 0.5, 1, 1)))
    geometry = VolumeGeometry((1, 3, 2), (2, 0.5, 1))
    geometry.add_layer(layer, layer.affine)
    geometry.set_crop(((0, 0), (2, 2), (0, 1)))
    assert inside_planes((0, 2, 1), layer.experimental_clipping_planes)
    assert not inside_planes((0, 1, 1), layer.experimental_clipping_planes)
    assert not inside_planes((-0.51, 2, 1), layer.experimental_clipping_planes)


@pytest.mark.parametrize("ranges", [((0, 9), (0, 2), (0, 3)), ((0, 1), (2, 0), (0, 3)),
                                    ((0, 1), (0, 2)), ((0.1, 1), (0, 2), (0, 3))])
def test_invalid_crop_keeps_geometry(ranges):
    geometry = VolumeGeometry((2, 3, 4), (1, 1, 1))
    initial = geometry.ranges
    with pytest.raises(ValueError, match="Crop ranges"):
        geometry.set_crop(ranges)
    assert geometry.ranges == initial


@pytest.mark.parametrize("options,match", [
    ({"timepoint": -1}, "timepoint"), ({"timepoint": True}, "timepoint"),
    ({"timepoint": 1.5}, "timepoint"), ({"rendering": "bad"}, "rendering"),
    ({"z_offset_mode": "bad"}, "z_offset_mode"), ({"fl_channel": ""}, "fl_channel"),
    ({"registration_path": "x.json"}, "explicit fl_channel"),
])
def test_invalid_options_rejected_before_file_or_gui(tmp_path, options, match):
    with pytest.raises(ValueError, match=match):
        viewer_3d.view_3d(tmp_path / "nonexistent.TCF", **options)


def test_timepoint_channel_selection_and_world_slice_center(make_tcf, fake_gui):
    path = make_tcf(fluorescence=True, timepoints={
        "2": np.full((3, 5, 3), 13400, dtype=np.uint16),
        "10": np.full((5, 6, 4), 13500, dtype=np.uint16),
    })
    viewer_3d.view_3d(path, timepoint=1, fl_channel="CH1", show_slices=True, z_offset_mode="start")
    viewer = fake_gui[0][0]
    assert [layer.name for layer in viewer.layers] == ["RI", "CH1"]
    assert viewer.layers[0].data.shape == (5, 6, 4)
    np.testing.assert_allclose(viewer.layers[0].data, 1.35)
    assert viewer.layers[0].metadata["timepoint"] == "10"
    assert viewer.layers[1].data.dtype == np.uint16
    assert viewer.layers[1].affine[0, 3] == 1
    assert viewer.dims.point[0] == 3  # geometric center in micrometers
    assert viewer.closed == 1


def test_missing_channel_rejected_before_optional_gui(make_tcf, monkeypatch):
    path = make_tcf(fluorescence=True)
    monkeypatch.setitem(sys.modules, "napari", None)
    with pytest.raises(ValueError, match="Unknown fluorescence channel"):
        viewer_3d.view_3d(path, fl_channel="CH9")


def test_explicit_channel_does_not_read_unselected_fluorescence(make_tcf, fake_gui, monkeypatch):
    path = make_tcf(fluorescence=True)
    reads = []
    original = h5py.Dataset.__array__

    def record(dataset, *args, **kwargs):
        reads.append(dataset.name)
        return original(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__array__", record)
    viewer_3d.view_3d(path, fl_channel="CH1")
    assert "/Data/3DFL/CH1/000000" in reads
    assert not any("/Data/3DFL/CH0/" in name for name in reads)


def test_failed_gui_setup_closes_viewer_and_loader(make_tcf, fake_gui, monkeypatch):
    path = make_tcf(fluorescence=True)
    loaders = []

    class TrackingLoader(TCFFileLoader):
        def __init__(self, path):
            super().__init__(path)
            loaders.append(self)

    def fail(*args):
        assert loaders[0]._file is None
        raise RuntimeError("dock setup failed")

    monkeypatch.setattr("tomocube.core.file.TCFFileLoader", TrackingLoader)
    monkeypatch.setattr(viewer_3d, "_create_camera_controls", fail)
    with pytest.raises(RuntimeError, match="dock setup failed"):
        viewer_3d.view_3d(path)
    assert fake_gui[0][0].closed == 1
    assert loaders[0]._file is None


def test_invalid_fluorescence_rejected_before_gui(make_tcf, fake_gui):
    path = make_tcf(fluorescence=True)
    with h5py.File(path, "a") as file:
        group = file["Data/3DFL/CH1"]
        del group["000000"]
        group.create_dataset("000000", data=np.full((2, 3, 4), np.nan))
    with pytest.raises(ValueError, match="finite intensities"):
        viewer_3d.view_3d(path, fl_channel="CH1")
    assert fake_gui[0] == []


def test_saved_alignment_applies_only_to_selected_channel(make_tcf, tmp_path, fake_gui):
    from tomocube.processing.alignment import AlignmentResult, save_alignment

    path = make_tcf(fluorescence=True)
    result = AlignmentResult(True, "accepted", (1, -2, 3), 0.2, 0.95, 0.3, 0.9,
                             "center", (1, 1, 1), (5, 5, 5), 0.6, 0.03, 0.5)
    saved = tmp_path / "alignment.json"
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        save_alignment(loader, "CH1", result, saved)
        expected = FluorescenceRegistration(loader.fl_data["CH1"], loader.data_3d.shape,
                                            loader.reg_params, "CH1", "center",
                                            translation_um=(1, -2, 3)).voxel_to_world
    viewer_3d.view_3d(path, fl_channel="CH1", registration_path=saved)
    layers = fake_gui[0][0].layers
    assert len(layers) == 2
    assert layers[1].name == "CH1"
    np.testing.assert_allclose(layers[1].affine, expected)
    assert layers[1].metadata["z_offset_mode"] == "center"
    assert layers[1].metadata["registration_path"] == str(saved.resolve())


def test_screenshot_failure_preserves_previous_file_and_closes_viewer(make_tcf, fake_gui, tmp_path, monkeypatch):
    path = make_tcf()
    output = tmp_path / "view.png"
    output.write_bytes(b"previous screenshot")

    def fail(self, path, **kwargs):
        from pathlib import Path
        Path(path).write_bytes(b"partial")
        raise RuntimeError("capture failed")

    monkeypatch.setattr(Viewer, "screenshot", fail)
    with pytest.raises(RuntimeError, match="capture failed"):
        viewer_3d.view_3d(path, screenshot=output)
    assert output.read_bytes() == b"previous screenshot"
    assert fake_gui[0][0].closed == 1
    assert not list(tmp_path.glob(".view-*"))


def test_screenshot_and_animation_cannot_overwrite_alignment_report(make_tcf, fake_gui, tmp_path):
    from tomocube.processing.alignment import AlignmentResult, save_alignment

    path = make_tcf(fluorescence=True)
    result = AlignmentResult(True, "accepted", (0, 0, 0), 0.9, 0.95, 0.3, 0.9,
                             "start", (1, 1, 1), (5, 5, 5), 0.6, 0.03, 0.5)
    # The sidecar format is JSON even when a caller supplied a media suffix.
    report = tmp_path / "alignment.gif"
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        save_alignment(loader, "CH1", result, report)
    original = report.read_bytes()
    with pytest.raises(ValueError, match="overwrite input"):
        viewer_3d.view_3d(path, fl_channel="CH1", registration_path=report, screenshot=report)
    viewer = fake_gui[0][0]
    assert report.read_bytes() == original
    assert viewer.closed == 1
    exporter = viewer_3d.AnimationExporter(viewer, tmp_path)
    exporter.start_turntable_export(report.name, 1, 100)
    exporter.capture_turntable_frame()
    with pytest.raises(ValueError, match="overwrite input"):
        exporter.finish_export()
    assert report.read_bytes() == original
    assert not exporter._is_exporting


@pytest.mark.parametrize("axis,expected", [(0, [-1, 1, 3]), (1, [0]), (2, np.arange(9) * 0.25)])
def test_animation_sweeps_world_positions_through_last_sample(axis, expected, tmp_path, fake_gui):
    viewer = Viewer()
    exporter = viewer_3d.AnimationExporter(viewer, tmp_path)
    n_frames = exporter.start_slice_sweep_export("sweep.gif", axis, 100)
    assert n_frames == len(expected)
    for index in range(n_frames):
        assert exporter.capture_sweep_frame() == (index + 1, n_frames, index + 1 == n_frames)
    np.testing.assert_allclose([point[axis] for point in viewer.screenshots], expected)
    exporter.cancel_export()
    assert viewer.dims.ndisplay == 2
    assert viewer.dims.point == (0, 0, 0)
    assert viewer.dims.order == (0, 1, 2)


def test_out_of_plane_rotation_rejects_sweep_without_changing_view(tmp_path):
    registration = FluorescenceRegistration(np.zeros((3, 4, 5)), (7, 9, 11), parameters(), "CH1")
    viewer = Viewer()
    viewer_3d._add_volume_layers(viewer, np.zeros((7, 9, 11)), (1, 0.25, 0.5),
                                 {"CH1": registration}, "mip", {})
    assert native_slice_conflicts(viewer, 0) == []
    assert native_slice_conflicts(viewer, 1) == ["CH1"]
    exporter = viewer_3d.AnimationExporter(viewer, tmp_path)
    with pytest.raises(ValueError, match="calibrated orthogonal planes"):
        exporter.start_slice_sweep_export("sweep.gif", 1, 100)
    assert viewer.dims.ndisplay == 2
    assert viewer.dims.order == (0, 1, 2)
    assert not exporter._is_exporting


def test_turntable_gif_retains_canvas_and_millisecond_timing(tmp_path, fake_gui):
    viewer = Viewer()
    exporter = viewer_3d.AnimationExporter(viewer, tmp_path)
    exporter.start_turntable_export("rotation.gif", 3, 150)
    for _ in range(3):
        exporter.capture_turntable_frame()
    output = exporter.finish_export()
    with PillowImage.open(output) as gif:
        assert gif.n_frames == 3
        assert gif.size == (7, 5)
        for frame in range(gif.n_frames):
            gif.seek(frame)
            assert gif.info["duration"] == 150
    assert viewer.scene.camera.angles == (10, 20, 30)
    assert viewer.dims.ndisplay == 2
    assert not exporter._frames
    assert not exporter._is_exporting


def test_failed_animation_encoding_restores_view_and_preserves_output(tmp_path, fake_gui, monkeypatch):
    import imageio.v3

    viewer = Viewer()
    exporter = viewer_3d.AnimationExporter(viewer, tmp_path)
    output = tmp_path / "rotation.gif"
    output.write_bytes(b"previous animation")
    exporter.start_turntable_export(output.name, 1, 100)
    exporter.capture_turntable_frame()

    def fail(*args, **kwargs):
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(imageio.v3, "imwrite", fail)
    with pytest.raises(RuntimeError, match="encoder failed"):
        exporter.finish_export()
    assert output.read_bytes() == b"previous animation"
    assert viewer.scene.camera.angles == (10, 20, 30)
    assert viewer.dims.ndisplay == 2
    assert not exporter._is_exporting


def test_display_statistics_bounded_and_deterministic():
    data = np.arange(2_000_002, dtype=np.uint32).reshape(2, 1_000_001)[:, ::-1]
    sample = viewer_3d._display_sample(data)
    assert sample.size <= 1_000_000
    np.testing.assert_array_equal(sample, viewer_3d._display_sample(data))
    low, high = viewer_3d._display_limits(np.ones((2, 3, 4), dtype=np.uint16))
    assert low < 1 < high


def test_real_napari_layer_uses_full_affine_and_native_clipping():
    napari_layers = pytest.importorskip("napari.layers")
    params = parameters()
    data = np.arange(60, dtype=np.uint16).reshape(3, 4, 5)
    registration = FluorescenceRegistration(data, (7, 9, 11), params, "CH1", "start")
    layer = napari_layers.Image(data, rgb=False, affine=registration.voxel_to_world)
    geometry = VolumeGeometry((7, 9, 11), (1, 0.25, 0.5))
    geometry.add_layer(layer, registration.voxel_to_world, fluorescence=True)
    geometry.set_crop(((1, 5), (2, 6), (2, 8)))
    geometry.set_fl_z_offset(0.75)
    assert layer.data is data
    expected = registration.voxel_to_world @ np.array((1, 1, 3, 1))
    expected[0] += 0.75
    np.testing.assert_allclose(layer.data_to_world((1, 1, 3)), expected[:3])
    assert len(layer.experimental_clipping_planes) == 6
    for plane in layer.experimental_clipping_planes:
        world = np.asarray(layer.data_to_world(plane.position))
        assert any(np.isclose(world[axis], bound) for axis, bounds in enumerate(
            ((0.5, 5.5), (0.375, 1.625), (0.75, 4.25))) for bound in bounds)


def test_real_napari_dims_sweep_count_and_endpoints(tmp_path, monkeypatch):
    components = pytest.importorskip("napari.components")
    monkeypatch.setitem(sys.modules, "qtpy.QtWidgets", SimpleNamespace(
        QApplication=SimpleNamespace(processEvents=lambda: None)))
    viewer = Viewer()
    viewer.dims = components.Dims(ndim=3, range=((-1, 3, 2), (0, 0, 0.5), (0, 2, 0.25)))
    exporter = viewer_3d.AnimationExporter(viewer, tmp_path)
    assert exporter.start_slice_sweep_export("sweep.gif", 0, 100) == 3
    for _ in range(3):
        exporter.capture_sweep_frame()
    np.testing.assert_allclose([point[0] for point in viewer.screenshots], (-1, 1, 3))


def test_real_napari_slice_guard_prevents_incorrect_native_slicing(monkeypatch):
    components = pytest.importorskip("napari.components")
    from napari.utils import notifications

    notices = []
    monkeypatch.setattr(notifications, "show_warning", notices.append)
    viewer = components.ViewerModel(ndisplay=3)
    params = parameters()
    params.rotation = np.pi / 4
    registration = FluorescenceRegistration(np.zeros((3, 4, 3), dtype=np.uint16),
                                            (7, 9, 4), params, "CH1")
    viewer_3d._add_volume_layers(viewer, np.zeros((7, 9, 4)), (1, 0.25, 0.5),
                                 {"CH1": registration}, "mip", {})
    assert all(layer.ndim == 3 and not layer.rgb for layer in viewer.layers)
    constrain_native_slicing(viewer)
    viewer.dims.order = (1, 0, 2)
    with pytest.warns(UserWarning, match="Non-orthogonal slicing"):
        viewer.dims.ndisplay = 2
    assert viewer.dims.order == (0, 1, 2)
    assert len(notices) == 1
    with pytest.warns(UserWarning, match="Non-orthogonal slicing"):
        viewer.dims.order = (2, 0, 1)
    assert viewer.dims.order == (0, 1, 2)
    assert len(notices) == 2
    # A GUI request emits napari's warning, then restores the supported view.
    viewer.dims.ndisplay = 3
    viewer.dims.order = (2, 0, 1)
    assert viewer.dims.order == (2, 0, 1)
