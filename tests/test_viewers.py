from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import CloseEvent, KeyEvent, MouseEvent

from tomocube.processing.registration import register_fl_to_ht
from tomocube.viewer import SliceViewer, TCFViewer


@pytest.mark.parametrize("viewer_type", [TCFViewer, SliceViewer])
def test_viewer_uses_selected_timepoint_channel_and_calibration(make_tcf, viewer_type):
    raw = np.arange(60, dtype=np.uint16).reshape(3, 4, 5) + 13300
    path = make_tcf(scalar_attrs=True, fluorescence=True, timepoints={"2": raw, "10": raw + 100})
    with viewer_type(path, timepoint=1, fl_channel="CH1") as viewer:
        if viewer_type is TCFViewer:
            assert viewer.loader.current_timepoint == "10"
            viewer._on_toggle_fluorescence()
            data, extent = viewer.loader.data_3d, viewer._im_xy.get_extent()
            z = viewer.s.current_z
            image = viewer._im_fl_xy
            fl = viewer.loader.fl_data["CH1"]
            params = viewer.loader.reg_params
            sampled = viewer._fl_mapper.get_slice(0, z).data
        else:
            assert viewer.timepoint_key == "10"
            data, extent = viewer.ht_3d, viewer._im_ht.get_extent()
            z = viewer.current_z
            image = viewer._im_fl
            fl, params = viewer.fl_3d, viewer.params
            sampled = image.get_array()
        np.testing.assert_allclose(data, (raw + 100) / 10000)
        np.testing.assert_allclose(extent, [-0.125, 1.125, 1.75, -0.25])
        np.testing.assert_allclose(image.get_extent(), extent)
        expected = register_fl_to_ht(fl, data.shape, params, "CH1")
        np.testing.assert_allclose(sampled, expected[z])
        viewer.fig.canvas.draw()


@pytest.mark.parametrize("viewer_type", [TCFViewer, SliceViewer])
def test_single_voxel_constant_acquisition_opens_without_warnings(make_tcf, viewer_type):
    path = make_tcf(timepoints={"0": np.full((1, 1, 1), 13300, dtype=np.uint16)})
    before = set(plt.get_fignums())
    with viewer_type(path) as viewer:
        slider = viewer.z_slider if viewer_type is TCFViewer else viewer.slider
        assert not slider.active
        assert not slider.ax.get_visible()
        viewer.fig.canvas.draw()
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize("viewer_type", [TCFViewer, SliceViewer])
def test_arrow_key_moves_exactly_one_plane(make_tcf, viewer_type):
    with viewer_type(make_tcf()) as viewer:
        fig = viewer.fig
        fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, key="up"))
        current = viewer.s.current_z if viewer_type is TCFViewer else viewer.current_z
        assert current == 2
        fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, key="down"))
        current = viewer.s.current_z if viewer_type is TCFViewer else viewer.current_z
        assert current == 1


def test_click_crosshairs_and_hover_use_independent_y_spacing(make_tcf):
    with TCFViewer(make_tcf()) as viewer:
        viewer.fig.canvas.draw()
        display_xy = viewer.ax_xy.transData.transform((0.5, 1.0))
        event = MouseEvent("button_press_event", viewer.fig.canvas, *display_xy, button=1)
        viewer.fig.canvas.callbacks.process("button_press_event", event)
        assert (viewer.s.current_x, viewer.s.current_y) == (2, 2)
        np.testing.assert_allclose(viewer._crosshairs["xy"]["h"].get_ydata(), [1, 1])
        np.testing.assert_allclose(viewer._crosshairs["yz"]["v"].get_xdata(), [1, 1])
        viewer._on_motion(SimpleNamespace(inaxes=viewer.ax_xy, xdata=0.5, ydata=1.0))
        assert "RI = 1.3332" in viewer.pixel_text.get_text()
        viewer._flush_histogram()


def test_channel_key_rebuilds_registration_with_selected_offset(make_tcf):
    with TCFViewer(make_tcf(fluorescence=True)) as viewer:
        assert viewer.s.current_fl_channel == "CH0"
        before = viewer._fl_mapper
        viewer.fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", viewer.fig.canvas, key="n"))
        assert viewer.s.current_fl_channel == "CH1"
        assert viewer._fl_mapper is not before
        assert viewer._im_fl_xy.get_visible()
        fl = viewer.loader.fl_data["CH1"]
        expected = register_fl_to_ht(fl, viewer.loader.data_3d.shape, viewer.loader.reg_params, "CH1")
        np.testing.assert_array_equal(viewer._fl_mapper.get_slice(0, viewer.s.current_z).data, expected[viewer.s.current_z])


def test_next_channel_can_reach_valid_channel_after_corrupt_one(make_tcf):
    path = make_tcf(fluorescence=True)
    with h5py.File(path, "a") as handle:
        handle.copy("Data/3DFL/CH0", "Data/3DFL/CH2")
        del handle["Data/3DFL/CH1/000000"]
        handle["Data/3DFL/CH1"].create_dataset("000000", data=np.full((3, 4, 5), np.nan))
    with TCFViewer(path) as viewer:
        viewer._on_next_channel()
        assert viewer.s.current_fl_channel == "CH2"
        assert viewer._im_fl_xy.get_visible()
        assert "Skipped CH1" in viewer.pixel_text.get_text()


def test_save_shortcut_writes_selected_plane_without_toolbar_handler(make_tcf):
    path = make_tcf()
    with TCFViewer(path) as viewer:
        canvas = viewer.fig.canvas
        handler_id = canvas.manager.key_press_handler_id
        assert handler_id not in canvas.callbacks.callbacks["key_press_event"]
        canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key="s"))
        files = list(path.parent.glob("sample acquisition_t000000_z1_*.png"))
        assert len(files) == 1
        assert plt.imread(files[0]).shape[:2] == (4, 5)


def test_timepoint_resize_clamps_navigation_and_removes_stale_fl(make_tcf):
    raw = np.arange(120, dtype=np.uint16).reshape(4, 5, 6) + 13300
    small = np.full((1, 2, 3), 14000, dtype=np.uint16)
    path = make_tcf(fluorescence=True, timepoints={"0": raw, "1": small})
    with h5py.File(path, "a") as handle:
        del handle["Data/3DFL/CH0/1"]
    with TCFViewer(path, fl_channel="CH0") as viewer:
        viewer._on_toggle_fluorescence()
        assert viewer._im_fl_xy.get_visible()
        viewer.tp_slider.set_val(1)
        assert (viewer.s.current_z, viewer.s.current_y, viewer.s.current_x) == (0, 1, 2)
        assert viewer._im_xy.get_array().shape == (2, 3)
        np.testing.assert_allclose(viewer._im_xy.get_extent(), [-0.125, 0.625, 0.75, -0.25])
        assert not viewer._im_fl_xy.get_visible()
        assert not viewer.ax_cbar_fl.get_visible()
        assert "unavailable" in viewer.info_text.get_text()
        assert not viewer.z_slider.active
        assert viewer.y_slider.valmax == 0.5
        assert viewer.contrast_slider.valmin < 1.4 < viewer.contrast_slider.valmax
        viewer._on_toggle_fluorescence()
        viewer._on_toggle_fluorescence()
        assert not viewer._im_fl_xy.get_visible()
        viewer.tp_slider.set_val(0)
        assert viewer._im_fl_xy.get_visible()
        assert viewer.z_slider.active
        assert viewer.y_slider.valmax == 2
        np.testing.assert_allclose(viewer.z_slider.ax.get_xlim(), [0, 4.5])
        viewer.fig.canvas.draw()


def test_initially_missing_fluorescence_appears_at_next_timepoint(make_tcf):
    raw = np.arange(60, dtype=np.uint16).reshape(3, 4, 5) + 13300
    path = make_tcf(fluorescence=True, timepoints={"0": raw, "1": raw})
    with h5py.File(path, "a") as handle:
        del handle["Data/3DFL/CH0/0"]
    with TCFViewer(path, fl_channel="CH0") as viewer:
        viewer._on_toggle_fluorescence()
        assert not viewer._im_fl_xy.get_visible()
        viewer.tp_slider.set_val(1)
        assert viewer._im_fl_xy.get_visible()
        assert viewer._fl_mapper is not None


@pytest.mark.parametrize("viewer_type", [TCFViewer, SliceViewer])
def test_uniform_positive_fluorescence_stays_visible(make_tcf, viewer_type):
    path = make_tcf(fluorescence=True)
    with h5py.File(path, "a") as handle:
        handle["Data/3DFL/CH0/000000"][...] = 10
    with viewer_type(path) as viewer:
        if viewer_type is TCFViewer:
            viewer._on_toggle_fluorescence()
            rgba = viewer._im_fl_xy.get_array()
        else:
            rgba = viewer._im_overlay_fl.get_array()
        assert rgba[:, :, 3].max() > 0


def test_window_close_releases_loader_and_timer(make_tcf):
    viewer = TCFViewer(make_tcf())
    fig = viewer.fig
    loader = viewer.loader
    handle = loader.file
    viewer._on_auto_contrast()
    fig.canvas.callbacks.process("close_event", CloseEvent("close_event", fig.canvas))
    assert viewer._loader is None
    assert viewer._fig is None
    assert viewer._histogram_pending is None
    assert not handle.id.valid
    with pytest.raises(RuntimeError, match="No data loaded"):
        _ = loader.data_3d
    viewer.close()


def test_failed_timepoint_selection_preserves_visible_acquisition(make_tcf):
    raw = np.arange(60, dtype=np.uint16).reshape(3, 4, 5) + 13300
    path = make_tcf(fluorescence=True, timepoints={"0": raw, "1": raw + 100})
    with h5py.File(path, "a") as handle:
        del handle["Data/3DFL/CH0/1"]
        bad_fl = np.ones(raw.shape, dtype=np.float32)
        bad_fl[0, 0, 0] = np.nan
        handle["Data/3DFL/CH0"].create_dataset("1", data=bad_fl)
    with TCFViewer(path) as viewer:
        viewer._on_toggle_fluorescence()
        old_loader, old_mapper = viewer.loader, viewer._fl_mapper
        old_image = np.array(viewer._im_xy.get_array())
        old_info = viewer.info_text.get_text()
        viewer.tp_slider.set_val(1)
        assert viewer.loader is old_loader and viewer._fl_mapper is old_mapper
        assert viewer.loader.current_timepoint == "0"
        assert viewer.s.current_timepoint == viewer.tp_slider.val == 0
        assert viewer.info_text.get_text() == old_info
        np.testing.assert_array_equal(viewer._im_xy.get_array(), old_image)
        assert "Could not load timepoint 1" in viewer.pixel_text.get_text()
        assert viewer._im_fl_xy.get_visible()


@pytest.mark.parametrize("viewer_type", [TCFViewer, SliceViewer])
def test_invalid_selection_does_not_leak_figures(make_tcf, viewer_type):
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="channel"):
        viewer_type(make_tcf(), fl_channel="CH99")
    assert set(plt.get_fignums()) == before
