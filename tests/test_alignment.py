from dataclasses import replace
import json

import h5py
import numpy as np
import pytest

from tomocube import RegistrationParams, TCFFileLoader
from tomocube.processing.alignment import estimate_translation, load_alignment, save_alignment
from tomocube.processing.registration import FluorescenceRegistration, register_fl_to_ht
from scipy.ndimage import gaussian_filter


def calibration(**kwargs):
    return replace(RegistrationParams(ht_res_z=1.2, ht_res_y=0.7, ht_res_x=0.5,
                                      fl_res_z=1.2, fl_res_y=0.7, fl_res_x=0.5), **kwargs)


def phantom(shape=(24, 32, 36), *, shift=(1.1, -0.8, 0.65), params=None):
    """Independent analytic positive-contrast fiducials in world coordinates."""
    params = params or calibration()
    spacing = np.array([params.ht_res_z, params.ht_res_y, params.ht_res_x])
    fl_spacing = np.array([params.fl_res_z, params.fl_res_y, params.fl_res_x])
    xyz = np.indices(shape, dtype=float) * spacing[:, None, None, None]
    fl_world = np.indices(shape, dtype=float) * fl_spacing[:, None, None, None]
    center = (np.array(shape) - 1) / 2
    y, x = fl_world[1] - center[1] * fl_spacing[1], fl_world[2] - center[2] * fl_spacing[2]
    fl_world[1] = np.cos(params.rotation) * y - np.sin(params.rotation) * x + center[1] * spacing[1] + params.translation_y
    fl_world[2] = np.sin(params.rotation) * y + np.cos(params.rotation) * x + center[2] * spacing[2] + params.translation_x
    fl_world[0] += params.get_offset_z("CH1")
    fl_world += np.array(shift)[:, None, None, None]
    rng = np.random.default_rng(72)
    centers = rng.uniform(0.2, 0.8, (22, 3)) * ((np.array(shape) - 1) * spacing)
    widths = rng.uniform(0.7, 1.5, (22, 3))
    heights = rng.uniform(0.5, 1.5, 22)
    def field(coordinates):
        volume = np.zeros(shape)
        for c, w, h in zip(centers, widths, heights):
            distance = (coordinates - c[:, None, None, None]) / w[:, None, None, None]
            volume += h * np.exp(-0.5 * np.sum(distance * distance, axis=0))
        return volume
    ht = (1.33 + 0.04 * field(xyz)).astype(np.float32)
    fl = (100 + 1200 * field(fl_world)).astype(np.uint16)
    return ht, fl, params


@pytest.mark.parametrize("shift", [(0, 0, 0), (2.4, -1.4, 1), (1.1, -0.8, 0.65)])
def test_recovers_translation_with_different_intensity_units(shift):
    ht, fl, params = phantom(shift=shift)
    before_ht, before_fl = ht.copy(), fl.copy()
    result = estimate_translation(ht, fl, params, max_shift_um=4)
    assert result.accepted, result
    np.testing.assert_allclose(result.translation_um, shift, atol=0.17)
    assert result.score_after > 0.97
    assert result.score_after >= result.score_before - 1e-5
    np.testing.assert_array_equal(ht, before_ht)
    np.testing.assert_array_equal(fl, before_fl)


def test_recovers_residual_after_anisotropic_metadata_rotation_and_offset():
    params = calibration(rotation=0.15, translation_y=-0.2, translation_x=0.4,
                         fl_res_z=1.3, fl_res_y=0.8, fl_res_x=0.6, channel_offsets_z={"CH1": 0.4})
    ht, fl, params = phantom(params=params)
    result = estimate_translation(ht, fl, params, channel="CH1", max_shift_um=3)
    assert result.accepted, result
    np.testing.assert_allclose(result.translation_um, [1.1, -0.8, 0.65], atol=0.18)


def test_unrelated_noise_is_rejected():
    rng = np.random.default_rng(123)
    result = estimate_translation(rng.normal(size=(20, 24, 28)), rng.normal(size=(20, 24, 28)),
                                  calibration(), max_shift_um=3)
    assert not result.accepted
    assert result.reason == "weak_correlation"


def test_periodic_structure_is_ambiguous():
    z, y, x = np.indices((24, 32, 36))
    data = (np.cos(z * np.pi / 2) + np.cos(y * np.pi / 2) + np.cos(x * np.pi / 2) + 4)
    result = estimate_translation(data, data.copy(), calibration(), max_shift_um=6)
    assert not result.accepted
    assert result.reason == "ambiguous_peak"


@pytest.mark.parametrize("ramp", [False, True])
@pytest.mark.parametrize("bounds", [1.5, (6, 6, 1.5)])
def test_small_search_never_accepts_unobservable_translation(ramp, bounds):
    z, y, x = np.indices((16, 16, 16))
    data = z + 2 * y + 3 * x if ramp else np.exp(-((z - 7.2) ** 2 + (y - 5.8) ** 2) / 15)
    params = calibration(ht_res_z=1, ht_res_y=1, ht_res_x=1, fl_res_z=1, fl_res_y=1, fl_res_x=1)
    result = estimate_translation(data, data.copy(), params, max_shift_um=bounds)
    assert not result.accepted, result


def test_repeated_structures_outside_initial_footprint_are_competing_matches():
    tile = gaussian_filter(np.random.default_rng(991).random((12, 12, 12)), 0.7, mode="wrap")
    fl = np.tile(tile, (4, 4, 4))
    ht = fl[16:32, 16:32, 16:32].copy()
    params = calibration(ht_res_z=1, ht_res_y=1, ht_res_x=1,
                         fl_res_z=1, fl_res_y=1, fl_res_x=1, fl_offset_z=-16)
    result = estimate_translation(ht, fl, params, max_shift_um=14)
    assert not result.accepted, result
    assert result.reason == "ambiguous_peak"


def test_downsampled_search_refines_interior_peak_near_last_coarse_sample():
    data = gaussian_filter(np.random.default_rng(354).random((72, 72, 72)), 1.6)
    # Analytic integer translation with valid original support, independently
    # cropped from a larger image so neither sampler creates the expectation.
    ht = data[12:60, 12:60, 12:60]
    fl = data[15:63, 10:58, 13:61]
    params = calibration(ht_res_z=1.3, ht_res_y=0.8, ht_res_x=0.6,
                         fl_res_z=1.3, fl_res_y=0.8, fl_res_x=0.6)
    result = estimate_translation(ht, fl, params, max_dimension=16, max_shift_um=(7, 5, 3))
    assert result.accepted, result
    np.testing.assert_allclose(result.translation_um, (3.9, -1.6, 0.6), atol=0.03)


@pytest.mark.parametrize("shape,reason", [((1, 10, 10), "insufficient_3d_extent"),
                                         ((10, 10, 10), "insufficient_signal")])
def test_flat_or_unobservable_inputs_have_no_accepted_transform(shape, reason):
    result = estimate_translation(np.ones(shape), np.ones(shape), calibration())
    assert not result.accepted
    assert result.reason == reason


def test_out_of_range_translation_is_not_silently_accepted():
    ht, fl, params = phantom(shift=(0, 0, 3))
    result = estimate_translation(ht, fl, params, max_shift_um=2)
    assert not result.accepted


def test_shared_forward_affine_and_residual_agree_with_sampling():
    data = np.arange(5 * 7 * 9).reshape(5, 7, 9)
    params = calibration(rotation=0.3, translation_x=0.6)
    mapper = FluorescenceRegistration(data, (5, 7, 9), params, translation_um=(1.2, -0.7, 0.5))
    fl_index = np.array([2.1, 3.4, 4.2])
    world = (mapper.voxel_to_world @ np.r_[fl_index, 1])[:3]
    np.testing.assert_allclose(mapper.matrix @ (world / mapper.ht_spacing) + mapper.offset, fl_index)
    registered = register_fl_to_ht(data, data.shape, params, translation_um=(1.2, -0.7, 0.5))
    np.testing.assert_array_equal(mapper.sample_plane(0, 3)[0], registered[3])


def test_report_replays_only_on_matching_source(make_tcf, tmp_path):
    ht, fl, _ = phantom()
    path = make_tcf(timepoints={"0": ht}, fluorescence=True)
    with h5py.File(path, "r+") as f:
        for axis, spacing in zip("ZYX", [1.2, 0.7, 0.5]):
            f["Data/3D"].attrs[f"Resolution{axis}"] = spacing
            f["Data/3DFL"].attrs[f"Resolution{axis}"] = spacing
        f["Data/3DFL/CH0/0"][...] = fl
    report = tmp_path / "alignment.json"
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        result = estimate_translation(loader.data_3d, loader.fl_data["CH0"], loader.reg_params, max_shift_um=4)
        assert result.accepted, result
        save_alignment(loader, "CH0", result, report)
        replay = load_alignment(report, loader, "CH0")
        np.testing.assert_array_equal(replay.translation_um, result.translation_um)
        with pytest.raises(ValueError, match="does not match"):
            load_alignment(report, loader, "CH1")
        with pytest.raises(FileExistsError):
            save_alignment(loader, "CH0", result, report)
        loader.data_3d[0, 0, 0] += 0.001
        with pytest.raises(ValueError, match="does not match"):
            load_alignment(report, loader, "CH0")


def test_rejected_and_malformed_reports_never_apply(make_tcf, tmp_path):
    path = make_tcf(fluorescence=True)
    report = tmp_path / "alignment.json"
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        result = estimate_translation(loader.data_3d, loader.fl_data["CH0"], loader.reg_params)
        assert not result.accepted
        save_alignment(loader, "CH0", result, report)
        with pytest.raises(ValueError, match="rejected"):
            load_alignment(report, loader, "CH0")
        payload = json.loads(report.read_text())
        payload["result"]["translation_um"] = [float("nan"), 0, 0]
        report.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="finite"):
            load_alignment(report, loader, "CH0")


@pytest.mark.parametrize("changes", [{"max_shift_um": 0}, {"min_score": 0}, {"min_overlap": 2},
                                     {"max_dimension": 4}, {"max_dimension": True}])
def test_invalid_estimator_options_fail(changes):
    with pytest.raises(ValueError):
        estimate_translation(np.ones((10, 10, 10)), np.ones((10, 10, 10)), calibration(), **changes)
