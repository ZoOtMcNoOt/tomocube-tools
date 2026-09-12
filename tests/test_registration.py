from dataclasses import replace

import numpy as np
import pytest

from tomocube.core.types import RegistrationParams
from tomocube.processing.registration import register_fl_to_ht
from tomocube.viewer.components import FluorescenceMapper


def calibration(**changes):
    return replace(RegistrationParams(ht_res_x=1, ht_res_y=1, ht_res_z=1,
                                      fl_res_x=1, fl_res_y=1, fl_res_z=1), **changes)


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 4, 5), (3, 4, 5), (4, 5)])
def test_identity_preserves_every_voxel_including_last_plane(shape):
    data = (np.arange(np.prod(shape)) + 1).reshape(shape).astype(np.uint16)
    before = data.copy()
    result = register_fl_to_ht(data, shape, calibration())
    np.testing.assert_array_equal(result, data)
    np.testing.assert_array_equal(data, before)
    assert result.dtype == np.float32


def test_different_xy_spacings_preserve_physical_fiducial_separation():
    data = np.zeros((9, 9))
    data[2, 2], data[6, 6] = 1, 2
    result = register_fl_to_ht(data, (17, 33), calibration(fl_res_y=2, fl_res_x=4))
    assert result[4, 8] == 1
    assert result[12, 24] == 2
    np.testing.assert_array_equal(np.argwhere(result == 2), [[12, 24]])


def test_anisotropic_rotation_then_translation_in_physical_units():
    data = np.zeros((9, 9))
    data[4, 2], data[4, 6] = 1, 2
    params = calibration(ht_res_y=2, fl_res_y=2, rotation=np.pi / 2, translation_x=2)
    result = register_fl_to_ht(data, data.shape, params)
    assert result[5, 6] == 1  # x=-2 um rotates to y=+2 um, then moves +2 um in X
    assert result[3, 6] == 2  # x=+2 um rotates to y=-2 um
    np.testing.assert_array_equal(np.argwhere(result == 2), [[3, 6]])


def test_arbitrary_rotation_of_calibrated_linear_ramp():
    z, y, x = np.indices((5, 11, 13))
    data = (100 * z + 10 * y + x).astype(np.float32)
    params = calibration(ht_res_x=0.7, ht_res_y=0.9, ht_res_z=0.6,
                         fl_res_x=1.1, fl_res_y=1.3, fl_res_z=1.2,
                         rotation=0.31, translation_x=0.8, translation_y=-0.4,
                         channel_offsets_z={"CH1": 0.2})
    result = register_fl_to_ht(data, (7, 7, 9), params, "CH1")
    # Independently evaluate the inverse physical mapping, without using the
    # implementation's transform matrix or its output as the expected data.
    for iz, iy, ix in [(1, 1, 2), (3, 3, 4), (6, 5, 7)]:
        ht_y = (iy - 3) * 0.9 + 0.4
        ht_x = (ix - 4) * 0.7 - 0.8
        fl_y = (np.cos(0.31) * ht_y + np.sin(0.31) * ht_x) / 1.3 + 5
        fl_x = (-np.sin(0.31) * ht_y + np.cos(0.31) * ht_x) / 1.1 + 6
        fl_z = (iz * 0.6 - 0.2) / 1.2
        assert result[iz, iy, ix] == pytest.approx(100 * fl_z + 10 * fl_y + fl_x, abs=2e-5)


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_right_angle_rotation_keeps_boundary_pixels(turns):
    data = np.arange(25).reshape(5, 5) + 1
    result = register_fl_to_ht(data, data.shape, calibration(rotation=turns * np.pi / 2))
    np.testing.assert_allclose(result, np.rot90(data, turns), atol=1e-6)


def test_z_interpolation_and_channel_offset_include_both_endpoints():
    data = np.array([10, 30, 50], dtype=np.uint16)[:, None, None]
    params = calibration(fl_res_z=2, channel_offsets_z={"CH1": 1})
    result = register_fl_to_ht(data, (7, 1, 1), params, channel="CH1")
    np.testing.assert_array_equal(result[:, 0, 0], [0, 10, 20, 30, 40, 50, 0])


def test_center_and_signal_alignment_use_voxel_centers():
    data = np.array([0, 8, 0], dtype=np.float32)[:, None, None]
    params = calibration(fl_offset_z=3)
    centered = register_fl_to_ht(data, (7, 1, 1), params, z_offset_mode="center")
    automatic = register_fl_to_ht(data, (7, 1, 1), params, z_offset_mode="auto")
    np.testing.assert_array_equal(centered[:, 0, 0], [0, 0, 0, 8, 0, 0, 0])
    np.testing.assert_array_equal(automatic, centered)
    data[2, 0, 0], data[1, 0, 0] = 8, 0
    np.testing.assert_array_equal(register_fl_to_ht(data, (7, 1, 1), params, z_offset_mode="auto"), centered)
    assert not register_fl_to_ht(np.zeros_like(data), (7, 1, 1), params, z_offset_mode="auto").any()


@pytest.mark.parametrize("mode", ["start", "center", "auto"])
def test_all_viewer_planes_match_registered_export(mode):
    z, y, x = np.indices((5, 7, 9))
    data = (100 * z + 10 * y + x).astype(np.float32)
    params = calibration(ht_res_z=0.7, ht_res_y=0.9, ht_res_x=1.1,
                         fl_res_z=1.2, fl_res_y=1.3, fl_res_x=0.8,
                         rotation=0.3, translation_x=0.4, translation_y=-0.2,
                         channel_offsets_z={"CH1": 0.5})
    shape = (7, 9, 11)
    mapper = FluorescenceMapper(data, shape, params, "CH1", mode)
    registered = register_fl_to_ht(data, shape, params, "CH1", mode)
    for axis, size in enumerate(shape):
        for index in (0, size // 2, size - 1):
            result = mapper.get_slice(axis, index)
            np.testing.assert_allclose(result.data, np.take(registered, index, axis), atol=1e-4)


def test_mapper_channel_offset_manual_adjustment_and_no_overlap():
    data = np.ones((1, 3, 3), dtype=np.float32)
    params = calibration(channel_offsets_z={"CH1": 3})
    mapper = FluorescenceMapper(data, (6, 3, 3), params, "CH1")
    assert not mapper.get_slice(0, 0).in_range
    np.testing.assert_array_equal(mapper.get_slice(0, 3).data, 1)
    np.testing.assert_array_equal(mapper.get_slice(0, 4, z_offset_um=1).data, 1)
    assert not mapper.get_slice(0, 3, z_offset_um=1).in_range
    np.testing.assert_array_equal(data, 1)
    assert params.channel_offsets_z == {"CH1": 3}


def test_mapper_keeps_native_integer_volume_and_interpolates_to_float32():
    data = np.array([0, 1], dtype=np.uint16)[:, None, None]
    mapper = FluorescenceMapper(data, (3, 1, 1), calibration(fl_res_z=2))
    assert np.shares_memory(mapper.registration.data, data)
    result = mapper.get_slice(0, 1)
    assert result.data.dtype == np.float32
    assert result.data[0, 0] == 0.5


@pytest.mark.parametrize("dtype", ["<f2", ">f2"])
def test_half_precision_inputs_support_both_byte_orders(dtype):
    data = np.array([0, 1], dtype=dtype)[:, None, None]
    result = register_fl_to_ht(data, (3, 1, 1), calibration(fl_res_z=2))
    np.testing.assert_array_equal(result[:, 0, 0], [0, 0.5, 1])


@pytest.mark.parametrize("data,shape,params,mode", [
    (np.ones((2, 2)), (2, 2, 2), calibration(), "start"),
    (np.ones((2, 2, 2)), (2, 2), calibration(), "start"),
    (np.ones((2, 2)), (2, 0), calibration(), "start"),
    (np.ones((2, 2)), (2, 2.5), calibration(), "start"),
    (np.array([[np.nan]]), (1, 1), calibration(), "start"),
    (np.array([[np.inf]]), (1, 1), calibration(), "start"),
    (np.ones((1, 1), dtype=complex), (1, 1), calibration(), "start"),
    (np.ones((1, 1)), (1, 1), calibration(fl_res_y=0), "start"),
    (np.ones((1, 1)), (1, 1), calibration(rotation=np.inf), "start"),
    (np.ones((1, 1)), (1, 1), calibration(), "guess"),
])
def test_registration_rejects_invalid_inputs(data, shape, params, mode):
    with pytest.raises(ValueError):
        register_fl_to_ht(data, shape, params, z_offset_mode=mode)
