import numpy as np
import pytest

from tomocube.viewer.measurements import extract_line_profile


def test_anisotropic_profile_recovers_physical_linear_ramp():
    y, x = np.indices((6, 9))
    spacing = (1.5, 0.25)
    data = 3 * (y * spacing[0]) + 7 * (x * spacing[1]) + 11
    p1, p2 = (0.125, 0.75), (1.875, 6.75)
    distances, values = extract_line_profile(data, p1, p2, spacing, num_points=17)
    points = np.linspace(p1, p2, 17)
    np.testing.assert_allclose(values, 3 * points[:, 1] + 7 * points[:, 0] + 11)
    np.testing.assert_allclose(distances, np.linspace(0, np.hypot(1.75, 6), 17))


def test_integer_profile_preserves_fractional_interpolation_and_endpoints():
    data = np.array([[0, 1], [2, 3]], dtype=np.uint16)
    distances, values = extract_line_profile(data, (0, 0), (1, 1), 1, num_points=5)
    assert values.dtype == np.float64
    np.testing.assert_allclose(values, [0, 0.75, 1.5, 2.25, 3])
    np.testing.assert_allclose(distances, np.linspace(0, np.sqrt(2), 5))


def test_default_sampling_is_at_most_one_pixel_apart():
    distances, values = extract_line_profile(np.arange(12).reshape(3, 4), (0, 0), (3, 2), 1)
    assert len(values) == 5
    assert np.diff(distances).max() <= 1


def test_singleton_image_and_coincident_endpoints():
    distances, values = extract_line_profile(np.array([[13]], dtype=np.float16), (0, 0), (0, 0), (2, 3))
    np.testing.assert_array_equal(distances, [0, 0])
    np.testing.assert_array_equal(values, [13, 13])


def test_reversing_a_profile_preserves_length_and_reverses_values():
    data = np.arange(30).reshape(5, 6)
    a = extract_line_profile(data, (0.5, 0.75), (2.5, 3), (0.75, 0.5), 11)
    b = extract_line_profile(data, (2.5, 3), (0.5, 0.75), (0.75, 0.5), 11)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1][::-1])


@pytest.mark.parametrize("spacing", [0, -1, np.nan, np.inf, (1, 0), (1, np.nan), (1, 2, 3), True, "1"])
def test_profile_rejects_invalid_calibration(spacing):
    with pytest.raises(ValueError, match="spacing"):
        extract_line_profile(np.zeros((3, 3)), (0, 0), (1, 1), spacing)


@pytest.mark.parametrize("count", [0, 1, -1, 1.5, np.nan, np.inf, True, "4"])
def test_profile_rejects_invalid_sample_count(count):
    with pytest.raises(ValueError, match="num_points"):
        extract_line_profile(np.zeros((3, 3)), (0, 0), (1, 1), 1, count)


@pytest.mark.parametrize("point", [(-0.01, 0), (0, -0.01), (2.01, 0), (0, 2.01), (np.nan, 1), (1, np.inf)])
def test_profile_rejects_invalid_physical_endpoints(point):
    with pytest.raises(ValueError, match="coordinates|endpoints"):
        extract_line_profile(np.zeros((3, 3)), (0, 0), point, 1)


@pytest.mark.parametrize("data", [np.zeros((0, 3)), np.zeros((3,)), np.zeros((1, 2, 3)),
                                  np.array([[np.nan]]), np.array([[np.inf]]), np.array([[1j]]), np.array([["1"]])])
def test_profile_rejects_invalid_image(data):
    with pytest.raises(ValueError, match="data"):
        extract_line_profile(data, (0, 0), (0, 0), 1)
