import hashlib
import json

import h5py
import numpy as np
import pytest

from tomocube import TCFFileError, TCFParseError
from tomocube.processing import analysis
from tomocube.processing.analysis import analyze_acquisition, inspect_acquisition


def test_inventory_is_header_only_and_includes_shapes_missing_and_orphan_channels(make_tcf, monkeypatch):
    path = make_tcf(timepoints={
        "10": np.full((4, 5, 6), 13500, dtype=np.uint16),
        "2": np.full((2, 3, 4), 13400, dtype=np.float32),
    }, fluorescence=True)
    with h5py.File(path, "a") as file:
        del file["Data/3DFL/CH1/2"]
        file.create_dataset("Data/3DFL/CH0/20", data=np.zeros((1, 2, 3), dtype=np.uint8))

    def no_pixels(*args, **kwargs):
        raise AssertionError("inventory must not read volume pixels")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", no_pixels)
    monkeypatch.setattr(h5py.Dataset, "__array__", no_pixels)
    result = inspect_acquisition(path)
    json.dumps(result, allow_nan=False)
    assert result["timepoints"] == ["2", "10"]
    assert result["fl_channels"] == ["CH0", "CH1"]
    assert result["source"] == str(path.resolve())
    assert result["metadata"]["medium_ri"] == 1.33
    assert result["metadata"]["registration"]["ht_res_x"] == 0.25
    volumes = {(item["acquisition_key"], item["channel"]): item for item in result["volumes"]}
    ht = volumes["2", "HT"]
    assert ht["shape_zyx"] == [2, 3, 4]
    assert ht["dtype"] == "float32"
    assert ht["voxel_count"] == 24
    assert ht["size_bytes"] == 96
    assert ht["spacing_zyx_um"] == [1.5, 0.5, 0.25]
    assert volumes["10", "HT"]["shape_zyx"] == [4, 5, 6]
    missing = volumes["2", "CH1"]
    assert missing["status"] == "missing"
    assert missing["shape_zyx"] is None
    assert missing["voxel_count"] is None
    orphan = volumes["20", "CH0"]
    assert orphan["timepoint_index"] is None
    assert orphan["shape_zyx"] == [1, 2, 3]
    assert orphan["spacing_zyx_um"] == [2.0, 1.0, 0.75]


def test_known_anisotropic_roi_statistics_and_inclusive_threshold(make_tcf):
    raw = np.arange(120, dtype=np.uint16).reshape(5, 4, 6) + 13300
    path = make_tcf(timepoints={"2": raw})
    roi = ((1, 4), (1, 4), (2, 6))
    values = raw[1:4, 1:4, 2:6].astype(np.float64) / 10000
    threshold = float(values[1, 1, 1])
    row, = analyze_acquisition(path, roi=roi, threshold=threshold, block_depth=2)
    json.dumps(row, allow_nan=False)
    assert row["channel"] == "HT"
    assert row["status"] == "ok"
    assert row["roi_zyx"] == [[1, 4], [1, 4], [2, 6]]
    assert row["voxel_count"] == values.size
    assert row["volume_um3"] == pytest.approx(values.size * 1.5 * 0.5 * 0.25)
    assert row["min"] == values.min()
    assert row["max"] == values.max()
    assert row["mean"] == pytest.approx(values.mean(), abs=1e-14)
    assert row["std"] == pytest.approx(values.std(), abs=1e-14)
    assert row["integral"] == pytest.approx(values.sum() * 1.5 * 0.5 * 0.25)
    assert row["value_unit"] == "RI"
    assert row["integral_unit"] == "RI*um^3"
    assert row["ri_divisor"] == 10000
    assert row["threshold_operator"] == ">="
    assert row["selected_voxel_count"] == np.count_nonzero(values >= threshold)
    assert row["selected_volume_um3"] == pytest.approx(np.count_nonzero(values >= threshold) * 0.1875)


def test_default_all_timepoints_ht_only_and_missing_fluorescence_is_explicit(make_tcf, capsys):
    path = make_tcf(timepoints={
        str(key): np.full((2, 3, 4), 13300 + key, dtype=np.uint16)
        for key in (10, 2, 0)
    }, fluorescence=True)
    with h5py.File(path, "a") as file:
        del file["Data/3DFL/CH1/2"]
    defaults = analyze_acquisition(path)
    assert [row["acquisition_key"] for row in defaults] == ["0", "2", "10"]
    assert all(row["channel"] == "HT" for row in defaults)
    rows = analyze_acquisition(path, channels=["CH1"], timepoints=[2, 1, 0])
    assert [row["acquisition_key"] for row in rows] == ["0", "2", "10"]
    missing = rows[1]
    assert missing["status"] == "missing"
    assert missing["dataset_path"] == "/Data/3DFL/CH1/2"
    for name in ("min", "max", "mean", "std", "integral", "voxel_count", "volume_um3"):
        assert missing[name] is None
    assert rows[0]["value_unit"] == "count"
    assert rows[0]["mean"] == pytest.approx(11.5)
    assert rows[0]["volume_um3"] == 36
    assert rows[0]["integral"] == pytest.approx(np.arange(24).sum() * 1.5)
    assert rows[0]["ri_divisor"] is None
    assert rows[0]["selected_voxel_count"] is None
    assert capsys.readouterr().out == ""
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize("depth", [1, 2, 3, 20])
def test_float_ri_scale_is_dataset_wide_even_outside_roi(make_tcf, depth):
    raw = np.linspace(20, 80, 60).reshape(5, 3, 4)
    raw[-1, -1, -1] = 14000
    path = make_tcf(timepoints={"0": raw})
    roi = ((0, 2), (0, 3), (0, 4))
    row, = analyze_acquisition(path, roi=roi, block_depth=depth)
    expected = raw[:2] / 10000
    assert row["ri_divisor"] == 10000
    assert row["mean"] == pytest.approx(expected.mean(), rel=1e-14)
    assert row["std"] == pytest.approx(expected.std(), rel=1e-14)
    assert row["max"] == pytest.approx(expected.max(), rel=1e-14)


@pytest.mark.parametrize("depth", [1, 3, 8])
def test_physical_ri_and_high_offset_variance_are_block_size_invariant(make_tcf, depth):
    raw = np.linspace(1.33, 1.4, 120).reshape(5, 4, 6)
    path = make_tcf(timepoints={"0": raw}, fluorescence=True)
    fluorescence = (10**12 + np.arange(120).reshape(5, 4, 6)).astype(np.float64)
    with h5py.File(path, "a") as file:
        del file["Data/3DFL/CH0/0"]
        file.create_dataset("Data/3DFL/CH0/0", data=fluorescence)
    ht, fl = analyze_acquisition(path, channels=["HT", "CH0"], block_depth=depth)
    assert ht["ri_divisor"] == 1
    assert ht["mean"] == pytest.approx(raw.mean(), rel=1e-14)
    assert ht["std"] == pytest.approx(raw.std(), rel=1e-14)
    assert fl["mean"] == pytest.approx(fluorescence.mean(), abs=1e-6)
    assert fl["std"] == pytest.approx(fluorescence.std(), rel=1e-14)


def test_analysis_never_eager_loads_and_read_selections_are_bounded(make_tcf, monkeypatch):
    raw = np.arange(180, dtype=np.uint16).reshape(9, 4, 5) + 13300
    path = make_tcf(timepoints={"0": raw}, fluorescence=True)
    reads = []
    original = h5py.Dataset.__getitem__

    def track(dataset, selection, *args, **kwargs):
        reads.append((dataset.name, selection))
        return original(dataset, selection, *args, **kwargs)

    def no_array(*args, **kwargs):
        raise AssertionError("analysis must not eagerly load volumes")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", track)
    monkeypatch.setattr(h5py.Dataset, "__array__", no_array)
    analyze_acquisition(path, channels=["CH0"], roi=((2, 8), (1, 3), (2, 5)), block_depth=2)
    assert len(reads) == 3
    for name, selection in reads:
        assert name == "/Data/3DFL/CH0/0"
        assert selection[0].stop - selection[0].start <= 2
        assert selection[1:] == (slice(1, 3), slice(2, 5))


def test_floating_ht_prepass_is_bounded_and_scale_is_chosen_once(make_tcf, monkeypatch):
    raw = np.ones((7, 3, 4), dtype=np.float32)
    raw[6] *= 13300
    path = make_tcf(timepoints={"0": raw})
    original = h5py.Dataset.__getitem__
    reads = []

    def track(dataset, selection, *args, **kwargs):
        reads.append(selection)
        return original(dataset, selection, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", track)
    row, = analyze_acquisition(path, block_depth=2, roi=((0, 1), (0, 2), (0, 3)))
    assert len(reads) == 5  # Four full-width slabs establish scale; one reads ROI.
    assert all(selection[0].stop - selection[0].start <= 2 for selection in reads)
    assert row["mean"] == 0.0001


@pytest.mark.parametrize("roi", [
    ((0, 0), (0, 2), (0, 2)),
    ((-1, 2), (0, 2), (0, 2)),
    ((0, 4), (0, 2), (0, 2)),
    ((0, 2.0), (0, 2), (0, 2)),
    ((False, 2), (0, 2), (0, 2)),
    ((0, 2), (0, 2)),
    (0, 1, 2),
])
def test_invalid_roi_rejected_before_pixel_read(make_tcf, monkeypatch, roi):
    path = make_tcf()

    def no_pixels(*args, **kwargs):
        raise AssertionError("invalid ROI must fail before pixel reads")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", no_pixels)
    with pytest.raises(ValueError, match="roi"):
        analyze_acquisition(path, roi=roi)


def test_roi_must_fit_every_selected_native_channel(make_tcf, monkeypatch):
    path = make_tcf(fluorescence=True)
    with h5py.File(path, "a") as file:
        del file["Data/3DFL/CH1/000000"]
        file.create_dataset("Data/3DFL/CH1/000000", data=np.ones((1, 2, 2)))

    def no_pixels(*args, **kwargs):
        raise AssertionError("all ROIs must validate before any read")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", no_pixels)
    with pytest.raises(ValueError, match="roi Z"):
        analyze_acquisition(path, channels=["HT", "CH1"], roi=((0, 2), (0, 2), (0, 2)))


@pytest.mark.parametrize("kwargs, error", [
    ({"timepoints": [-1]}, IndexError), ({"timepoints": [4]}, IndexError),
    ({"timepoints": [0.0]}, ValueError), ({"timepoints": [True]}, ValueError),
    ({"timepoints": [0, 0]}, ValueError), ({"timepoints": []}, ValueError),
    ({"channels": "CH0"}, ValueError), ({"channels": ["missing"]}, ValueError),
    ({"channels": []}, ValueError), ({"channels": ["HT", "HT"]}, ValueError),
    ({"threshold": np.nan}, ValueError), ({"threshold": np.inf}, ValueError),
    ({"threshold": True}, ValueError),
    ({"block_depth": 0}, ValueError), ({"block_depth": 1.2}, ValueError),
    ({"block_depth": True}, ValueError),
])
def test_invalid_selections_rejected(make_tcf, kwargs, error):
    with pytest.raises(error):
        analyze_acquisition(make_tcf(fluorescence=True), **kwargs)


@pytest.mark.parametrize("calibration", [0, -1, np.inf, np.nan])
@pytest.mark.parametrize("operation", [inspect_acquisition, analyze_acquisition])
def test_invalid_calibration_rejected(make_tcf, calibration, operation):
    path = make_tcf()
    with h5py.File(path, "a") as file:
        file["Data/3D"].attrs["ResolutionZ"] = calibration
    with pytest.raises(TCFParseError, match="ResolutionZ"):
        operation(path)


def test_nonfinite_selected_values_fail_and_release_file(make_tcf, monkeypatch):
    path = make_tcf(fluorescence=True)
    with h5py.File(path, "a") as file:
        del file["Data/3DFL/CH0/000000"]
        values = np.ones((3, 4, 5))
        values[2, 0, 0] = np.nan
        file.create_dataset("Data/3DFL/CH0/000000", data=values)
    handles = []
    original = analysis._read_info

    def track(file):
        handles.append(file)
        return original(file)

    monkeypatch.setattr(analysis, "_read_info", track)
    with pytest.raises(TCFFileError, match="CH0/000000.*nonfinite"):
        analyze_acquisition(path, channels=["CH0"], block_depth=1)
    assert not handles[0].id.valid
    # HT-only work should not read the invalid, unrelated fluorescence channel.
    row, = analyze_acquisition(path)
    assert row["status"] == "ok"


def test_nonfinite_float_ht_outside_roi_rejects_ambiguous_scale(make_tcf):
    path = make_tcf()
    with h5py.File(path, "a") as file:
        del file["Data/3D/000000"]
        raw = np.ones((3, 4, 5), dtype=np.float32)
        raw[2, 3, 4] = np.nan
        file.create_dataset("Data/3D/000000", data=raw)
    with pytest.raises(TCFFileError, match="nonfinite"):
        analyze_acquisition(path, roi=((0, 1), (0, 2), (0, 2)), block_depth=1)


def test_analysis_and_inventory_never_modify_source(make_tcf):
    path = make_tcf(fluorescence=True)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    inspect_acquisition(path)
    analyze_acquisition(path, channels=["HT", "CH1"], block_depth=1)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_calibration_default_provenance_is_preserved_in_reports(make_tcf):
    path = make_tcf(fluorescence=True)
    with h5py.File(path, "a") as file:
        del file["Data/3D"].attrs["ResolutionX"]
        del file["Data/3DFL/CH1"].attrs["OffsetZ"]
    inventory = inspect_acquisition(path)
    ht = inventory["volumes"][0]
    assert ht["spacing_zyx_source"] == ["metadata", "metadata", "instrument_default"]
    assert ht["spacing_zyx_um"] == [1.5, 0.5, 0.196]
    assert inventory["volumes"][2]["offset_z_source"] == "fallback"
    row, = analyze_acquisition(path)
    assert row["spacing_zyx_source"] == ht["spacing_zyx_source"]


def test_float64_overflow_is_an_explicit_error_not_nonfinite_json(make_tcf):
    path = make_tcf(fluorescence=True)
    with h5py.File(path, "a") as file:
        del file["Data/3DFL/CH0/000000"]
        values = np.array([1e300, -1e300]).reshape(2, 1, 1)
        file.create_dataset("Data/3DFL/CH0/000000", data=values)
    with pytest.raises(ValueError, match="float64 numeric range"):
        analyze_acquisition(path, channels=["CH0"])
