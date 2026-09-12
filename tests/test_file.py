import h5py
import numpy as np
import pytest

from tomocube import TCFFile, TCFFileError, TCFFileLoader, TCFParseError, extract_metadata


@pytest.mark.parametrize("scalar_attrs", [False, True])
def test_scalar_and_singleton_attributes(make_tcf, scalar_attrs):
    path = make_tcf(scalar_attrs=scalar_attrs, fluorescence=True)
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        info = loader.tcf_info
        assert info.device_model == "HTX"
        assert info.device_serial == "TEST-001"
        assert info.software_version == "1.0"
        assert info.magnification == 60
        assert info.ht_resolution == (1.5, 0.5, 0.25)
        assert info.fl_resolution == (2.0, 1.0, 0.75)
        assert info.ri_min == pytest.approx(1.33)
        assert loader.reg_params.get_offset_z("CH0") == 0.0
        assert loader.reg_params.get_offset_z("CH1") == 1.0
        assert loader.reg_params.get_offset_z() == 0.0
        assert extract_metadata(loader.file)["device"]["Magnification"] == 60
        np.testing.assert_allclose(loader.data_3d.flat[0], 1.33)


def test_ht_only_spacing_matches_metadata(make_tcf):
    with TCFFileLoader(make_tcf()) as loader:
        params = loader.reg_params
        assert (params.ht_res_z, params.ht_res_y, params.ht_res_x) == loader.tcf_info.ht_resolution


def test_timepoints_follow_numeric_order(make_tcf):
    volumes = {str(i): np.full((2, 3, 4), 13300 + i, dtype=np.uint16) for i in (10, 2, 1, 0)}
    with TCFFileLoader(make_tcf(timepoints=volumes)) as loader:
        assert loader.timepoints == ["0", "1", "2", "10"]
        loader.load_timepoint(2)
        np.testing.assert_allclose(loader.data_3d, 1.3302)


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float32])
def test_ri_and_mip_conversion(make_tcf, dtype):
    raw = np.arange(24).reshape(2, 3, 4).astype(dtype) + 13300
    path = make_tcf(timepoints={"0": raw})
    with h5py.File(path, "a") as f:
        f.create_dataset("Data/2DMIP/0", data=raw.max(axis=0))
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        assert loader.data_3d.dtype == np.float32
        np.testing.assert_allclose(loader.data_3d, raw / 10000)
        np.testing.assert_allclose(loader.data_mip, loader.data_3d.max(axis=0))


def test_physical_ri_is_not_rescaled(make_tcf):
    raw = np.linspace(1.33, 1.4, 24, dtype=np.float32).reshape(2, 3, 4)
    with TCFFileLoader(make_tcf(timepoints={"0": raw})) as loader:
        loader.load_timepoint(0)
        np.testing.assert_array_equal(loader.data_3d, raw)
        np.testing.assert_array_equal(loader.data_mip, raw.max(axis=0))


@pytest.mark.parametrize("shape", [(2, 3), (0, 3, 4), (1, 2, 3, 4)])
def test_invalid_volume_rejected_with_dataset_path(tmp_path, shape):
    path = tmp_path / "invalid.TCF"
    with h5py.File(path, "w") as f:
        f.create_dataset("Data/3D/0", shape=shape, dtype="uint16")
    with pytest.raises(TCFFileError, match="Data/3D/0"):
        with TCFFileLoader(path):
            pass


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf, [1.0, 2.0]])
def test_invalid_resolution_is_not_silently_replaced(make_tcf, invalid):
    path = make_tcf()
    with h5py.File(path, "a") as f:
        f["Data/3D"].attrs["ResolutionX"] = invalid
    with pytest.raises(TCFParseError, match="ResolutionX"):
        with TCFFileLoader(path):
            pass


def test_missing_ht_rejected(tmp_path):
    path = tmp_path / "empty.TCF"
    with h5py.File(path, "w"):
        pass
    with h5py.File(path, "r") as f:
        with pytest.raises(TCFFileError, match="Data/3D"):
            TCFFile.from_hdf5(f)


def test_failed_open_releases_handle(make_tcf, monkeypatch):
    loader = TCFFileLoader(make_tcf())
    handles = []

    def fail(f):
        handles.append(f)
        raise TCFParseError("malformed metadata")

    monkeypatch.setattr(TCFFile, "from_hdf5", fail)
    with pytest.raises(TCFParseError):
        with loader:
            pass
    try:
        assert not handles[0].id.valid
        with pytest.raises(RuntimeError):
            _ = loader.file
    finally:
        loader.close()


def test_repeated_load_reuses_handle_and_close_clears_state(make_tcf):
    loader = TCFFileLoader(make_tcf(fluorescence=True))
    try:
        loader.load()
        original = loader.file
        loader.load_timepoint(0)
        loader.load()
        assert loader.file is original
    finally:
        loader.close()
    assert not original.id.valid
    assert loader.fl_data == {}
    for name in ("file", "tcf_info", "data_3d", "data_mip", "current_timepoint"):
        with pytest.raises(RuntimeError):
            getattr(loader, name)


def test_failed_timepoint_does_not_mix_acquisitions(make_tcf):
    volumes = {str(i): np.full((2, 3, 4), 13300 + i * 100, dtype=np.uint16) for i in (0, 1)}
    path = make_tcf(timepoints=volumes, fluorescence=True)
    with h5py.File(path, "a") as f:
        f.create_dataset("Data/2DMIP/1", data=np.ones((5, 6)))
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        previous = loader.data_3d.copy()
        with pytest.raises(TCFFileError, match="Data/2DMIP/1"):
            loader.load_timepoint(1)
        np.testing.assert_array_equal(loader.data_3d, previous)
        assert loader.current_timepoint == "0"


def test_missing_resolution_uses_documented_defaults(make_tcf, caplog):
    path = make_tcf()
    with h5py.File(path, "a") as f:
        del f["Data/3D"].attrs["ResolutionX"]
    with TCFFileLoader(path) as loader:
        assert loader.tcf_info.ht_resolution == (1.5, 0.5, 0.196)
        assert loader.reg_params.ht_res_x == 0.196
    assert "ResolutionX" in caplog.text


def test_empty_fluorescence_group_is_not_reported_as_available(make_tcf):
    path = make_tcf()
    with h5py.File(path, "a") as f:
        f.create_group("Data/3DFL/CH0")
    with TCFFileLoader(path) as loader:
        assert not loader.has_fluorescence
        assert loader.fl_channels == []


def test_missing_channel_at_later_timepoint_clears_previous_fl(make_tcf):
    raw = np.ones((2, 3, 4), dtype=np.uint16) * 13300
    path = make_tcf(timepoints={"0": raw, "1": raw}, fluorescence=True)
    with h5py.File(path, "a") as f:
        del f["Data/3DFL/CH1/1"]
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        assert "CH1" in loader.fl_data
        loader.load_timepoint(1)
        assert list(loader.fl_data) == ["CH0"]


def test_selective_fluorescence_loading_reads_only_requested_channels(make_tcf, monkeypatch):
    path = make_tcf(fluorescence=True)
    original = h5py.Dataset.__array__
    reads = []

    def track(dataset, *args, **kwargs):
        reads.append(dataset.name)
        return original(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__array__", track)
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0, fl_channels=["CH1"])
        assert list(loader.fl_data) == ["CH1"]
        assert reads == ["/Data/3D/000000", "/Data/3DFL/CH1/000000"]
        reads.clear()
        loader.load_timepoint(0, fl_channels=[])
        assert loader.fl_data == {}
        assert reads == ["/Data/3D/000000"]


def test_explicit_missing_channel_preserves_loaded_acquisition(make_tcf):
    raw = np.ones((2, 3, 4), dtype=np.uint16) * 13300
    path = make_tcf(timepoints={"0": raw, "1": raw + 100}, fluorescence=True)
    with h5py.File(path, "a") as file:
        del file["Data/3DFL/CH1/1"]
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        previous = loader.data_3d
        previous_fl = loader.fl_data
        with pytest.raises(TCFFileError, match="CH1.*missing.*1"):
            loader.load_timepoint(1, fl_channels=["CH1"])
        assert loader.data_3d is previous
        assert loader.fl_data is previous_fl
        assert loader.current_timepoint == "0"


@pytest.mark.parametrize("channels", ["CH0", ["unknown"], ["CH0", "CH0"]])
def test_invalid_selective_channels_rejected(make_tcf, channels):
    with TCFFileLoader(make_tcf(fluorescence=True)) as loader:
        with pytest.raises(ValueError):
            loader.load_timepoint(0, fl_channels=channels)


def test_block_reads_preserve_eager_state_and_native_fluorescence(make_tcf):
    raw = np.arange(120, dtype=np.uint16).reshape(5, 4, 6) + 13300
    path = make_tcf(timepoints={"0": raw, "1": raw + 100}, fluorescence=True)
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        previous = loader.data_3d
        blocks = list(loader.iter_volume_blocks(1, roi=((1, 5), (1, 3), (2, 6)), block_depth=2))
        assert len(blocks) == 2
        assert blocks[0].dtype == np.float64
        np.testing.assert_allclose(np.concatenate(blocks), (raw + 100)[1:5, 1:3, 2:6] / 10000)
        fluorescence = list(loader.iter_volume_blocks(1, channel="CH1", block_depth=2))
        assert fluorescence[0].dtype == np.uint16
        np.testing.assert_array_equal(np.concatenate(fluorescence), np.arange(120).reshape(5, 4, 6))
        assert loader.data_3d is previous
        assert loader.current_timepoint == "0"


def test_streaming_float_scale_cache_is_dataset_wide_and_cleared_on_close(make_tcf, monkeypatch):
    raw = np.ones((5, 3, 4), dtype=np.float32) * 40
    raw[-1] = 14000
    path = make_tcf(timepoints={"0": raw})
    original = h5py.Dataset.__getitem__
    reads = []

    def track(dataset, selection, *args, **kwargs):
        reads.append(selection)
        return original(dataset, selection, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", track)
    loader = TCFFileLoader(path)
    with loader:
        roi = ((0, 1), (0, 3), (0, 4))
        block, = loader.iter_volume_blocks(0, roi=roi, block_depth=2)
        np.testing.assert_array_equal(block, np.full((1, 3, 4), 0.004))
        assert len(reads) == 4
        reads.clear()
        list(loader.iter_volume_blocks(0, roi=roi, block_depth=2))
        assert len(reads) == 1
        with pytest.raises(RuntimeError, match="No data loaded"):
            _ = loader.data_3d
    assert loader._ri_divisors == {}


@pytest.mark.parametrize("idx", [True, 0.0, "0"])
def test_loader_rejects_noninteger_timepoint_indices(make_tcf, idx):
    with TCFFileLoader(make_tcf()) as loader:
        with pytest.raises(ValueError, match="integer"):
            loader.load_timepoint(idx)
        with pytest.raises(ValueError, match="integer"):
            list(loader.iter_volume_blocks(idx))
