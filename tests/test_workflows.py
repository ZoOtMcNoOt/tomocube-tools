"""End-to-end scientific products, independent of GUI screenshots."""
import json

import h5py
import numpy as np
import pytest
import tifffile
from scipy.io import loadmat

from tomocube import TCFFileLoader, TCFViewer, SliceViewer, register_fl_to_ht
from tomocube.cli import main
from tomocube.processing.alignment import load_alignment
from tomocube.processing.export import export_to_mat, export_to_png_sequence
from tomocube.core.exceptions import TCFFileError
from test_alignment import phantom


@pytest.fixture
def aligned_acquisition(make_tcf, tmp_path, capsys):
    ht, fl, _ = phantom()
    path = make_tcf(timepoints={"2": ht, "10": ht + 0.01}, fluorescence=True)
    with h5py.File(path, "r+") as file:
        for axis, spacing in zip("ZYX", [1.2, 0.7, 0.5]):
            for modality in ("3D", "3DFL"):
                file[f"Data/{modality}"].attrs[f"Resolution{axis}"] = spacing
        file["Data/3DFL/CH0/2"][...] = fl
    report = tmp_path / "registration.json"
    assert main(["register", str(path), str(report), "--fl", "CH0", "--max-shift", "4", "4", "4"]) == 0
    assert json.loads(capsys.readouterr().out)["accepted"]
    return path, report


def test_cli_alignment_replayed_in_scientific_and_display_exports(aligned_acquisition, tmp_path):
    path, report = aligned_acquisition
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0, fl_channels=["CH0"])
        alignment = load_alignment(report, loader, "CH0")
        expected = register_fl_to_ht(loader.fl_data["CH0"], loader.data_3d.shape, loader.reg_params,
                                     "CH0", alignment.z_offset_mode, translation_um=alignment.translation_um)
        native = loader.fl_data["CH0"].copy()
    common = [str(path), "--fl", "CH0", "--registration", str(report)]
    tiff, mat = tmp_path / "registered.tiff", tmp_path / "registered.mat"
    assert main(["tiff", *common, str(tiff)]) == 0
    assert main(["mat", *common, str(mat)]) == 0
    with tifffile.TiffFile(tiff) as file:
        np.testing.assert_array_equal(file.asarray(), expected)
        metadata = json.loads(file.imagej_metadata["Info"])
        assert metadata["spacing_zyx_um"] == [1.2, 0.7, 0.5]
        assert metadata["registration"]["estimate"]["accepted"]
        assert not metadata["normalized"]
    data = loadmat(mat, simplify_cells=True)
    np.testing.assert_array_equal(data["fl_ch0_registered"], expected)
    np.testing.assert_array_equal(data["fl_ch0"], native)
    assert main(["gif", *common, str(tmp_path / "overlay.gif"), "--overlay"]) == 0
    assert main(["png", *common, str(tmp_path / "png")]) == 0
    assert len(list((tmp_path / "png").glob("*.png"))) == expected.shape[0]


@pytest.mark.parametrize("viewer_type", [TCFViewer, SliceViewer])
def test_saved_alignment_matches_every_viewer_axis(aligned_acquisition, viewer_type):
    path, report = aligned_acquisition
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        alignment = load_alignment(report, loader, "CH0")
        expected = register_fl_to_ht(loader.fl_data["CH0"], loader.data_3d.shape, loader.reg_params,
                                     "CH0", alignment.z_offset_mode, translation_um=alignment.translation_um)
    with viewer_type(path, fl_channel="CH0", registration_path=report) as viewer:
        for axis in range(3):
            index = expected.shape[axis] // 2
            np.testing.assert_allclose(viewer._fl_mapper.get_slice(axis, index).data,
                                       np.take(expected, index, axis=axis), atol=1e-4)
        viewer.fig.canvas.draw()
        if viewer_type is TCFViewer:
            before = viewer.loader.data_3d
            with pytest.raises(ValueError, match="does not match"):
                viewer._load_timepoint(1)
            assert viewer.loader.data_3d is before
            assert viewer.loader.current_timepoint == "2"


@pytest.mark.parametrize("command", ["tiff", "mat", "gif", "png"])
def test_mismatched_registration_cannot_publish_output(aligned_acquisition, tmp_path, command):
    path, report = aligned_acquisition
    output = tmp_path / f"wrong.{command}"
    assert main([command, str(path), str(output), "--fl", "CH1", "--registration", str(report)]) == 1
    assert not output.exists()


def test_rejected_alignment_writes_diagnostics_but_cannot_be_used(make_tcf, tmp_path):
    path = make_tcf(fluorescence=True)
    report, output = tmp_path / "rejected.json", tmp_path / "out.tiff"
    assert main(["register", str(path), str(report), "--fl", "CH0"]) == 1
    assert not json.loads(report.read_text())["result"]["accepted"]
    assert main(["tiff", str(path), str(output), "--fl", "CH0", "--registration", str(report)]) == 1
    assert not output.exists()


def test_png_sequence_preserves_prior_output_and_publishes_no_partial_frames(make_tcf, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    directory = tmp_path / "slices"
    directory.mkdir()
    previous = directory / "ht_0000.png"
    previous.write_bytes(b"prior acquisition")
    with TCFFileLoader(make_tcf()) as loader:
        loader.load_timepoint(0)
        with pytest.raises(FileExistsError, match="new or empty"):
            export_to_png_sequence(loader, directory)
        assert previous.read_bytes() == b"prior acquisition"
        original = plt.imsave
        calls = 0
        def fail_after_one(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated disk failure")
            return original(*args, **kwargs)
        monkeypatch.setattr(plt, "imsave", fail_after_one)
        with pytest.raises(OSError, match="disk failure"):
            export_to_png_sequence(loader, tmp_path / "new_sequence")
    assert not (tmp_path / "new_sequence").exists()
    assert not list(tmp_path.glob(".new_sequence-*"))


def test_mat_cannot_export_ht_as_a_fluorescence_channel(make_tcf, tmp_path):
    with TCFFileLoader(make_tcf(fluorescence=True)) as loader:
        loader.load_timepoint(0)
        with pytest.raises(ValueError, match="FL channel"):
            export_to_mat(loader, tmp_path / "mislabelled.mat", fl_channel="HT")


def test_corrupt_projection_does_not_replace_previously_loaded_acquisition(make_tcf):
    raw = np.full((3, 4, 5), 13300, dtype=np.uint16)
    path = make_tcf(timepoints={"0": raw, "1": raw})
    with h5py.File(path, "r+") as file:
        mip = np.full((4, 5), 13341.0)
        mip[0, 0] = np.nan
        file.create_dataset("Data/2DMIP/1", data=mip)
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        previous = loader.data_3d
        with pytest.raises(TCFFileError, match="nonfinite"):
            loader.load_timepoint(1)
        assert loader.data_3d is previous
        assert loader.current_timepoint == "0"
