import numpy as np
import pytest
import tifffile
from PIL import Image
from scipy.io import loadmat

from tomocube import TCFFileLoader, export_overlay_gif, export_to_gif, export_to_mat, export_to_png_sequence, export_to_tiff


@pytest.mark.parametrize("channel", ["ht", "CH1"])
def test_tiff_preserves_data_and_xyz_spacing(make_tcf, tmp_path, channel):
    with TCFFileLoader(make_tcf(fluorescence=True)) as loader:
        loader.load_timepoint(0)
        output = export_to_tiff(loader, tmp_path / "volume.tiff", channel=channel, bit_depth=32, normalize=False)
        expected = loader.data_3d if channel == "ht" else loader.fl_data[channel]
        spacing = loader.tcf_info.ht_resolution if channel == "ht" else loader.tcf_info.fl_resolution
        with tifffile.TiffFile(output) as tif:
            np.testing.assert_array_equal(tif.asarray(), expected)
            assert tif.series[0].axes == "ZYX"
            page = tif.pages[0]
            for tag, value in zip(("XResolution", "YResolution"), (spacing[2], spacing[1])):
                numerator, denominator = page.tags[tag].value
                assert denominator / numerator == pytest.approx(value)
            assert tif.imagej_metadata["spacing"] == spacing[0]
            assert tif.imagej_metadata["unit"] == "um"
            assert loader.tcf_path.name in tif.imagej_metadata["Info"]


@pytest.mark.parametrize("bit_depth", [16, 32])
@pytest.mark.parametrize("value", [0.0, 1.33])
def test_normalized_constant_tiff_has_no_nan(make_tcf, tmp_path, bit_depth, value):
    path = make_tcf(timepoints={"0": np.full((2, 3, 4), value, dtype=np.float32)})
    with TCFFileLoader(path) as loader:
        loader.load_timepoint(0)
        with np.errstate(all="raise"):
            output = export_to_tiff(loader, tmp_path / "constant.tif", bit_depth=bit_depth, normalize=True)
        np.testing.assert_array_equal(tifffile.imread(output), np.zeros((2, 3, 4)))


def test_tiff_rejects_lossy_16bit_request(make_tcf, tmp_path):
    output = tmp_path / "existing.tiff"
    output.write_bytes(b"existing data")
    with TCFFileLoader(make_tcf()) as loader:
        loader.load_timepoint(0)
        with pytest.raises(ValueError, match="normaliz"):
            export_to_tiff(loader, output, bit_depth=16, normalize=False)
    assert output.read_bytes() == b"existing data"


def test_mat_roundtrip_without_optional_device_metadata(make_tcf, tmp_path):
    with TCFFileLoader(make_tcf(device=False)) as loader:
        loader.load_timepoint(0)
        output = export_to_mat(loader, tmp_path / "volume.mat")
        result = loadmat(output, simplify_cells=True)
        np.testing.assert_array_equal(result["ht_3d"], loader.data_3d)
        np.testing.assert_array_equal(result["ht_mip"], loader.data_mip)
        assert "magnification" not in result["metadata"]
        assert "medium_ri" not in result["metadata"]
        assert result["resolution"]["ht_res_y_um"] == 0.5


@pytest.mark.parametrize("axis,size,frames", [("z", (5, 4), 3), ("y", (5, 3), 4), ("x", (4, 3), 5)])
def test_gif_can_be_opened_with_expected_frames(make_tcf, tmp_path, axis, size, frames):
    with TCFFileLoader(make_tcf()) as loader:
        loader.load_timepoint(0)
        output = export_to_gif(loader, tmp_path / "volume.gif", axis=axis, fps=10)
    with Image.open(output) as gif:
        assert gif.size == size
        assert gif.n_frames == frames
        assert gif.info["duration"] == 100


@pytest.mark.parametrize("overlay", [False, True])
def test_constant_gif_has_finite_frames(make_tcf, tmp_path, overlay):
    raw = np.full((3, 4, 5), 13300, dtype=np.uint16)
    with TCFFileLoader(make_tcf(timepoints={"0": raw}, fluorescence=True)) as loader:
        loader.load_timepoint(0)
        export = export_overlay_gif if overlay else export_to_gif
        with np.errstate(all="raise"):
            output = export(loader, tmp_path / "constant.gif")
    with Image.open(output) as gif:
        assert gif.size == (5, 4)


@pytest.mark.parametrize("fps", [0, -1, np.inf, np.nan, 101])
@pytest.mark.parametrize("export", [export_to_gif, export_overlay_gif])
def test_invalid_gif_rate_rejected_before_writing(make_tcf, tmp_path, fps, export):
    output = tmp_path / "invalid.gif"
    with TCFFileLoader(make_tcf(fluorescence=True)) as loader:
        loader.load_timepoint(0)
        with pytest.raises(ValueError, match="fps"):
            export(loader, output, fps=fps)
    assert not output.exists()


def test_png_sequence_roundtrip(make_tcf, tmp_path):
    with TCFFileLoader(make_tcf()) as loader:
        loader.load_timepoint(0)
        outputs = export_to_png_sequence(loader, tmp_path / "slices")
    assert len(outputs) == 3
    for output in outputs:
        with Image.open(output) as png:
            assert png.size == (5, 4)
