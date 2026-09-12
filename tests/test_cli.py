import os
import csv
import io
import json
import shutil
import subprocess
import sys
import sysconfig
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile
from PIL import Image
from scipy.io import loadmat

from tomocube import __version__
from tomocube.__main__ import main


@pytest.fixture
def cli(monkeypatch):
    def run(*args):
        monkeypatch.setattr(sys, "argv", ["tomocube", *map(str, args)])
        return main()
    return run


@pytest.mark.parametrize("command", ["tiff", "mat", "gif", "view", "slice", "view3d", "png", "analyze"])
def test_export_command_help(cli, capsys, command):
    assert cli(command, "--help") == 0
    assert "--timepoint" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["view", "slice"])
@pytest.mark.parametrize("options", [["--bogus"], ["--fl"], ["--timepoint", "-1"],
                                    ["--z-offset-mode", "guess"], ["--timepoint", "1.5"]])
def test_viewer_rejects_invalid_arguments_before_loading(cli, capsys, command, options):
    assert cli(command, "nonexistent.TCF", *options) == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("command,class_name", [("view", "TCFViewer"), ("slice", "SliceViewer")])
def test_cli_passes_viewer_selection(cli, monkeypatch, command, class_name):
    import tomocube.viewer

    received = {}

    class Viewer:
        def __init__(self, path, **kwargs):
            received.update(path=path, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            received["closed"] = True

        def show(self):
            received["shown"] = True

    monkeypatch.setattr(tomocube.viewer, class_name, Viewer)
    assert cli(command, "--timepoint", "2", "sample file.TCF", "--fl", "CH1",
               "--z-offset-mode", "center") == 0
    assert received == {"path": "sample file.TCF", "timepoint": 2, "fl_channel": "CH1",
                        "z_offset_mode": "center", "shown": True, "closed": True}


@pytest.mark.parametrize("command", ["view", "slice"])
def test_cli_reports_invalid_viewer_selection_without_traceback(cli, make_tcf, capsys, command):
    assert cli(command, make_tcf(), "--timepoint", "100") == 1
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("command,options", [
    ("tiff", ["--bogus"]),
    ("tiff", ["--fl"]),
    ("tiff", ["--16bit"]),
    ("tiff", ["--16bit", "--32bit", "--normalize"]),
    ("mat", ["--timepoint", "-1"]),
    ("mat", ["--timepoint", "nope"]),
    ("mat", ["--timepoint"]),
    ("mat", ["--unexpected"]),
    ("gif", ["--fps"]),
    ("gif", ["--fps", "fast"]),
    ("gif", ["--fps", "0"]),
    ("gif", ["--fps", "101"]),
    ("gif", ["--axis", "q"]),
    ("gif", ["--z-offset-mode", "guess"]),
])
def test_export_rejects_invalid_arguments_without_writing(cli, make_tcf, tmp_path, capsys, command, options):
    source = make_tcf()
    output = tmp_path / f"result.{command}"
    assert cli(command, source, output, *options) == 2
    assert not output.exists()
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("command", ["tiff", "mat", "gif"])
def test_export_rejects_missing_timepoint_before_writing(cli, make_tcf, tmp_path, command):
    output = tmp_path / f"out.{command}"
    assert cli(command, make_tcf(), output, "--timepoint", "100") == 1
    assert not output.exists()


@pytest.mark.parametrize("command", ["tiff", "mat"])
def test_cli_exports_selected_timepoint(cli, make_tcf, tmp_path, command):
    volumes = {str(i): np.full((2, 3, 4), 13300 + i, dtype=np.uint16) for i in (0, 1, 2, 10)}
    path = make_tcf(timepoints=volumes, scalar_attrs=True)
    output = tmp_path / f"selected timepoint.{command}"
    assert cli(command, path, "--timepoint", "2", output) == 0
    if command == "tiff":
        with tifffile.TiffFile(output) as tif:
            data = tif.asarray()
            assert json.loads(tif.imagej_metadata["Info"])["timepoint"] == "2"
    else:
        mat = loadmat(output, simplify_cells=True)
        data = mat["ht_3d"]
        assert str(mat["metadata"]["timepoint"]) == "2"
    np.testing.assert_allclose(data, 1.3302)


def test_default_output_distinguishes_timepoint(cli, make_tcf, tmp_path, monkeypatch):
    raw = np.full((2, 3, 4), 13300, dtype=np.uint16)
    source = make_tcf(timepoints={"0": raw, "1": raw})
    monkeypatch.chdir(tmp_path)
    assert cli("tiff", source, "--timepoint", "1") == 0
    assert (tmp_path / "sample acquisition_t1_ht.tiff").exists()


@pytest.mark.parametrize("overlay", [False, True])
def test_gif_accepts_selected_fl_channel(cli, make_tcf, tmp_path, overlay):
    path = make_tcf(fluorescence=True)
    with h5py.File(path, "a") as f:
        del f["Data/3DFL/CH0"]
    output = tmp_path / "selected channel.gif"
    options = ["--overlay"] if overlay else []
    assert cli("gif", path, output, "--fl", "CH1", *options) == 0
    with Image.open(output) as gif:
        assert gif.size == (5, 4)


def test_cli_and_distribution_versions_match(cli, capsys):
    assert __version__ == version("tomocube-tools")
    assert cli("--version") == 0
    assert f"v{__version__}" in capsys.readouterr().out


def test_full_help_shows_version_and_export_selection(cli, capsys):
    assert cli("--help") == 0
    output = capsys.readouterr().out
    assert f"v{__version__}" in output
    assert "--timepoint N" in output
    assert "gif --fl CH1" in output


def test_installed_module_and_console_entrypoints(make_tcf, tmp_path):
    source = make_tcf(scalar_attrs=True)
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    console = Path(sysconfig.get_path("scripts")) / ("tomocube.exe" if os.name == "nt" else "tomocube")
    for command in (
        [sys.executable, "-m", "tomocube", "info", str(source)],
        [str(console), "--version"],
    ):
        result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("command,options", [
    ("info", ["--timepoint", "0"]), ("info", ["--bogus"]),
    ("view3d", ["--render", "guess"]), ("view3d", ["--screenshot"]),
    ("view3d", ["--bogus"]), ("view3d", ["--timepoint", "-1"]),
    ("view3d", ["--fl"]), ("view3d", ["--fl", ""]),
    ("gif", ["--z-offset-mode", "center"]),
    ("png", ["--prefix", "../escape"]), ("png", ["--prefix", "..\\escape"]),
    ("png", ["--cmap", "not-a-colormap"]), ("png", ["--vmin", "nan"]),
    ("png", ["--vmin", "2", "--vmax", "1"]),
    ("analyze", ["--all-timepoints", "--timepoint", "0"]),
    ("analyze", ["--threshold", "inf"]), ("analyze", ["--block-depth", "0"]),
    ("analyze", ["--roi", "0", "0", "0", "1", "0", "1"]),
    ("analyze", ["--roi", "0", "1", "0", "1", "0"]),
    ("analyze", ["--channel", "ht", "--channel", "HT"]),
    ("analyze", ["--channel", " "]),
])
def test_new_commands_reject_invalid_arguments_before_loading(cli, capsys, command, options):
    assert cli(command, "nonexistent.TCF", *options) == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert captured.out == ""
    assert "Traceback" not in captured.err


def test_view3d_forwards_every_selection(cli, monkeypatch):
    import tomocube.viewer.viewer_3d

    received = {}
    monkeypatch.setattr(tomocube.viewer.viewer_3d, "view_3d", lambda path, **kwargs: received.update(path=path, **kwargs))
    assert cli("view3d", "--timepoint", "2", "sample file.TCF", "--fl", "CH1", "--slices",
               "--render", "average", "--screenshot", "view.png", "--z-offset-mode", "center") == 0
    assert received == dict(path="sample file.TCF", timepoint=2, fl_channel="CH1", show_slices=True,
                            rendering="average", screenshot="view.png", z_offset_mode="center")


def test_info_json_contains_inventory_and_related_metadata(cli, make_tcf, tmp_path, capsys, monkeypatch):
    source = make_tcf(fluorescence=True, scalar_attrs=True)
    (tmp_path / ".experiment").write_text(json.dumps({"experimentTitle": "Calibration study"}), encoding="utf-8")
    (tmp_path / ".vessel").write_text(json.dumps({"vessel": {"name": "Plate A"}}), encoding="utf-8")
    profiles = tmp_path / "profile"
    profiles.mkdir()
    (profiles / "Cell.img.prf").write_text("[DefaultParameters]\nDefaultStep=0.3\n", encoding="utf-8")

    def no_volume_reads(*args, **kwargs):
        raise AssertionError("info must read headers only")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", no_volume_reads)
    assert cli("-V", "info", source, "--json") == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["source"] == str(source.resolve())
    assert report["metadata"]["device_model"] == "HTX"
    assert report["metadata"]["magnification"] == 60
    assert report["volumes"][0]["shape_zyx"] == [3, 4, 5]
    assert report["volumes"][0]["spacing_zyx_um"] == [1.5, 0.5, 0.25]
    assert report["related_metadata"]["experiment"]["experimentTitle"] == "Calibration study"
    assert report["related_metadata"]["vessel"]["vessel"]["name"] == "Plate A"
    assert "img" in report["related_metadata"]["profiles"]
    assert "Loading:" not in captured.out


def test_info_human_output_retains_units_optics_and_instrument(cli, make_tcf, capsys):
    assert cli("info", make_tcf()) == 0
    output = capsys.readouterr().out
    assert "ZYX" in output and "FOV" in output and "um" in output
    assert "HTX" in output and "TEST-001" in output and "magnification" in output


def test_analyze_batches_paths_with_all_timepoints_channels_and_roi(cli, make_tcf, tmp_path, capsys):
    raw = np.arange(60, dtype=np.uint16).reshape(3, 4, 5) + 13300
    first = make_tcf(timepoints={"0": raw, "10": raw + 100}, fluorescence=True)
    second = tmp_path / "second acquisition.TCF"
    shutil.copyfile(first, second)
    assert cli("analyze", first, "--all-timepoints", second, "--channel", "ht", "--channel", "CH1",
               "--roi", "1", "3", "1", "4", "0", "4", "--threshold", "1.333", "--block-depth", "1") == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 8
    assert {row["source"] for row in rows} == {str(first.resolve()), str(second.resolve())}
    assert {(row["acquisition_key"], row["channel"]) for row in rows} == {
        ("0", "HT"), ("10", "HT"), ("0", "CH1"), ("10", "CH1")}
    ht = rows[0]
    expected = raw[1:3, 1:4, 0:4] / 10000.0
    assert ht["roi_zyx"] == [[1, 3], [1, 4], [0, 4]]
    assert ht["mean"] == pytest.approx(expected.mean())
    assert ht["volume_um3"] == pytest.approx(expected.size * 1.5 * 0.5 * 0.25)
    assert ht["selected_voxel_count"] == np.count_nonzero(expected >= 1.333)


def test_analyze_csv_exposes_missing_channels_and_native_roi(cli, make_tcf, capsys):
    source = make_tcf(fluorescence=True)
    with h5py.File(source, "a") as file:
        file.move("Data/3DFL/CH1/000000", "Data/3DFL/CH1/orphan")
    assert cli("analyze", source, "--format", "csv", "--channel", "HT", "--channel", "CH1") == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 2
    assert json.loads(rows[0]["roi_zyx"]) == [[0, 3], [0, 4], [0, 5]]
    assert rows[1]["status"] == "missing" and rows[1]["mean"] == ""


def test_analyze_defaults_to_first_acquisition_and_allows_selection(cli, make_tcf, capsys):
    raw = np.full((2, 3, 4), 13300, dtype=np.uint16)
    source = make_tcf(timepoints={"2": raw, "10": raw + 100})
    for arguments, expected in [((), "2"), (("--timepoint", "1"), "10")]:
        assert cli("analyze", source, *arguments) == 0
        rows = json.loads(capsys.readouterr().out)
        assert len(rows) == 1 and rows[0]["acquisition_key"] == expected


def test_analyze_writes_one_complete_report_without_stdout(cli, make_tcf, tmp_path, capsys):
    source = make_tcf()
    output = tmp_path / "results" / "measurements.json"
    assert cli("analyze", source, "--output", output) == 0
    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text(encoding="utf-8"))[0]["source"] == str(source.resolve())
    assert list(output.parent.iterdir()) == [output]


def test_analyze_preserves_existing_report_if_a_batch_input_fails(cli, make_tcf, tmp_path, capsys):
    output = tmp_path / "measurements.json"
    output.write_text("previous report", encoding="utf-8")
    assert cli("analyze", make_tcf(), tmp_path / "missing.TCF", "--output", output) == 1
    assert output.read_text(encoding="utf-8") == "previous report"
    captured = capsys.readouterr()
    assert captured.out == "" and "error:" in captured.err


def test_analyze_preserves_existing_report_if_atomic_publish_fails(cli, make_tcf, tmp_path, monkeypatch, capsys):
    import tomocube.processing.outputs

    output = tmp_path / "measurements.json"
    output.write_text("previous report", encoding="utf-8")
    source = make_tcf()
    before = set(tmp_path.iterdir())

    def fail_replace(*args):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(tomocube.processing.outputs.os, "replace", fail_replace)
    assert cli("analyze", source, "--output", output) == 1
    assert output.read_text(encoding="utf-8") == "previous report"
    assert set(tmp_path.iterdir()) == before
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("alias", [False, True])
def test_analyze_cannot_replace_input_or_hardlink(cli, make_tcf, tmp_path, alias):
    source = make_tcf()
    before = source.read_bytes()
    output = tmp_path / "alias.json" if alias else source
    if alias:
        output.hardlink_to(source)
    assert cli("analyze", source, "--output", output) == 1
    assert source.read_bytes() == before and output.read_bytes() == before


def test_png_exports_selected_native_fluorescence_sequence(cli, make_tcf, tmp_path, capsys):
    source = make_tcf(fluorescence=True)
    output = tmp_path / "png frames"
    assert cli("png", source, "--fl", "CH1", output, "--prefix", "channel1", "--cmap", "viridis",
               "--vmin", "0", "--vmax", "59") == 0
    paths = [Path(line) for line in capsys.readouterr().out.splitlines()]
    assert paths == [output / f"channel1_{z:04d}.png" for z in range(3)]
    for path in paths:
        with Image.open(path) as frame:
            assert frame.size == (5, 4)


def test_export_stdout_only_contains_the_result_path(cli, make_tcf, tmp_path, capsys):
    output = tmp_path / "volume.tiff"
    assert cli("tiff", make_tcf(), output) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == str(output)
    assert "Loading:" in captured.err


@pytest.mark.parametrize("command,options", [("tiff", []), ("mat", ["--no-fl"]), ("gif", []), ("png", [])])
def test_ht_only_exports_do_not_read_fluorescence_pixels(cli, make_tcf, tmp_path, monkeypatch, command, options):
    source = make_tcf(fluorescence=True)
    read_dataset = h5py.Dataset.__getitem__

    def protect_fluorescence(dataset, key):
        if dataset.name.startswith("/Data/3DFL/"):
            raise AssertionError("HT-only export must not read fluorescence arrays")
        return read_dataset(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", protect_fluorescence)
    assert cli(command, source, tmp_path / f"export.{command}", *options) == 0


@pytest.mark.parametrize("args", [("--verbose", "info"), ("info", "--verbose")])
def test_verbose_works_before_or_after_command_without_polluting_json(cli, make_tcf, capsys, args):
    assert cli(*args, make_tcf(), "--json") == 0
    assert json.loads(capsys.readouterr().out)["axis_order"] == "ZYX"


def test_help_command_describes_a_subcommand(cli, capsys):
    assert cli("help", "analyze") == 0
    assert "--all-timepoints" in capsys.readouterr().out
