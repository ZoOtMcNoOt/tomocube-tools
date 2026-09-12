import os
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


@pytest.mark.parametrize("command", ["tiff", "mat", "gif"])
def test_export_command_help(cli, capsys, command):
    assert cli(command, "--help") == 0
    assert "--timepoint" in capsys.readouterr().out


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
            assert "timepoint 2" in tif.imagej_metadata["Info"]
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
