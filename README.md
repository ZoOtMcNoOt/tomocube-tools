# Tomocube Tools

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python library and CLI for working with Tomocube TCF (Tomocube Cell File) holotomography data.

## Features

- Inspect TCF metadata (`info`) including dimensions, resolutions, instrument, and sidecar metadata when present.
- Explore HT data in interactive orthogonal viewers (`view` and `slice`) with physical units.
- Render 3D volumes in napari (`view3d`) with camera presets, crop controls, layer controls, histogram, FL Z-offset slider, and animation export widgets.
- Register fluorescence (FL) into HT space (`start`, `center`, `auto` modes).
- Export to TIFF, MATLAB `.mat`, PNG sequence (API), and GIF.
- Select acquisition timepoints and fluorescence channels in 2D viewers and exports.
- Preserve independent X/Y/Z calibration and source/timepoint metadata in scientific TIFF and MAT exports.

## Installation

```bash
# From a local clone
pip install -e .

# With 3D viewer extras (napari + animation tooling)
pip install -e ".[3d]"

# All extras
pip install -e ".[all]"
```

Core package requirements come from `pyproject.toml` and include: `h5py`, `numpy`, `scipy`, `matplotlib`, `tifffile`, `imageio`, and `imagecodecs`.

Installation also provides the `tomocube` command, equivalent to `python -m tomocube`.

## Quick Start

```bash
# Show file metadata
python -m tomocube info path/to/file.TCF

# Interactive orthogonal viewer
python -m tomocube view path/to/file.TCF

# Compare the third acquisition and fluorescence channel CH1
tomocube slice path/to/file.TCF --timepoint 2 --fl CH1

# 3D viewer (requires [3d] extras)
python -m tomocube view3d path/to/file.TCF

# Export HT volume as 32-bit TIFF preserving physical RI values
python -m tomocube tiff path/to/file.TCF output.tiff --32bit

# Export the third acquisition (zero-based index)
tomocube tiff path/to/file.TCF timepoint2.tiff --timepoint 2

# Export HT+FL overlay GIF
python -m tomocube gif path/to/file.TCF overlay.gif --overlay --z-offset-mode center

# Choose a fluorescence channel for the overlay
tomocube gif path/to/file.TCF channel1.gif --overlay --fl CH1
```

## CLI Commands

| Command | Purpose |
|---|---|
| `info` | Show metadata for one TCF file |
| `view` | 2D orthogonal HT viewer with optional FL overlay and measurements |
| `slice` | Side-by-side HT / FL / overlay slice viewer |
| `view3d` | 3D napari viewer |
| `tiff` | Export TIFF stack |
| `mat` | Export MATLAB `.mat` |
| `gif` | Export animated GIF (HT only or HT+FL overlay) |

Global flag:
- `-V`, `--verbose`: prints detailed registration diagnostics.

Key option notes:
- `view` and `slice`: `--z-offset-mode` defaults to `start`; `--fl CH1` selects a channel (default: first available). A file path is required. In `view`, press `F` to show the overlay.
- `gif --overlay`: `--z-offset-mode` default is `start`.
- `view3d`: `--z-offset-mode` default is `auto`.
- `tiff`: CLI default is `--32bit` with physical RI values; `--16bit` requires `--normalize`.
- `view`, `slice`, `tiff`, `mat`, and `gif`: `--timepoint N` selects a zero-based acquisition index (default `0`). Numeric acquisition keys are sorted numerically, so `2` precedes `10`.
- `gif`: `--fl CH1` selects a fluorescence channel; combine with `--overlay` to blend it with HT. `--fps` accepts integers from 1 to 100; GIF timing is rounded to 10 ms intervals.
- Invalid 2D viewer and export options are rejected before loading data. Use `tomocube <command> --help` for command-specific usage.

Run `python -m tomocube help` for full CLI help text.

## Registration Behavior

`z-offset-mode` controls FL Z placement:

- `start`: the selected channel's `OffsetZ` places the center of FL slice 0 in HT coordinates.
- `center`: the channel's `OffsetZ` places the geometric center of the FL volume in HT coordinates.
- `auto`: in 2D viewers and registration/export, the FL intensity-weighted Z center aligns with the HT geometric center, ignoring `OffsetZ`. Weights are nonnegative per-plane intensity sums; a volume without positive signal uses its geometric center. The separate 3D viewer uses geometric centering.

The 2D viewers and `register_fl_to_ht` share one physical-coordinate transform. Voxel centers are at `index × spacing`, so each axis's geometric center is `(size - 1) × spacing / 2`; displayed image edges extend half a voxel beyond the first and last centers. Independent X/Y spacings are respected. In YX coordinate order, forward rotation is `[[cos, -sin], [sin, cos]]`, followed by translation in HT micrometers. Resolution metadata determines scaling; the legacy `Scale` attribute is not applied again.

Fluorescence is sampled linearly on the HT grid, including the last plane and single-plane volumes. Positions outside the FL sample centers are zero. The viewer samples only visible planes and retains the original intensities; its manual FL Z adjustment changes display alignment only. Registration rejects nonfinite intensities and invalid calibration rather than producing misleading coordinates.

These corrections change aligned output compared with earlier versions, which inverted XY spacing ratios, omitted the last FL plane, and used a different overlay mapping. The napari 3D viewer has its own native-volume placement and is not covered by the 2D/export agreement described here.

The orthogonal viewer updates shapes, physical extents, contrast controls, and overlays when the timepoint changes. Press `N` or click **Channel** to show the next fluorescence channel with its own calibration. Missing fluorescence is marked unavailable, and a failed acquisition load preserves the previously displayed data. Arrow keys move a focused position slider by one sample. Navigation controls are hidden for axes containing a single sample.

Defaults by entry point:

| Entry point | Default |
|---|---|
| `python -m tomocube view` | `start` |
| `python -m tomocube slice` | `start` |
| `python -m tomocube gif --overlay` | `start` |
| `python -m tomocube view3d` | `auto` |
| `register_fl_to_ht(...)` | `start` |

## Python API

```python
import h5py
from tomocube import TCFFile, TCFFileLoader, register_fl_to_ht, export_to_tiff

with h5py.File("path/to/file.TCF", "r") as f:
    info = TCFFile.from_hdf5(f)
    print(info.ht_shape, info.ht_resolution, info.fl_channels)

with TCFFileLoader("path/to/file.TCF") as loader:
    loader.load_timepoint(0)
    ht = loader.data_3d              # (Z, Y, X), physical RI units
    fl = loader.fl_data.get("CH0")   # raw FL volume if present

    if fl is not None:
        fl_reg = register_fl_to_ht(fl, ht.shape, loader.reg_params, channel="CH0")

    # API default differs from CLI:
    # export_to_tiff(...): bit_depth=16, normalize=True by default
    export_to_tiff(loader, "output.tiff", bit_depth=32, normalize=False)
```

## TCF Structure (General)

TCF is HDF5-based. Typical paths:

```text
Data/3D/<timepoint>          HT volume (Z, Y, X)
Data/2DMIP/<timepoint>       optional MIP
Data/3DFL/<channel>/<tp>     optional FL volume
Info/Device                  optics/device metadata
Info/MetaData/...            embedded config/experiment metadata
```

The loader normalizes HT values to physical RI units when files store scaled integer-like values. Metadata accepts scalars and singleton arrays. Missing resolution attributes use the documented instrument defaults with a warning; invalid supplied calibration raises a clear error instead of silently substituting a different spacing.

## Development and Verification

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m build
```

The regression suite creates synthetic TCF acquisitions and reads exported TIFF, MAT, GIF, and PNG files back to verify values, calibration, selection, metadata, and error behavior. Analytic fiducials cover anisotropic scaling, physical rotation/translation, channel offsets, and boundary planes. Matplotlib runs with the Agg backend to test rendered figures, keyboard/mouse callbacks, acquisition changes, and resource cleanup. These tests require no experimental data or GUI extras. CI runs on Linux with Python 3.10 and 3.14 and Windows with Python 3.12, then builds and installs the wheel.

Interactive viewer behavior and instrument-specific registration should also be checked with representative acquisitions before research use; the synthetic suite does not establish registration accuracy on experimental data.

## Documentation

- [INSTRUCTIONS.md](INSTRUCTIONS.md): detailed command and workflow reference.
- [DATA_ANALYSIS.md](DATA_ANALYSIS.md): general Tomocube file/data format reference.

## License

MIT
