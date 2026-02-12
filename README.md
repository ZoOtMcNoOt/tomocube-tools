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

## Quick Start

```bash
# Show file metadata
python -m tomocube info path/to/file.TCF

# Interactive orthogonal viewer
python -m tomocube view path/to/file.TCF

# 3D viewer (requires [3d] extras)
python -m tomocube view3d path/to/file.TCF

# Export HT volume as 32-bit TIFF preserving physical RI values
python -m tomocube tiff path/to/file.TCF output.tiff --32bit

# Export HT+FL overlay GIF
python -m tomocube gif path/to/file.TCF overlay.gif --overlay --z-offset-mode center
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
- `view` and `slice`: `--z-offset-mode` default is `start`.
- `gif --overlay`: `--z-offset-mode` default is `start`.
- `view3d`: `--z-offset-mode` default is `auto`.
- `tiff`: CLI default is `--32bit` with physical RI values; `--16bit` requires `--normalize`.

Run `python -m tomocube help` for full CLI help text.

## Registration Behavior

`z-offset-mode` controls FL Z placement:

- `start`: file `OffsetZ` is treated as where FL slice 0 starts in HT coordinates.
- `center`: file `OffsetZ` is treated as FL volume center in HT coordinates.
- `auto`: FL is centered in HT Z range (3D viewer), or FL signal center is aligned to HT center for `register_fl_to_ht`.

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

The loader normalizes HT values to physical RI units when files store scaled integer-like values.

## Documentation

- [INSTRUCTIONS.md](INSTRUCTIONS.md): detailed command and workflow reference.
- [DATA_ANALYSIS.md](DATA_ANALYSIS.md): general Tomocube file/data format reference.

## License

MIT
