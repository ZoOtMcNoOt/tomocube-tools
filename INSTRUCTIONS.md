# Tomocube Tools Instructions

Reference guide for the `tomocube-tools` CLI and Python API.

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
# napari 3D viewer + animation dependencies
pip install -e ".[3d]"

# all extras
pip install -e ".[all]"
```

## CLI Basics

General form:

```bash
python -m tomocube [global-options] <command> <file.TCF> [command-options]
```

The installed `tomocube` command accepts the same arguments.

Global options:

- `-V`, `--verbose`: verbose registration/debug output.

Help/version entry points:

- `python -m tomocube help`
- `python -m tomocube --help`
- `python -m tomocube --version`
- `python -m tomocube tiff --help` (also `mat --help` and `gif --help`)

For `tiff`, `mat`, and `gif`, `--timepoint N` selects a zero-based index into the available HT acquisitions. Numeric keys sort numerically (`0`, `1`, `2`, `10`); the default index is `0`. Missing or invalid option values are reported before exports begin. Out-of-range indices produce an error without creating an output file.

## Commands

### `info`

Show TCF metadata.

```bash
python -m tomocube info path/to/file.TCF
```

Includes:

- HT dimensions/resolution/FOV/RI range
- fluorescence channels and sizes (if present)
- optics metadata (magnification, NA, medium RI)
- detected instrument identifiers (if embedded)
- related sidecar metadata (`.experiment`, `.vessel`, `profile/*.prf`) when present

### `view`

Orthogonal 2D HT viewer with optional FL overlays and measurement tools.

```bash
python -m tomocube view path/to/file.TCF [--z-offset-mode start|center|auto]
```

Default `z-offset-mode` for `view`: `start`.

If no path is provided, the legacy viewer fallback path selection logic is used.

### `slice`

Side-by-side HT / FL / overlay slice viewer.

```bash
python -m tomocube slice path/to/file.TCF [--z-offset-mode start|center|auto]
```

Default `z-offset-mode` for `slice`: `start`.

If no path is provided, fallback file discovery logic is used.

### `view3d`

3D napari viewer.

```bash
python -m tomocube view3d path/to/file.TCF [options]
```

Options:

- `--slices`: start in 2D slice mode.
- `--render mip|attenuated_mip|minip|average`: initial rendering mode.
- `--z-offset-mode auto|start|center`: FL alignment mode.
- `--screenshot <file>`: save screenshot path.

Default `z-offset-mode` for `view3d`: `auto`.

### `tiff`

Export HT or FL stack to TIFF.

```bash
python -m tomocube tiff path/to/file.TCF [output.tiff] [options]
```

Options:

- `--fl <channel>`: export FL channel (example: `CH0`) instead of HT.
- `--32bit`: 32-bit float output (CLI default).
- `--16bit`: 16-bit output (requires `--normalize`).
- `--normalize`: normalize to display range before export.
- `--timepoint <N>`: export one acquisition by zero-based index (default `0`).

Notes:

- CLI default behavior preserves HT physical RI values (`--32bit` without normalization).
- If output path is omitted, default is `<stem>_<channel>.tiff`.
- A nonzero timepoint index adds `_t<N>` before the channel suffix, for example `<stem>_t2_ht.tiff`.
- X, Y, and Z spacing are preserved independently in ImageJ metadata. The ImageJ `Info` field records the source filename, acquisition key, and normalization information.
- Constant volumes normalize to zero without producing NaN values. The Python API also rejects `bit_depth=16, normalize=False`.

### `mat`

Export to MATLAB `.mat`.

```bash
python -m tomocube mat path/to/file.TCF [output.mat] [--no-fl] [--timepoint N]
```

Notes:

- If output path is omitted, default is `<stem>.mat`.
- `--no-fl` excludes fluorescence volumes.
- A nonzero `--timepoint N` changes the default filename to `<stem>_t<N>.mat`.

MAT keys written by current exporter:

- `ht_3d`
- `ht_mip`
- `fl_ch0`, `fl_ch1`, ... (if included)
- `metadata` (dict)
- `resolution` (dict)

`metadata` includes the source filename and selected acquisition key (`timepoint`). Unavailable optical values such as magnification, numerical aperture, or medium RI are omitted rather than replaced with invented values.

### `gif`

Export GIF animation.

```bash
python -m tomocube gif path/to/file.TCF [output.gif] [options]
```

Options:

- `--overlay`: export HT+FL overlay animation.
- `--fps <N>`: integer frame rate from 1 to 100 (default `10`), rounded to the nearest 10 ms frame interval supported by GIF.
- `--axis z|y|x`: animation axis (default `z`).
- `--z-offset-mode start|center|auto`: used for overlay mode.
- `--timepoint <N>`: zero-based acquisition index (default `0`).
- `--fl <channel>`: select an FL channel for a fluorescence-only GIF or an HT+FL overlay.

Notes:

- Overlay mode uses FL channel `CH0` unless `--fl` is supplied.
- Default output naming is `<stem>_<axis>.gif` (non-overlay) or `<stem>_overlay.gif` (overlay).
- A selected nonzero timepoint and explicit FL channel add `_t<N>` and `_<channel>` before the final suffix.
- Default `z-offset-mode` for overlay exports: `start`.

## Registration Modes

`z-offset-mode` meanings:

- `start`: file `OffsetZ` is FL slice-0 start in HT space.
- `center`: file `OffsetZ` is FL center in HT space.
- `auto`: alignment heuristic.

Defaults by path:

| Path | Default mode |
|---|---|
| `view` | `start` |
| `slice` | `start` |
| `gif --overlay` | `start` |
| `view3d` | `auto` |
| `register_fl_to_ht(...)` API | `start` |

## Viewer Controls

### `view` (2D orthogonal viewer)

Keys:

- Arrow keys: move active slider.
- `Home` / `End`: min/max active slider.
- `A`: auto contrast (slice).
- `G`: global contrast.
- `R`: reset view.
- `I`: invert colormap.
- `F`: toggle FL overlay.
- `D`: distance measurement.
- `P`: area/polygon measurement.
- `C`: clear measurements.
- `M`: save MIP PNG.
- `1`-`6`: colormap selection.
- `Q` / `Escape`: quit or cancel active measurement.

Mouse:

- Scroll in XY view: move Z.
- Scroll in XZ view: move Y.
- Click in views: move crosshair position.

UI buttons also expose save/reset/contrast actions.

### `slice`

Keys:

- Arrow keys: move active slider.
- `Home` / `End`: min/max active slider.
- `Q` / `Escape`: quit.

### `view3d` (napari)

Dock panels:

- Left: `Camera`, `Crop`
- Right: `Layers`, `Histogram`, `FL Z` (when FL exists), `Animation`

Camera shortcuts:

- `1`-`6`: top/bottom/front/back/left/right
- `0`: isometric
- `R`: reset
- `F`: fit
- `+` / `=` and `-`: zoom

Animation export is provided from the `Animation` dock (turntable and slice sweep, GIF/MP4).

## Python API Quick Reference

```python
import h5py
from tomocube import (
    TCFFile,
    TCFFileLoader,
    register_fl_to_ht,
    export_to_tiff,
    export_to_mat,
    export_to_gif,
)

with h5py.File("path/to/file.TCF", "r") as f:
    meta = TCFFile.from_hdf5(f)
    print(meta.ht_shape, meta.has_fluorescence, meta.fl_channels)

with TCFFileLoader("path/to/file.TCF") as loader:
    loader.load_timepoint(0)
    ht = loader.data_3d
    fl = loader.fl_data.get("CH0")

    if fl is not None:
        fl_reg = register_fl_to_ht(fl, ht.shape, loader.reg_params, channel="CH0")

    # API defaults differ from CLI for TIFF:
    # bit_depth=16, normalize=True
    export_to_tiff(loader, "out.tiff", bit_depth=32, normalize=False)
    export_to_mat(loader, "out.mat", include_fl=True)
    export_to_gif(loader, "out.gif", axis="z", fps=10)
```

## Troubleshooting

### `view3d` fails with napari import error

Install 3D extras:

```bash
pip install -e ".[3d]"
```

### `tiff` with `--16bit` fails

`--16bit` requires `--normalize` by design.

```bash
python -m tomocube tiff file.TCF out.tiff --16bit --normalize
```

### Overlay GIF reports missing fluorescence

The selected acquisition has no FL data or lacks the requested channel (`CH0` by default).
Check with:

```bash
python -m tomocube info file.TCF
```

Select an available channel explicitly, for example `tomocube gif file.TCF --overlay --fl CH1`.

### Invalid resolution metadata

Resolution values must be positive, finite numbers stored as scalars or singleton arrays. Errors identify the HDF5 path and attribute. Missing attributes still use documented defaults with a warning; present but invalid values are rejected to avoid changing physical calibration silently.

### File cannot be opened

Validate that the TCF is readable HDF5 and has expected paths like `Data/3D/<timepoint>`.
