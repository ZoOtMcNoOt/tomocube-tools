# Tomocube Tools

Python library and CLI for inspecting, viewing, measuring and exporting Tomocube TCF holotomography acquisitions. Arrays use **ZYX** order; physical coordinates and spacing use micrometers.

## Install

```bash
pip install -e .
pip install -e ".[3d]"   # optional napari viewer and animation tools
pip install -e ".[dev]"  # regression tests and package builds
```

Both `tomocube` and `python -m tomocube` run the same command tree. The core package requires Python 3.10+; the 3D extra requires Python 3.11+ and uses napari 0.9.1+ with PyQt6.

## Inspect, measure, align, export

```bash
# Header inventory: every acquisition, channel, shape, dtype and calibration
# No volume pixels are loaded by info.
tomocube info sample.TCF --json

# Analyze every acquisition in two files, using bounded Z blocks
tomocube analyze first.TCF second.TCF --all-timepoints --channel HT --channel CH1 --output measurements.json

# Measure an explicit native-voxel region; threshold is inclusive, in RI units
tomocube analyze sample.TCF --roi 0 20 50 150 40 140 --threshold 1.36 --format csv --output roi.csv

# Estimate residual FL translation from shared image structures
# Z, Y and X bounds are in micrometers. A rejected result exits with code 1.
tomocube register sample.TCF alignment.json --fl CH1 --max-shift 8 5 5

# Inspect an accepted estimate, then reuse precisely that acquisition/channel
tomocube slice sample.TCF --fl CH1 --registration alignment.json
tomocube view3d sample.TCF --fl CH1 --registration alignment.json
tomocube tiff sample.TCF aligned.tiff --fl CH1 --registration alignment.json
tomocube mat sample.TCF aligned.mat --fl CH1 --registration alignment.json
tomocube gif sample.TCF overlay.gif --overlay --fl CH1 --registration alignment.json

# Use metadata placement directly, or export original values
tomocube tiff sample.TCF metadata-aligned.tiff --fl CH1 --registered --z-offset-mode start
tomocube tiff sample.TCF native-ri.tiff
tomocube png sample.TCF new-slice-folder --timepoint 2
```

`--timepoint N` selects a zero-based acquisition index in numeric key order (`2` before `10`). All viewers and exporters support it. `analyze` defaults to index 0; `--all-timepoints` selects the whole acquisition series. A missing FL acquisition is an explicit report row with null statistics; unknown channels and invalid selections fail.

## What the tools do

| Command | Capability |
|---|---|
| `info` | Header-only inventory, missing/orphan fluorescence keys, calibration provenance, optics and related sidecar metadata; JSON output |
| `analyze` | Batch and timepoint-series statistics for native HT/FL, strict ROI bounds, calibrated volume and threshold-selected volume; JSON/CSV |
| `register` | Bounded image-based translation, objective diagnostics, ambiguity/failure handling and a replayable source-bound JSON report |
| `view` | Orthogonal 2D navigation, channel/timepoint switching, contrast, overlays, physical distance/area tools |
| `slice` | Side-by-side HT, fluorescence and overlay planes |
| `view3d` | Native napari volume placement, calibrated clipping, camera/layer controls and animation |
| `tiff` | Native or registered scientific stacks with XYZ calibration and provenance |
| `mat` | Physical RI, native FL and optional separately named registered FL, plus calibration and transform metadata |
| `gif` / `png` | Native or registered display images and overlay animations |

Run `tomocube <command> --help` for options. Invalid arguments return 2; data/runtime failures and rejected alignments return 1; success returns 0. Machine-readable output goes to stdout; diagnostic messages go to stderr.

## Registration contract

A single `FluorescenceRegistration` maps native FL voxel centers to HT coordinates for all viewers and scientific exports. Centers lie at `index * spacing`; geometric centers use `(size - 1) / 2`. Independent XYZ spacings determine scale. Forward physical XY rotation in YX order is `[[cos, -sin], [sin, cos]]`, followed by translation in HT micrometers. The legacy `Scale` attribute is not applied a second time. Each channel retains its own `OffsetZ`.

The initial Z placement modes are:

- `start`: `OffsetZ` places the first FL voxel center in HT coordinates.
- `center`: `OffsetZ` places the FL geometric center in HT coordinates.
- `auto`: the nonnegative plane-sum weighted FL Z center is placed at the HT geometric center, with geometric fallback when there is no positive signal. **This is centering, not image matching.**

Defaults are `start` for APIs, 2D viewers and overlay exports, and `auto` for the 3D viewer. Napari now uses the same intensity centering as the other entry points. Native napari layers use the shared forward affine; 2D/export samplers interpolate linearly on the HT grid, including final and singleton planes. Samples outside native voxel centers are zero. A manual viewer Z shift changes display placement only.

### Image-based translation

`register` applies a bounded residual ZYX translation after the chosen initial metadata placement. It uses overlap-normalized positive-intensity correlation, a regular coarse search with native FL support extending across the whole shift range, and bounded refinement against original FL samples. It estimates **translation only**. It does not estimate rotation, scale, deformation or arbitrary correspondence between different structures.

HT and FL can depict different biology. An accepted score is evidence for this objective, not a probability that the biological correspondence is correct. Constant volumes, insufficient 3D extent, weak correlations, ambiguous peaks, unsupported overlap and solutions at the search boundary are rejected. Each axis must distinguish the estimate from its physical search boundaries. Periodic and invariant-direction regressions check false acceptance.

The coarse grid defaults to at most 64 samples per reference axis and bounds its search halo by adjusting spacing. Fine texture can become unresolvable on that grid; a rejected estimate may require a smaller physical search or a larger `--max-dimension` (8-128). A broad search over a small acquisition can also leave too few samples. Inspect the recorded grid spacing and diagnostics. Higher grid limits increase memory and computation. Initial overlap is required; choose an appropriate initial mode before estimating a correction.

Reports contain the candidate translation, before/after correlation, competing-peak margin, overlap, search settings, full affine, original calibration and SHA-256 fingerprints of loaded HT/FL arrays. Rejected reports are saved for inspection but cannot be applied. Existing reports require `--overwrite` to replace. Replay checks the filename, acquisition key, channel, shapes, pixels and calibration. A report for another acquisition or channel fails, including during a 2D viewer timepoint change; the existing display is retained.

Algorithm building blocks: [SciPy correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html) and [bounded optimization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html). See the regression tests for analytic fiducials and independent crop/ambiguity cases.

## Quantitative analysis and resource use

Analysis reads Z slabs from HDF5 and never populates the eager viewer cache. `--block-depth` controls slab depth. Memory still scales with the selected XY plane area; it is not an arbitrary byte-budget guarantee. Each channel uses its own native grid. The same integer ROI applied to HT and FL does **not** select the same physical region when their calibration differs.

Reports include source, acquisition, channel, native shape/dtype, XYZ spacing and its metadata/default provenance, ROI, RI scale divisor, population count/mean/std/min/max, physical sampled volume, intensity-times-volume integral and units. `--threshold` adds inclusive selected-voxel counts and physical volume. These are descriptive measurements; thresholding is not cell segmentation, and an RI integral is not dry mass.

RI scaling is decided once per full dataset, even for ROI/block reads. Integer HT uses the TCF divisor of 10000. Floating HT uses a bounded full-dataset scan to distinguish physical from scaled storage. FL retains native values. Invalid calibration and nonfinite selected intensities are rejected. Failed eager loads preserve the preceding acquisition. `load_timepoint(..., fl_channels=[])` avoids loading fluorescence; a list loads only those channels.

2D viewers cache at most one FL plane per axis. Napari retains native FL integer arrays and clips for display without copying/cropping source volumes. Overlay GIF sampling also avoids a registered 3D intermediate. Scientific TIFF/MAT export and non-overlay display registration currently materialize a single selected volume; GIF encoders may buffer frames.

## Export integrity and changes in 1.1

- TIFF API and CLI now both default to unnormalized float32. Request `bit_depth=16, normalize=True` or `--16bit --normalize` for display output. Floating output preserves values representable in float32; it does not promise exact conversion of every 64-bit integer.
- TIFF ImageJ `Info` is structured JSON. Registered TIFF uses HT spacing and records the native calibration and affine.
- MAT preserves original `fl_ch*` arrays; registered data is added as `fl_<channel>_registered`. `registration_json` and `calibration_json` preserve transforms and channel offsets.
- Single-file exports stage writes before replacing destinations and protect source files and alignment reports. PNG sequences publish only after every slice is written, into a new or empty directory; existing nonempty folders are preserved.
- The CLI uses one argparse tree. Unsupported or meaningless option combinations fail rather than being ignored.
- The obsolete standalone diagnostic, which duplicated incorrect registration math, and its unused path-configuration helper were removed. Use `info`, `register` and the shared viewers. MATLAB uses the core SciPy exporter; the unused `matlab` and duplicate `all` extras were removed in favor of the `3d` extra.
- `extract_line_profile(data, p1, p2, spacing)` accepts scalar isotropic spacing or `(y, x)` spacing. Physical points remain `(x, y)`; the old `res_xy` keyword was removed. It returns floating interpolated values and rejects out-of-range endpoints.

## Python API

```python
from tomocube import (
    TCFFileLoader, inspect_acquisition, analyze_acquisition,
    estimate_translation, save_alignment, export_to_tiff,
)

inventory = inspect_acquisition("sample.TCF")
# Python analysis defaults to every timepoint; the CLI defaults to index 0.
rows = analyze_acquisition("sample.TCF", channels=["HT", "CH1"], block_depth=8)

with TCFFileLoader("sample.TCF") as loader:
    loader.load_timepoint(0, fl_channels=["CH1"])
    estimate = estimate_translation(loader.data_3d, loader.fl_data["CH1"],
                                    loader.reg_params, channel="CH1", max_shift_um=(8, 5, 5))
    save_alignment(loader, "CH1", estimate, "alignment.json")
    if estimate.accepted:
        export_to_tiff(loader, "aligned.tiff", channel="CH1", registration_path="alignment.json")
```

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for command details and controls, and [DATA_ANALYSIS.md](DATA_ANALYSIS.md) for the general TCF format.

## Verification

```bash
python -m pytest -q
python -m build
```

Core regression tests generate synthetic acquisitions, recover known transforms, challenge ambiguous inputs, read scientific exports back, exercise CLI workflows and Matplotlib callbacks, and verify block-reading/resource behavior. Optional napari tests exercise actual Image/ViewerModel geometry and PyQt6 docks. Core CI covers Linux Python 3.10/3.14 and Windows Python 3.12, including built-wheel tests. A separate Windows job checks napari geometry and Qt controls.

Run `python examples/registration_workflow.py output/new-qa-folder` to reproduce a synthetic acquisition, alignment, calibrated report, registered TIFF and viewer image. It records known-transform errors; alignment reports also record software versions. Use a fresh output directory.

This extension has no experimental-acquisition accuracy evidence. Synthetic numerical and optional layer/model tests do not establish full interactive or GPU rendering acceptance. Napari cannot faithfully slice a rotated native volume out of its transform plane: unsupported X/Y animation sweeps are rejected, and incompatible manual slice order is restored with a notice. Use `view`, `slice` or GIF export for those resampled orthogonal planes. Crop is render-only and applies to 3D display.

MIT license.
