# Tomocube Tools command reference

Use `tomocube COMMAND --help` for the parser's complete option list. `python -m tomocube` is equivalent. See [README.md](README.md) for coordinate conventions, scientific limits, installation and changes in 1.1.

## Common selection and outputs

`--timepoint N` is a zero-based index into numerically sorted HT keys. Viewers and exporters default to 0. `--fl CH1` selects fluorescence; 2D viewers default to the first available channel, 3D displays all channels by default, and overlay GIF defaults to CH0.

Viewer placement accepts `--z-offset-mode start|center|auto` or `--registration alignment.json`. Export registration accepts `--registered` with an optional Z mode, or a saved `--registration` (which implies registration). A saved report requires explicit `--fl`; its own base mode applies. Native exports reject placement options that would have no effect.

Default output paths use the current directory and input stem. Nonzero timepoints add `_tN`; registered exports add `_registered`; TIFF/PNG and explicitly selected FL GIF outputs also identify the channel. Explicit output paths may occur before or after options. Source acquisitions and saved registrations are protected from replacement.

Single-file outputs replace previous output files only after a successful write. Alignment reports require `--overwrite`. PNG output directories must be new or empty; complete sequences are published together. Choose a new directory to keep separate acquisitions or reruns.

## Inspection and measurement

```bash
tomocube info sample.TCF
tomocube info sample.TCF --json
tomocube analyze sample.TCF --all-timepoints --channel HT --channel CH0 --output measurements.json
tomocube analyze one.TCF two.TCF --timepoint 1 --format csv --output comparison.csv
tomocube analyze sample.TCF --roi 0 10 20 80 30 90 --threshold 1.36 --block-depth 4
```

`info` reads headers and related `.experiment`, `.vessel` and `profile/*.prf` metadata. JSON includes every HT/FL combination and orphan FL keys, native shapes/dtypes, uncompressed byte sizes, instrument metadata and calibration/default provenance.

`analyze` accepts multiple explicit file paths. `--timepoint N` and `--all-timepoints` are exclusive. Repeat `--channel` to choose HT and/or FL channels; default is HT. `--format json|csv` defaults to JSON; omit `--output` to write stdout. CSV encodes vector/nested fields as JSON within cells.

`--roi Z0 Z1 Y0 Y1 X0 X1` uses strict nonnegative, half-open native voxel bounds. Coordinates must lie within every selected available volume; no clipping is silently applied. `--threshold N` means `value >= N` in RI units for HT or native fluorescence counts. Missing channel acquisitions yield rows with `status=missing` and null statistics. Unknown channel names fail. No incomplete report is published if another input fails.

Statistics are count, min/max, population mean/std, sampled volume in cubic micrometers and intensity-times-volume integral. Threshold adds selected count and volume. This does not perform cell segmentation, background subtraction or dry-mass inference. See README for block memory and RI scaling behavior.

## Image alignment

```bash
tomocube register sample.TCF alignment.json --fl CH1 --timepoint 0 --max-shift 8 5 5
tomocube slice sample.TCF --fl CH1 --registration alignment.json
```

Initial placement is `--z-offset-mode start` unless selected otherwise. `auto` centers FL signal in Z; the subsequent image estimator is a separate operation. `--max-shift Z Y X` bounds each translation component in micrometers (default 10,10,10). `--max-dimension` controls coarse sampling (8-128, default 64). Diagnostic acceptance settings are `--min-score` (default 0.6), `--min-peak-margin` (0.03), and `--min-overlap` (0.5), each in `(0,1]`.

The JSON records accepted/rejected status, reason, candidate residual, original calibration, full affine, source hashes and numerical diagnostics. A rejected estimate still writes its diagnostic report, prints a JSON summary and exits 1. It cannot be applied to exports or viewers. Existing reports require `--overwrite`.

This method assumes shared positive-contrast structures and estimates translation after metadata rotation/scale. It can reject different biological contrast, periodic structures, flat directions, insufficient initial overlap, too-small sampled extent or fine texture missed by coarse sampling. Scores are not probabilities of biological correctness. Check the images and report; no experimental-acquisition accuracy is established.

## Scientific exports

```bash
# Default TIFF: native physical RI, float32
tomocube tiff sample.TCF output.tiff
# Display normalization is explicit
tomocube tiff sample.TCF display.tiff --16bit --normalize
# Metadata-aligned FL or a saved image estimate
tomocube tiff sample.TCF fl.tiff --fl CH1 --registered --z-offset-mode center
tomocube tiff sample.TCF fl.tiff --fl CH1 --registration alignment.json
# MATLAB keeps native FL and optionally adds registered data
tomocube mat sample.TCF output.mat
tomocube mat sample.TCF ht-only.mat --no-fl
tomocube mat sample.TCF aligned.mat --fl CH1 --registration alignment.json
```

TIFF preserves independent XYZ spacing using ImageJ metadata and XY resolution tags. The `Info` field is JSON with source, acquisition, channel, value units, normalization and optional registration. Float32 is the API and CLI default; 16-bit requires normalization.

MAT writes `ht_3d`, `ht_mip`, native `fl_ch*`, `metadata`, `resolution`, and `calibration_json`. `--fl` limits the included FL channel. Registered FL is an additional `fl_<channel>_registered` variable, accompanied by `registration_json`; native arrays stay intact. Missing optical fields are omitted. `--no-fl` cannot combine with FL selection or registration.

## Display exports

```bash
tomocube gif sample.TCF animation.gif --axis y --fps 15
tomocube gif sample.TCF fluorescence.gif --fl CH1
tomocube gif sample.TCF overlay.gif --overlay --fl CH1 --registration alignment.json
tomocube png sample.TCF slices --fl CH1 --registered --prefix channel1 --cmap inferno
```

GIF accepts `--axis z|y|x` and integer `--fps` from 1 to 100; timing rounds to 10 ms intervals. `--overlay` blends HT with registered FL. PNG exports Z planes with optional `--prefix`, `--cmap`, `--vmin` and `--vmax`. Both formats contain display colors rather than scientific intensity arrays; use TIFF/MAT for quantitative work.

## Viewers

```bash
tomocube view sample.TCF --timepoint 2 --fl CH1
tomocube slice sample.TCF --fl CH1 --registration alignment.json
tomocube view3d sample.TCF --timepoint 2 --fl CH1 --render mip
```

`view` provides XY/XZ/YZ navigation. It prepares both data and registration before switching acquisitions. Failed loads preserve preceding pictures/hover/save data; absent fluorescence is marked unavailable. Saved registration files bind to one acquisition/channel; switching to an incompatible selection fails visibly. `slice` opens one selected timepoint.

| Key | Orthogonal viewer action |
|---|---|
| Arrows / Home / End | Adjust focused slider / endpoints |
| A / G / R | Slice contrast / global contrast / reset |
| I / 1-6 | Invert / choose colormap |
| F / N | Toggle fluorescence / next channel |
| D / P / C | Distance / polygon area / clear measurements |
| M | Save MIP PNG |
| Q / Escape | Quit, or cancel an active measurement |

Scroll on XY moves Z; scroll on XZ moves Y; clicks move crosshairs. Navigation sliders use physical coordinates and hide singleton axes. Manual FL Z adjustment affects display only. The slice viewer supports slider navigation, arrows, Home/End and Q/Escape.

`view3d` requires optional napari dependencies. Options are `--slices`, `--render mip|attenuated_mip|minip|average`, `--screenshot PATH`, common timepoint/FL selection and alignment options. Camera/Crop and Layers/Histogram/FL Z/Animation docks expose navigation, 3D clipping and animation. Native arrays stay unchanged. Camera keys 1-6 select orthogonal presets; 0 is isometric; R resets; F fits; +/- zoom.

Napari's native rotated volumes have an out-of-slice limitation. Unsupported X/Y animation sweeps fail before capture, and incompatible manual slicing orders are restored with a notice. Use the 2D viewers or GIF exporter for resampled orthogonal planes. Crop is render-only in 3D and does not export a cropped scientific dataset.

## Python measurements

```python
from tomocube import TCFFileLoader, analyze_acquisition, extract_line_profile

# API defaults to all timepoints (CLI defaults to index 0).
rows = analyze_acquisition("sample.TCF", channels=["HT"],
                           roi=((0, 10), (20, 80), (30, 90)), threshold=1.36)

with TCFFileLoader("sample.TCF") as loader:
    loader.load_timepoint(0, fl_channels=[])
    # Points are physical (x,y); spacing is (y,x), in micrometers.
    distances, values = extract_line_profile(
        loader.data_3d[0], (0, 0), (5, 3),
        (loader.reg_params.ht_res_y, loader.reg_params.ht_res_x),
    )
    # Streaming reads do not replace the eager cache.
    for block in loader.iter_volume_blocks(0, channel="HT", block_depth=4):
        pass
```

Profile endpoints must lie within voxel-center bounds, spacing must be positive, and sample count must be at least two. Native integer inputs interpolate to floating values. The scalar spacing form remains useful for isotropic images; the obsolete `res_xy` keyword is removed.

## Troubleshooting

- Missing/invalid metadata: absent spacing uses instrument defaults with a warning and provenance; present nonpositive or nonfinite spacing is rejected.
- Missing channel: inspect `info --json`, then select an available channel at the requested acquisition.
- Rejected alignment: inspect `reason`, scores, grid spacing and bounds. Do not edit an accepted flag to force application; choose appropriate metadata placement or obtain representative shared structures.
- Mismatched saved registration: re-estimate for the selected source/acquisition/channel. Original calibration and pixels are part of the binding.
- Existing PNG directory: choose a new or empty directory; existing frames are preserved.
- 3D import/rendering failure: install the 3D extra with a supported Qt/OpenGL environment. Offscreen platforms do not necessarily supply an OpenGL context.
