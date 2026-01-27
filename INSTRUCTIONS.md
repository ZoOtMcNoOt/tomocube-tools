# Tomocube Tools - User Instructions

Complete guide to using tomocube-tools for viewing, analyzing, and exporting Tomocube TCF files.

## Table of Contents

1. [Installation](#installation)
2. [Command Line Interface](#command-line-interface)
   - [view - 2D Orthogonal Viewer](#view---2d-orthogonal-viewer)
   - [view3d - 3D Volume Viewer](#view3d---3d-volume-viewer)
   - [slice - Side-by-Side Comparison](#slice---side-by-side-comparison)
   - [info - File Information](#info---file-information)
   - [tiff - TIFF Export](#tiff---tiff-export)
   - [mat - MATLAB Export](#mat---matlab-export)
   - [gif - Animated GIF Export](#gif---animated-gif-export)
3. [3D Viewer Controls](#3d-viewer-controls)
4. [FL-HT Registration](#fl-ht-registration)
5. [Python API](#python-api)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-username/tomocube-tools.git
cd tomocube-tools

# Install in development mode
pip install -e .
```

### With 3D Viewer Support

```bash
# Install with napari for 3D visualization
pip install -e ".[3d]"

# Or install napari separately
pip install napari[all] superqt
```

### Dependencies

- **Core:** h5py, numpy, scipy, matplotlib, tifffile, pillow
- **3D Viewer:** napari, superqt (for range sliders)
- **Video Export:** imageio-ffmpeg (for MP4 export)

---

## Command Line Interface

All commands use the format:
```bash
python -m tomocube <command> <file.TCF> [options]
```

### view - 2D Orthogonal Viewer

Interactive viewer with XY, XZ, YZ slices and measurement tools.

```bash
python -m tomocube view path/to/file.TCF
```

**Keyboard Shortcuts:**

| Key | Action |
|-----|--------|
| Up/Down or Scroll | Navigate Z-slices |
| Home/End | Jump to first/last slice |
| A | Auto-contrast (current slice) |
| G | Auto-contrast (global) |
| R | Reset view |
| S | Save current slice as PNG |
| M | Save MIP as PNG |
| I | Invert colormap |
| F | Toggle fluorescence overlay |
| D | Distance measurement mode |
| P | Polygon/area measurement mode |
| C | Clear all measurements |
| 1-6 | Switch colormap |
| Q/Escape | Quit |

**Measurement Tools:**
- **Distance (D):** Click two points to measure distance in micrometers
- **Area (P):** Click multiple points, double-click to finish. Shows area and perimeter

---

### view3d - 3D Volume Viewer

Interactive 3D volume rendering using napari.

```bash
python -m tomocube view3d path/to/file.TCF [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--slices` | Start in 2D slice mode instead of 3D |
| `--render <mode>` | Rendering mode: `mip`, `attenuated_mip`, `minip`, `average` |
| `--z-offset-mode <mode>` | FL Z alignment: `auto`, `start`, `center` |
| `--screenshot <file>` | Save screenshot to file |

**Examples:**

```bash
# Basic 3D view
python -m tomocube view3d cell.TCF

# Start in slice mode
python -m tomocube view3d cell.TCF --slices

# Use attenuated MIP rendering
python -m tomocube view3d cell.TCF --render attenuated_mip

# Force FL registration to use file's offset
python -m tomocube view3d cell.TCF --z-offset-mode start

# Save a screenshot
python -m tomocube view3d cell.TCF --screenshot output.png
```

**Z Offset Modes Explained:**

| Mode | Behavior |
|------|----------|
| `auto` | Smart alignment - finds FL signal center and aligns to HT center (recommended) |
| `start` | Uses file's OffsetZ as position where FL slice 0 starts |
| `center` | Uses file's OffsetZ as position of FL volume center |

See [FL-HT Registration](#fl-ht-registration) for details.

---

### slice - Side-by-Side Comparison

Compare HT and FL data side-by-side.

```bash
python -m tomocube slice path/to/file.TCF
```

Shows three panels: HT (grayscale), FL (green), and RGB overlay.

---

### info - File Information

Display TCF file metadata.

```bash
python -m tomocube info path/to/file.TCF
```

**Output includes:**
- HT shape and resolution
- Magnification and numerical aperture
- Medium refractive index
- Number of timepoints
- Fluorescence channels and shapes
- RI value range

---

### tiff - TIFF Export

Export TCF data to multi-page TIFF stack.

```bash
python -m tomocube tiff path/to/file.TCF [output.tiff] [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--fl CH0` | Export fluorescence channel instead of HT |
| `--16bit` | 16-bit output (default) |
| `--32bit` | 32-bit float output |

**Examples:**

```bash
# Export HT as 16-bit TIFF
python -m tomocube tiff data.TCF ht_stack.tiff

# Export FL channel as 32-bit TIFF
python -m tomocube tiff data.TCF fl_stack.tiff --fl CH0 --32bit
```

---

### mat - MATLAB Export

Export TCF data to MATLAB .mat format.

```bash
python -m tomocube mat path/to/file.TCF [output.mat] [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--no-fl` | Exclude fluorescence data |

**Output variables:**
- `ht_data` - 3D HT volume (Z, Y, X)
- `fl_CH0`, `fl_CH1`, ... - FL channels (if present)
- `ht_resolution` - [Z, Y, X] resolution in um
- `fl_resolution` - [Z, Y, X] resolution in um
- `magnification`, `numerical_aperture`, `medium_ri`

---

### gif - Animated GIF Export

Create animated GIF from TCF data.

```bash
python -m tomocube gif path/to/file.TCF [output.gif] [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--overlay` | Create HT+FL overlay animation |
| `--fps N` | Frame rate (default: 10) |
| `--axis z\|y\|x` | Slice axis for animation (default: z) |

**Examples:**

```bash
# Z-stack animation
python -m tomocube gif data.TCF z_animation.gif

# Y-slice animation at 15 fps
python -m tomocube gif data.TCF y_slices.gif --axis y --fps 15

# HT+FL overlay animation
python -m tomocube gif data.TCF overlay.gif --overlay --fps 12
```

---

## 3D Viewer Controls

The 3D viewer (view3d) has several control panels:

### Camera Panel (Left)

**View Presets:**
- Top [1], Bottom [2], Front [3], Back [4], Left [5], Right [6]
- Isometric [0], Reset [R], Fit [F]

**Zoom:**
- Zoom In [+/=], Zoom Out [-]
- Fit to view [F]

**Mouse Mode:**
- **Rotate:** Left-drag to rotate view (default)
- **Pan:** Shift + left-drag to pan view
- Scroll to zoom

**Keyboard Shortcuts:**

| Key | Action |
|-----|--------|
| 1-6 | Camera presets (Top/Bottom/Front/Back/Left/Right) |
| 0 | Isometric view |
| R | Reset camera |
| F | Fit view to data |
| +/- | Zoom in/out |
| 2/3 | Toggle 2D slice / 3D volume mode |

### Volume Crop Panel (Left)

Drag the range sliders to crop the volume on each axis:
- **Z (depth):** Crop top/bottom slices
- **Y (height):** Crop front/back
- **X (width):** Crop left/right

Shows physical size in micrometers.

### Layers Panel (Right)

Control visibility and appearance of each layer:
- **RI:** Holotomography (refractive index) volume
- **CH0, CH1, ...:** Fluorescence channels

For each layer:
- Eye icon: Toggle visibility
- Opacity slider: Adjust transparency
- Contrast sliders: Adjust min/max display values

### FL Z Offset Panel (Right)

*Only appears when FL data is present.*

Adjust fluorescence Z alignment:
- **Mode dropdown:** auto, start, center, manual
- **Offset slider:** Fine-tune Z position (manual mode only)
- **Reset to Auto:** Return to automatic alignment

### Animation Panel (Right)

Export animations:

**Turntable (360°):**
- Frames: Number of rotation steps
- Speed: Slow/Normal/Fast
- Export GIF or MP4

**Slice Sweep:**
- Axis: Z/Y/X
- Export GIF or MP4

---

## FL-HT Registration

Fluorescence (FL) and holotomography (HT) images are captured at different resolutions and may need Z-axis alignment.

### The Problem

| Property | HT | FL |
|----------|----|----|
| XY Resolution | ~0.196 um/px | ~0.122 um/px |
| Z Resolution | ~0.839 um/slice | ~1.044 um/slice |
| Image Size | 1172 x 1172 | 1893 x 1893 |
| Z Slices | ~74-96 | ~28-32 |

Both cover the same physical field of view (~230 um), but the file's `OffsetZ` parameter often doesn't correctly align the volumes.

### How Auto Mode Works

The `auto` mode uses signal-based alignment:

1. **Calculate FL signal center:** Find the intensity-weighted center of mass along the Z-axis
2. **Align to HT center:** Position the FL signal center at the HT volume center

This works better than geometric centering because it finds where the actual fluorescence signal is, not just the middle of the acquisition volume.

### When to Use Each Mode

| Mode | When to Use |
|------|-------------|
| `auto` | Default - works best for most files |
| `start` | If you know the file's OffsetZ is correct and represents FL start position |
| `center` | If the file's OffsetZ represents FL center position |
| `manual` | Fine-tune alignment visually in the viewer |

### Visual Verification

1. Open the 3D viewer: `python -m tomocube view3d file.TCF`
2. Look at the FL Z Offset panel on the right
3. Try different modes and see which aligns best with the cell structure visible in HT
4. Use manual mode with the slider for fine adjustment

---

## Python API

### Quick Start

```python
from tomocube import TCFFileLoader

with TCFFileLoader("data.TCF") as loader:
    loader.load_timepoint(0)

    # Access data
    ht_data = loader.data_3d      # 3D numpy array (Z, Y, X)
    mip = loader.data_mip         # 2D maximum intensity projection
    fl_data = loader.fl_data      # {"CH0": array, ...}

    # Get metadata
    print(f"Shape: {ht_data.shape}")
    print(f"Has FL: {loader.has_fluorescence}")
    print(f"Magnification: {loader.magnification}x")
```

### FL-HT Registration

```python
from tomocube import register_fl_to_ht, TCFFileLoader

with TCFFileLoader("data.TCF") as loader:
    loader.load_timepoint(0)

    if loader.has_fluorescence:
        fl_data = loader.fl_data["CH0"]
        ht_shape = loader.data_3d.shape
        params = loader.reg_params

        # Register FL volume to HT coordinates
        fl_registered = register_fl_to_ht(
            fl_data,
            ht_shape,
            params,
            channel="CH0",
            z_offset_mode="auto"  # or "start", "center"
        )

        # fl_registered now has same shape as ht_data
```

### Export Functions

```python
from tomocube import (
    export_to_tiff,
    export_to_mat,
    export_to_gif,
)

with TCFFileLoader("data.TCF") as loader:
    loader.load_timepoint(0)

    # TIFF export
    export_to_tiff(loader, "output.tiff", channel="ht", bit_depth=16)

    # MATLAB export
    export_to_mat(loader, "output.mat", include_fl=True)

    # GIF export
    export_to_gif(loader, "z_stack.gif", axis="z", fps=10)
```

---

## Troubleshooting

### 3D Viewer Won't Start

**Error:** `ImportError: napari is required for 3D viewing`

**Solution:**
```bash
pip install 'tomocube-tools[3d]'
# or
pip install napari[all] superqt
```

### FL Data Not Aligned

**Symptoms:** Fluorescence appears at top/bottom of volume, not overlapping with cell

**Solutions:**
1. Use `--z-offset-mode auto` (default, should work for most files)
2. Try other modes: `--z-offset-mode center` or `--z-offset-mode start`
3. Use the FL Z Offset panel in the 3D viewer to manually adjust

### Slow Performance with Large Files

**Solutions:**
1. Use the Volume Crop sliders to reduce the displayed region
2. Close other applications to free memory
3. For files >500MB, consider exporting a subset first

### Animation Export Fails

**Error:** `ValueError: could not broadcast input array`

**Solution:** This usually means the clipping/crop settings are invalid. Click "Reset" in the Volume Crop panel before exporting.

### File Won't Open

**Error:** `TCFParseError: Invalid TCF structure`

**Possible causes:**
- File is corrupted
- File is not a valid Tomocube TCF file
- File was partially written (acquisition interrupted)

**Solution:** Try opening in HDF5 viewer (HDFView, h5dump) to verify structure.
