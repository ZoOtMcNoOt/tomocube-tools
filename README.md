# Tomocube Tools

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python library for working with Tomocube TCF (Tomocube Cell File) holotomography data.

## Features

- **View** TCF files interactively with orthogonal slice navigation
- **3D Volume Rendering** with napari for full volumetric visualization
- **Compare** HT and FL data side-by-side with overlay
- **Measure** distances and areas in physical units (micrometers)
- **Export** to TIFF, MATLAB (.mat), PNG sequence, or animated GIF
- **Register** fluorescence (FL) to holotomography (HT) coordinates
- **Auto-detect** instrument model for correct resolution defaults

## Installation

```bash
# Clone the repository
git clone https://github.com/ZoOtMcNoOt/tomocube-tools.git
cd tomocube-tools

# Install dependencies
pip install -e .

# For 3D viewing with napari
pip install -e ".[3d]"
```

### Dependencies

| Feature | Packages |
|---------|----------|
| Core | h5py, numpy, scipy, matplotlib, tifffile, pillow |
| 3D Viewer | napari, superqt |
| Video Export | imageio, imageio-ffmpeg |

---

## Quick Start

### Command Line

```bash
# View file information
python -m tomocube info sample.TCF

# Interactive 2D viewer with measurements
python -m tomocube view sample.TCF

# 3D volume viewer with napari
python -m tomocube view3d sample.TCF

# Export to TIFF stack
python -m tomocube tiff sample.TCF output.tiff

# Create animated GIF with FL overlay
python -m tomocube gif sample.TCF --overlay output.gif
```

### Python API

```python
from tomocube import TCFFile, TCFFileLoader
import h5py

# Load file metadata
with h5py.File("data.TCF", "r") as f:
    tcf = TCFFile.from_hdf5(f)
    print(f"Instrument: {tcf.device_model}")  # e.g., "HTX"
    print(f"Shape: {tcf.ht_shape}")
    print(f"Has FL: {tcf.has_fluorescence}")

# Load and process data
with TCFFileLoader("data.TCF") as loader:
    loader.load_timepoint(0)
    ht_data = loader.data_3d      # 3D numpy array (Z, Y, X)
    fl_data = loader.fl_data      # {"CH0": array, ...}
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `info` | Display file metadata (shape, resolution, instrument, FL channels) |
| `view` | Interactive 2D viewer with orthogonal slices and measurements |
| `view3d` | 3D volume renderer with napari (requires `[3d]` install) |
| `slice` | Side-by-side HT/FL comparison viewer |
| `tiff` | Export to multi-page TIFF stack |
| `mat` | Export to MATLAB .mat format |
| `gif` | Create animated GIF (Z-stack or HT+FL overlay) |

### Global Options

| Option | Description |
|--------|-------------|
| `-V, --verbose` | Show detailed registration and processing information |
| `--z-offset-mode` | FL Z alignment: `auto` (default), `start`, `center` |

### Examples

```bash
# View with verbose registration info
python -m tomocube -V view3d sample.TCF

# Export FL channel as 32-bit TIFF
python -m tomocube tiff sample.TCF fl.tiff --fl CH0 --32bit

# Create overlay GIF with custom FPS and FL alignment
python -m tomocube gif sample.TCF overlay.gif --overlay --fps 15 --z-offset-mode center
```

---

## Keyboard Shortcuts (2D Viewers)

| Key | Action |
|-----|--------|
| ↑/↓ or Scroll | Navigate Z-slices |
| Home/End | Jump to first/last slice |
| A | Auto-contrast (current slice) |
| G | Auto-contrast (global) |
| I | Invert colormap |
| F | Toggle fluorescence overlay |
| D | Distance measurement mode |
| P | Polygon/area measurement mode |
| C | Clear measurements |
| S | Save slice as PNG |
| M | Save MIP as PNG |
| 1-6 | Switch colormap |
| Q | Quit |

---

## FL-HT Registration

Fluorescence (FL) and holotomography (HT) volumes have different resolutions and require alignment.

### Resolution Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    Same Physical FOV (~230 µm)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   HT Volume                      FL Volume                      │
│   ┌──────────────────┐           ┌────────────────────────────┐ │
│   │ 1172 × 1172 px   │           │ 1893 × 1893 px             │ │
│   │ 0.196 µm/px (XY) │           │ 0.122 µm/px (XY)           │ │
│   │ 0.839 µm/slice   │           │ 1.044 µm/slice             │ │
│   │ ~74-96 Z slices  │           │ ~18-32 Z slices            │ │
│   └──────────────────┘           └────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Z-Offset Modes

The `--z-offset-mode` option controls how FL is aligned to HT in the Z-axis:

```
HT Volume (74 slices × 0.839 µm = 62 µm)
┌────────────────────────────────────────────────────────────────┐
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
└────────────────────────────────────────────────────────────────┘
0 µm                           31 µm                          62 µm

FL Volume (18 slices × 1.044 µm = 19 µm)

Mode: "auto" (default) - Centers FL signal on HT volume
┌────────────────────────────────────────────────────────────────┐
│                    ┌───────────────────┐                       │
│                    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                       │
└────────────────────────────────────────────────────────────────┘
                     ↑ FL signal center aligned to HT center

Mode: "start" - FL starts at file's OffsetZ position
┌────────────────────────────────────────────────────────────────┐
│          ┌───────────────────┐                                 │
│          │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                                 │
└────────────────────────────────────────────────────────────────┘
           ↑ OffsetZ from file

Mode: "center" - FL center at file's OffsetZ position
┌────────────────────────────────────────────────────────────────┐
│               ┌───────────────────┐                            │
│               │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                            │
└────────────────────────────────────────────────────────────────┘
                        ↑ OffsetZ (FL center)
```

### Which Mode to Use?

| Mode | When to Use |
|------|-------------|
| `auto` | **Default.** Best for most files - finds actual FL signal position |
| `start` | When file's OffsetZ correctly indicates FL start position |
| `center` | When file's OffsetZ indicates FL center position |

### Debugging Alignment

Use `--verbose` to see registration details:

```bash
python -m tomocube -V view3d sample.TCF
```

Output shows:
- FL and HT resolutions
- Z offset calculations
- Physical coverage ranges

---

## TCF File Structure

TCF files are HDF5 containers:

```
sample.TCF (HDF5)
├── Data/
│   ├── 3D/000000           # HT volume (Z, Y, X) float32
│   ├── 2DMIP/000000        # Maximum intensity projection
│   └── 3DFL/               # Fluorescence (optional)
│       └── CH0/000000      # FL channel volume
├── Info/
│   └── Device/             # Magnification, NA, RI
└── (root attrs)
    ├── DeviceModelType     # "HTX", "HT-2H", etc.
    ├── DeviceSerial        # Serial number
    └── SoftwareVersion     # Processing software version
```

### Supported Instruments

The tool auto-detects instrument model and uses appropriate defaults:

| Model | XY Res (µm) | Z Res (µm) | Notes |
|-------|-------------|------------|-------|
| HTX | 0.196 | 0.839 | Default configuration |
| HT-2H-60x | 0.196 | 0.839 | 60x objective |

---

## Python API Reference

### Core Classes

```python
from tomocube import TCFFile, TCFFileLoader, RegistrationParams

# TCFFile - Metadata container (read-only)
with h5py.File("data.TCF", "r") as f:
    tcf = TCFFile.from_hdf5(f)
    tcf.device_model        # "HTX"
    tcf.ht_shape            # (74, 1172, 1172)
    tcf.ht_resolution       # (0.839, 0.196, 0.196)
    tcf.has_fluorescence    # True
    tcf.fl_channels         # ["CH0"]
    tcf.magnification       # 40.0
    tcf.numerical_aperture  # 0.54

# TCFFileLoader - Full data access with context manager
with TCFFileLoader("data.TCF") as loader:
    loader.load_timepoint(0)
    ht = loader.data_3d       # 3D numpy array
    fl = loader.fl_data       # {"CH0": ndarray, ...}
    params = loader.reg_params  # RegistrationParams
```

### Registration

```python
from tomocube import register_fl_to_ht

# Register FL volume to HT coordinates
fl_registered = register_fl_to_ht(
    fl_data,
    ht_shape,
    params,
    channel="CH0",
    z_offset_mode="auto"  # or "start", "center"
)
# fl_registered now has same shape as HT volume
```

### Export Functions

```python
from tomocube import export_to_tiff, export_to_mat, export_to_gif

with TCFFileLoader("data.TCF") as loader:
    loader.load_timepoint(0)
    
    export_to_tiff(loader, "output.tiff", bit_depth=16)
    export_to_mat(loader, "output.mat", include_fl=True)
    export_to_gif(loader, "output.gif", axis="z", fps=10)
```

---

## Troubleshooting

### 3D Viewer Won't Start

**Error:** `ImportError: napari is required for 3D viewing`

```bash
pip install 'tomocube-tools[3d]'
```

### FL Not Aligned with HT

**Symptoms:** Fluorescence appears at top/bottom of volume, not overlapping with cell

**Solutions:**
1. Try `--z-offset-mode auto` (default)
2. Try `--z-offset-mode center` if file uses center-based offset
3. Use verbose mode to see alignment details: `python -m tomocube -V view3d file.TCF`
4. In 3D viewer, use the FL Z Offset panel to manually adjust

### Slow Performance

**Solutions:**
1. Use Volume Crop sliders to reduce displayed region
2. Start in slice mode: `--slices`
3. Close other applications to free memory

### File Won't Open

**Error:** `TCFParseError: Invalid TCF structure`

**Causes:**
- Corrupted file
- Incomplete acquisition (file partially written)
- Not a valid TCF file

**Verify with:**
```bash
# Check HDF5 structure
python -c "import h5py; h5py.File('file.TCF', 'r').visit(print)"
```

### Animation Export Fails

**Error:** `ValueError: could not broadcast input array`

**Solution:** Reset Volume Crop sliders before exporting.

### Missing FL Channels

**Check file structure:**
```bash
python -m tomocube info file.TCF
```

If FL is listed but not showing, check that the channel exists in `3DFL/CHx/` groups.

---

## Package Structure

```
src/tomocube/
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point
├── core/
│   ├── file.py          # TCFFile, TCFFileLoader
│   ├── types.py         # RegistrationParams, ViewerState
│   ├── constants.py     # Instrument defaults, HDF5 paths
│   ├── config.py        # Runtime config (verbose mode)
│   └── exceptions.py    # TCFError hierarchy
├── processing/
│   ├── registration.py  # FL-to-HT registration
│   ├── image.py         # Normalization functions
│   ├── metadata.py      # INI/JSON parsing
│   └── export.py        # TIFF, MAT, GIF export
└── viewer/
    ├── tcf_viewer.py    # 2D interactive viewer
    ├── slice_viewer.py  # Side-by-side comparison
    ├── viewer_3d.py     # 3D napari viewer
    ├── components.py    # FluorescenceMapper
    └── measurements.py  # Distance, area tools
```

---

## Documentation

- **[INSTRUCTIONS.md](INSTRUCTIONS.md)** - Complete user guide with detailed options
- **[DATA_ANALYSIS.md](DATA_ANALYSIS.md)** - TCF file format and data structure reference
- **[TODO.md](TODO.md)** - Development roadmap and technical debt

---

## License

MIT License

## Contributing

Contributions welcome! Please read the existing code style and add tests for new features.

## Acknowledgments

Developed for use with Tomocube holotomography microscopy systems.
