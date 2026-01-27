# Tomocube Tools

Python library for working with Tomocube TCF (Tomocube Cell File) holotomography data.

## Features

- **View** TCF files interactively with orthogonal slice navigation
- **Compare** HT and FL data side-by-side with overlay
- **Measure** distances and areas in physical units (micrometers)
- **Export** to TIFF, MATLAB (.mat), PNG sequence, or animated GIF
- **Register** fluorescence (FL) to holotomography (HT) coordinates
- **Process** images with normalization and analysis utilities

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/tomocube-tools.git
cd tomocube-tools

# Install dependencies
pip install h5py numpy scipy matplotlib tifffile pillow

# Install the package in development mode
pip install -e .
```

## Quick Start

### Command Line

```bash
# View a TCF file interactively
python -m tomocube view path/to/file.TCF

# Compare HT and FL slices side-by-side
python -m tomocube slice path/to/file.TCF

# Show file information
python -m tomocube info path/to/file.TCF

# Export to TIFF stack
python -m tomocube tiff path/to/file.TCF output.tiff

# Export to MATLAB .mat
python -m tomocube mat path/to/file.TCF output.mat

# Create animated GIF
python -m tomocube gif path/to/file.TCF output.gif
```

### Python API

```python
from tomocube import TCFFile, TCFViewer, TCFFileLoader
import h5py

# Load file metadata
with h5py.File("data.TCF", "r") as f:
    tcf = TCFFile.from_hdf5(f)
    print(f"Shape: {tcf.ht_shape}")
    print(f"Has FL: {tcf.has_fluorescence}")

# Interactive viewer
with TCFViewer("data.TCF") as viewer:
    viewer.show()

# Load and process data
with TCFFileLoader("data.TCF") as loader:
    loader.load_timepoint(0)
    ht_data = loader.data_3d      # 3D numpy array (Z, Y, X)
    fl_data = loader.fl_data      # {"CH0": array, ...}
```

---

## CLI Commands

### `python -m tomocube view <file.TCF>`

Full-featured interactive 3D viewer for TCF files.

**Features:**
- Navigate Z-slices with slider, keyboard, or scroll wheel
- XY, XZ, YZ orthogonal views with crosshairs
- Physical scale bars and axis labels in micrometers
- Colorbar showing refractive index values
- Fluorescence overlay with intensity colorbar
- Adjustable contrast with auto and percentile options
- Multiple colormaps with invert option
- Interactive distance and area measurements
- Export slices or MIP as PNG

**Keyboard Shortcuts:**

| Key | Action |
|-----|--------|
| Up/Down or scroll | Navigate Z-slices |
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
| Escape | Cancel measurement (or quit) |
| Q | Quit |

**Measurement Tools:**
- **Distance (D)**: Click two points to measure distance in micrometers
- **Area (P)**: Click multiple points to define a polygon, double-click to finish. Shows area (um²) and perimeter (um)

---

### `python -m tomocube slice <file.TCF>`

Side-by-side comparison of HT and FL data.

**Features:**
- HT (grayscale), FL (green), and RGB overlay views
- Z-slice navigation with keyboard or slider
- Optimized for comparing registration quality

---

### `python -m tomocube info <file.TCF>`

Display TCF file metadata.

**Output includes:**
- HT shape and resolution
- Magnification and numerical aperture
- Medium refractive index
- Number of timepoints
- Fluorescence channels and shapes
- RI value range

---

### `python -m tomocube tiff <file.TCF> [output.tiff] [options]`

Export TCF data to a multi-page TIFF stack.

**Options:**
- `--fl CH0` - Export fluorescence channel instead of HT
- `--16bit` - 16-bit output (default)
- `--32bit` - 32-bit float output

**Examples:**
```bash
# Export HT as 16-bit TIFF
python -m tomocube tiff data.TCF ht_stack.tiff

# Export FL channel as 32-bit TIFF
python -m tomocube tiff data.TCF fl_stack.tiff --fl CH0 --32bit
```

**Output:**
- Multi-page TIFF with ImageJ-compatible metadata
- Resolution and spacing encoded for proper scaling
- LZW compression for smaller file size

---

### `python -m tomocube mat <file.TCF> [output.mat] [options]`

Export TCF data to MATLAB .mat format.

**Options:**
- `--no-fl` - Exclude fluorescence data

**Examples:**
```bash
# Export with all data
python -m tomocube mat data.TCF output.mat

# Export HT only
python -m tomocube mat data.TCF ht_only.mat --no-fl
```

**Output variables:**
- `ht_data` - 3D HT volume (Z, Y, X)
- `fl_CH0`, `fl_CH1`, ... - FL channels (if present)
- `ht_resolution` - [Z, Y, X] resolution in um
- `fl_resolution` - [Z, Y, X] resolution in um
- `magnification`, `numerical_aperture`, `medium_ri`

---

### `python -m tomocube gif <file.TCF> [output.gif] [options]`

Create animated GIF from TCF data.

**Options:**
- `--overlay` - Create HT+FL overlay animation (green FL on grayscale HT)
- `--fps N` - Frame rate (default: 10)
- `--axis z|y|x` - Slice axis for animation (default: z)

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

## Python API

### Core Classes

```python
from tomocube import TCFFile, TCFFileLoader, RegistrationParams

# TCFFile - Metadata container
with h5py.File("data.TCF", "r") as f:
    tcf = TCFFile.from_hdf5(f)
    tcf.ht_shape          # (Z, Y, X) dimensions
    tcf.ht_resolution     # (Z, Y, X) um/pixel
    tcf.has_fluorescence  # bool
    tcf.fl_channels       # ["CH0", ...]
    tcf.magnification     # 40.0
    tcf.numerical_aperture # 0.95

# TCFFileLoader - File I/O with context manager
with TCFFileLoader("data.TCF") as loader:
    loader.load_timepoint(0)
    data = loader.data_3d     # 3D numpy array
    mip = loader.data_mip     # 2D MIP
    fl = loader.fl_data       # {"CH0": array, ...}
    params = loader.reg_params  # RegistrationParams
```

### Export Functions

```python
from tomocube import (
    export_to_tiff,
    export_to_mat,
    export_to_png_sequence,
    export_to_gif,
    export_overlay_gif,
)

with TCFFileLoader("data.TCF") as loader:
    loader.load_timepoint(0)

    # Export to TIFF stack
    export_to_tiff(loader, "output.tiff", channel="ht", bit_depth=16)

    # Export FL channel
    export_to_tiff(loader, "fl.tiff", channel="CH0", bit_depth=32)

    # Export to MATLAB
    export_to_mat(loader, "output.mat", include_fl=True)

    # Export PNG sequence
    export_to_png_sequence(loader, "frames/", channel="ht")

    # Create Z-stack GIF
    export_to_gif(loader, "z_stack.gif", axis="z", fps=10)

    # Create HT+FL overlay GIF
    export_overlay_gif(loader, "overlay.gif", fl_channel="CH0", fps=12)
```

### Processing Functions

```python
from tomocube import (
    register_fl_to_ht,
    normalize_image,
    normalize_with_bounds,
    compute_overlap_score,
    extract_metadata,
    parse_ini_string,
)

# Register FL to HT coordinates
fl_registered = register_fl_to_ht(fl_data, ht_shape, params)

# Normalize image to [0, 1] using percentiles
normalized = normalize_image(image, percentile_low=1, percentile_high=99)

# Normalize with explicit bounds
normalized = normalize_with_bounds(image, vmin=1.33, vmax=1.40)

# Compute FL-HT overlap score
score = compute_overlap_score(ht_image, fl_image)

# Extract metadata from HDF5 group
metadata = extract_metadata(hdf5_group)

# Parse INI-style string
config = parse_ini_string(ini_text)
```

### External Metadata Files

```python
from tomocube import (
    load_profile_file,
    load_vessel_file,
    load_project_file,
    load_experiment_file,
)

# Load .prf profile file (INI format) - processing parameters
profile = load_profile_file("Cell.img.prf")
print(profile["DefaultParameters"])  # Default imaging settings

# Load .vessel file (JSON) - well plate geometry
vessel = load_vessel_file(".vessel")
print(vessel["vessel"]["model"])  # "Ibidi μ-Slide 8well"

# Load .tcxpro project file (JSON)
project = load_project_file("2052_lab.tcxpro")
print(project["projectTitle"])

# Load .tcxexp or .experiment file (JSON)
experiment = load_experiment_file(".experiment")
print(experiment["medium"]["mediumRI"])  # 1.337
```

### Measurement Tools

```python
from tomocube import (
    MeasurementTool,
    DistanceMeasurement,
    AreaMeasurement,
    extract_line_profile,
)

# Extract intensity profile along a line
distances, values = extract_line_profile(
    data_2d,
    p1=(10.0, 20.0),   # Start point in um
    p2=(50.0, 60.0),   # End point in um
    res_xy=0.196,      # um/pixel
)

# Programmatic measurements
dist = DistanceMeasurement(points=[(0, 0), (10, 10)])
dist.calculate()
print(f"Distance: {dist.distance_um} um")

area = AreaMeasurement(points=[(0, 0), (10, 0), (10, 10), (0, 10)])
area.calculate()
print(f"Area: {area.area_um2} um², Perimeter: {area.perimeter_um} um")
```

### Viewer Classes

```python
from tomocube import TCFViewer, SliceViewer, FluorescenceMapper

# Interactive 3D viewer
with TCFViewer("data.TCF") as viewer:
    viewer.show()

# Side-by-side comparison
viewer = SliceViewer("data.TCF")
viewer.show()

# FL coordinate mapping
mapper = FluorescenceMapper(reg_params)
fl_xy_slice = mapper.get_xy_slice(fl_3d, z_index=50, ht_shape=(96, 1172, 1172))
```

---

## Package Structure

```
src/tomocube/
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point
├── core/                # File I/O, types, constants
│   ├── file.py          # TCFFile, TCFFileLoader
│   ├── types.py         # RegistrationParams, ViewerState
│   ├── constants.py     # HDF5 paths, default resolutions
│   └── exceptions.py    # TCFError hierarchy
├── processing/          # Data processing utilities
│   ├── registration.py  # FL-to-HT registration
│   ├── image.py         # Normalization functions
│   ├── metadata.py      # INI/JSON parsing
│   └── export.py        # TIFF, MAT, GIF export
└── viewer/              # Visualization tools
    ├── tcf_viewer.py    # Main interactive viewer
    ├── slice_viewer.py  # Side-by-side comparison
    ├── components.py    # FluorescenceMapper
    └── measurements.py  # Distance, area tools
```

---

## TCF File Format

TCF files are HDF5 containers with this structure:

```
Data/
├── 3D/000000           # HT volume (Z, Y, X) float32
├── 2DMIP/000000        # Maximum intensity projection
└── 3DFL/               # Fluorescence (optional)
    └── CH0/000000      # FL channel volume

Info/
├── Device              # Magnification, NA, RI
└── MetaData/
    ├── FL/Registration # Registration parameters
    └── RawData/        # Config and experiment JSON
```

**Typical Resolutions:**

| Modality | XY Resolution | Z Resolution | Image Size |
|----------|---------------|--------------|------------|
| HT | 0.196 um/px | 0.839 um/slice | 1172 x 1172 |
| FL | 0.122 um/px | 1.044 um/slice | 1893 x 1893 |

---

## FL-HT Registration

Both modalities cover the same physical field of view (~230 um). Registration involves:

1. **XY**: Resize FL to match HT pixel dimensions
2. **Z**: Map FL slices to HT Z-coordinates using `OffsetZ` parameter

```python
from tomocube import register_fl_to_ht, TCFFileLoader

with TCFFileLoader("data.TCF") as loader:
    loader.load_timepoint(0)

    if loader.has_fluorescence:
        fl_data = loader.fl_data["CH0"]
        ht_shape = loader.data_3d.shape
        params = loader.reg_params

        # Register FL volume to HT coordinates
        fl_registered = register_fl_to_ht(fl_data, ht_shape, params)
```

---

## Exception Handling

```python
from tomocube import (
    TCFError,           # Base exception
    TCFFileError,       # File not found, permission denied
    TCFParseError,      # Invalid file structure
    TCFNoFluorescenceError,  # No FL data in file
)

try:
    with TCFFileLoader("data.TCF") as loader:
        loader.load_timepoint(0)
except TCFFileError as e:
    print(f"Could not open file: {e}")
except TCFParseError as e:
    print(f"Invalid TCF structure: {e}")
```

---

## See Also

- [DATA_ANALYSIS.md](DATA_ANALYSIS.md) - Detailed TCF file format and data structure reference

## License

MIT License
