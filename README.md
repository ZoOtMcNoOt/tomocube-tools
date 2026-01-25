# Tomocube Tools

Python library for working with Tomocube TCF (Tomocube Cell File) holotomography data.

## Features

- **View** TCF files interactively with orthogonal slice navigation
- **Compare** HT and FL data side-by-side with overlay
- **Register** fluorescence (FL) to holotomography (HT) coordinates
- **Process** images with normalization and analysis utilities

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/tomocube-tools.git
cd tomocube-tools

# Install dependencies
pip install h5py numpy scipy matplotlib

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
```

### Python API

```python
from tomocube import TCFFile, TCFViewer, normalize_image
import h5py

# Load file metadata
with h5py.File("data.TCF", "r") as f:
    tcf = TCFFile.from_hdf5(f)
    print(f"Shape: {tcf.ht_shape}")
    print(f"Has FL: {tcf.has_fluorescence}")

# Interactive viewer
with TCFViewer("data.TCF") as viewer:
    viewer.show()
```

## Tools

### TCF Viewer (`python -m tomocube view`)

Full-featured interactive viewer for TCF files.

**Features:**
- Navigate Z-slices with slider, keyboard, or scroll wheel
- XY, XZ, YZ orthogonal views with crosshairs
- Adjustable contrast with auto and percentile options
- Multiple colormaps with invert option
- Fluorescence overlay with registration
- Export slices or MIP as PNG

**Keyboard Shortcuts:**

| Key | Action |
|-----|--------|
| Up/Down or scroll | Navigate Z-slices |
| Home/End | Jump to first/last slice |
| A | Auto-contrast (current slice) |
| G | Auto-contrast (global) |
| F | Toggle fluorescence overlay |
| S | Save current slice as PNG |
| M | Save MIP as PNG |
| I | Invert colormap |
| 1-6 | Switch colormap |
| Q/Escape | Quit |

### Slice Viewer (`python -m tomocube slice`)

Side-by-side comparison of HT and FL data.

**Features:**
- HT (grayscale), FL (green), and RGB overlay
- Z-slice navigation with keyboard or slider

### File Info (`python -m tomocube info`)

Display TCF file metadata including shape, resolution, and FL channels.

## Package Structure

```
tomocube/
├── core/               # File I/O, types, constants
│   ├── file.py         # TCFFile, TCFFileLoader
│   ├── types.py        # RegistrationParams, ViewerState
│   ├── constants.py    # HDF5 paths, default resolutions
│   └── exceptions.py   # TCFError hierarchy
├── processing/         # Data processing utilities
│   ├── registration.py # FL-to-HT registration
│   ├── image.py        # Normalization functions
│   └── metadata.py     # INI/JSON parsing
└── viewer/             # Visualization tools
    ├── tcf_viewer.py   # Main interactive viewer
    ├── slice_viewer.py # Side-by-side comparison
    └── components.py   # FluorescenceMapper
```

## API Reference

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
    data = loader.data_3d    # 3D numpy array
    mip = loader.data_mip    # 2D MIP
    fl = loader.fl_data      # {"CH0": array, ...}
```

### Processing Functions

```python
from tomocube import (
    register_fl_to_ht,
    normalize_image,
    normalize_with_bounds,
    compute_overlap_score,
)

# Register FL to HT coordinates
fl_registered = register_fl_to_ht(fl_data, ht_shape, params)

# Normalize image to [0, 1] using percentiles
normalized = normalize_image(image, percentile_low=1, percentile_high=99)

# Normalize with explicit bounds
normalized = normalize_with_bounds(image, vmin=1.33, vmax=1.40)

# Compute FL-HT overlap score
score = compute_overlap_score(ht_image, fl_image)
```

### Viewer Classes

```python
from tomocube import TCFViewer, SliceViewer

# Interactive 3D viewer
with TCFViewer("data.TCF") as viewer:
    viewer.show()

# Side-by-side comparison
viewer = SliceViewer("data.TCF")
viewer.show()
```

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

## FL-HT Registration

Both modalities cover the same physical field of view (~230 um). Registration involves:

1. **XY**: Resize FL to match HT pixel dimensions
2. **Z**: Map FL slices to HT Z-coordinates using `OffsetZ` parameter

```python
from tomocube import register_fl_to_ht
from tomocube.core.file import load_registration_params

# Load registration parameters from file
with h5py.File("data.TCF", "r") as f:
    params = load_registration_params(f)

# Register FL volume to HT coordinates
fl_registered = register_fl_to_ht(fl_data, ht_shape, params)
```

## See Also

- [DATA_ANALYSIS.md](DATA_ANALYSIS.md) - Detailed TCF file format and data structure reference

## License

MIT License
