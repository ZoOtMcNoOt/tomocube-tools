# Tomocube Data Directory Analysis

This document provides a comprehensive analysis of all files and file types found in the `data/` directory.

## Overview

| Metric | Value |
| -------- | ------- |
| **Total Files** | 159 |
| **Total Directories** | 46 |
| **Total Size** | ~2.5 GB |

---

## File Type Summary

| Extension | Count | Total Size (MB) | Description |
| ----------- | ------- | ----------------- | ------------- |
| `.TCF` | 10 | 2,434.38 | Tomocube Cell File (HDF5-based 3D imaging data) |
| `.PNG` | 63 | 65.50 | Image thumbnails and previews |
| `.experiment` | 10 | 0.09 | Experiment metadata (JSON format) |
| `.prf` | 44 | 0.03 | Profile/calibration files (INI format) |
| `.dat` | 10 | 0.02 | Configuration data (INI format) |
| `.vessel` | 10 | 0.02 | Vessel/well plate configuration (JSON format) |
| `.tcxexp` | 1 | 0.01 | Tomocube Experiment file (JSON format) |
| `.parent` | 10 | 0.00 | Parent path reference (plain text) |
| `.tcxpro` | 1 | 0.00 | Tomocube Project file (JSON format) |

---

## Directory Structure

```text
data/
└── lab 2056/                           # Lab/Project folder
    ├── 2052 lab.tcxpro                 # Project definition file
    └── melanoma b16/                   # Experiment folder
        ├── melanoma b16.tcxexp         # Experiment configuration
        ├── Cell.img.prf                # Imaging profile
        ├── Cell.prc.fl.prf             # Fluorescence processing profile
        ├── Cell.prc.ht.prf             # Holotomography processing profile
        ├── Cell.psf.prf                # Point spread function profile
        ├── tcf/                         # TCF file references
        └── [Session folders]/           # Individual acquisition sessions
            ├── *.TCF                    # 3D imaging data
            ├── config.dat               # Session configuration
            ├── .experiment              # Experiment metadata
            ├── .vessel                  # Vessel configuration
            ├── .parent                  # Parent path reference
            ├── *-MIP.PNG               # Maximum intensity projection
            ├── bgImages/               # Background images (0-3.png)
            ├── profile/                # Session-specific profiles
            └── thumbnail/              # HT thumbnail images
```

---

## Detailed File Type Analysis

### 1. TCF Files (`.TCF`)

**Format:** HDF5 (Hierarchical Data Format 5)  
**Magic Bytes:** `89 48 44 46 0D 0A 1A 0A` (HDF5 signature)  
**Purpose:** Primary 3D holotomographic imaging data storage

**Characteristics:**

- Largest files in the dataset (~243 MB each)
- Contains reconstructed 3D refractive index tomograms
- Binary format requiring HDF5 libraries to read (e.g., h5py in Python)
- Named with pattern: `DDMMYY.HHMMSS.experiment.###.Group#.Well#.S###.TCF`

**Naming Convention:**

```text
260114.123923.melanoma b16.001.Group1.A1.S001.TCF
│      │      │            │   │      │  │
│      │      │            │   │      │  └── Session number
│      │      │            │   │      └── Well position
│      │      │            │   └── Group name
│      │      │            └── Acquisition number
│      │      └── Experiment name
│      └── Time (HHMMSS)
└── Date (DDMMYY)
```

**TCF Internal HDF5 Structure:**

```text
Data/
├── 2DMIP/
│   └── 000000                    # 2D MIP image (1172x1172, float32)
├── 3D/
│   ├── 000000                    # 3D HT volume (74x1172x1172, float32)
│   └── attrs: ResolutionX/Y/Z    # Pixel size in micrometers
└── 3DFL/                         # (Only in files with fluorescence)
    ├── CH0/
    │   ├── 000000                # 3D FL volume (varies, e.g. 22x1893x1893)
    │   └── attrs: OffsetZ        # Z-offset from HT origin (micrometers)
    └── attrs: ResolutionX/Y/Z    # FL pixel size in micrometers

Info/
├── MetaData/
│   ├── FL/
│   │   └── Registration/
│   │       └── attrs: Rotation, Scale, TranslationX, TranslationY
│   └── RawData/
│       ├── Config               # Raw config.dat content
│       ├── Experiment           # Raw .experiment JSON
│       └── FLProcProfile        # FL processing settings
└── Version                      # File format version
```

**Resolution Information:**

| Modality | Resolution X/Y | Resolution Z | Image Size |
| ---------- | ---------------- | -------------- | ------------ |
| HT | 0.196 µm/pixel | 0.839 µm/slice | 1172 × 1172 |
| FL | 0.122 µm/pixel | 1.044 µm/slice | 1893 × 1893 |

**FL to HT Registration:**

- Both HT and FL cover the **same physical field of view** (~230 µm)
- Registration is simply **resizing** FL to match HT pixel dimensions
- Z-axis: FL OffsetZ indicates where FL slice 0 starts in HT coordinates
- Example: FL OffsetZ = 32.1 µm means FL covers HT slices ~38-65

**Files with Fluorescence Data:**

| Session | Has 3DFL | FL Slices | HT Slice Coverage |
| --------- | ---------- | ----------- | ------------------- |
| S007 | ✓ | 28 | 60-94 (partial overlap) |
| S008 | ✓ | 22 | 38-65 (good overlap) |
| S010 | ✓ | 18 | 50-72 (good overlap) |

---

### 2. PNG Files (`.PNG`)

**Purpose:** Visual representations and thumbnails

**Types Found:**

| Type | Location | Description |
| ------ | ---------- | ------------- |
| `*-MIP.PNG` | Session root | Maximum Intensity Projection of 3D data |
| `0.png` - `3.png` | `bgImages/` | Background calibration images |
| `HT_0000.png` | `thumbnail/` | Holotomography slice thumbnails |

---

### 3. Experiment Files (`.experiment`)

**Format:** JSON  
**Purpose:** Complete experiment metadata and acquisition settings

**Key Fields:**

```json
{
    "createdDate": "YYYYMMDD",
    "experimentId": "SHA1 hash",
    "experimentTitle": "experiment name",
    "experimentSettings": [{
        "imagingScenario": { ... },
        "imaginglocations": [ ... ],
        "singleImagingConditions": [ ... ],
        "vesselIndex": 0,
        "wellGroups": [ ... ]
    }],
    "medium": {
        "mediumName": "DMEM",
        "mediumRI": 1.337
    },
    "sampleType": "Cell",
    "vesselModel": "Ibidi-slide-8well"
}
```

**Imaging Modalities Supported:**

- `HT3D` - 3D Holotomography
- `HT2D` - 2D Holotomography
- `FL3D` - 3D Fluorescence
- `FL2D` - 2D Fluorescence
- `BFGray` - Brightfield Grayscale
- `BFColor` - Brightfield Color

---

### 4. Profile Files (`.prf`)

**Format:** INI (Windows configuration file format)  
**Purpose:** Processing and calibration parameters

**Profile Types:**

| File | Purpose |
| ------ | --------- |
| `Cell.img.prf` | Imaging parameters (slices, light source, NA settings) |
| `Cell.prc.ht.prf` | Holotomography processing parameters |
| `Cell.prc.fl.prf` | Fluorescence processing parameters |
| `Cell.psf.prf` | Point spread function calibration |

**Example Structure (`Cell.img.prf`):**

```ini
[ImagingVersion]
version=0.0.2
type=Standard

[SupportedNA]
0\NA=0.38
1\NA=0.54
2\NA=0.68
3\NA=0.72
size=4

[DefaultParameters]
Step=20
Slices=60
LightSource=Blue
Phototoxicity=false
ProcessingAlgorithm=Cell
```

---

### 5. Configuration Files (`config.dat`)

**Format:** INI  
**Purpose:** Per-session acquisition and device configuration

**Sections:**

| Section | Contents |
| --------- | ---------- |
| `[JobInfo]` | Title, User ID |
| `[AcquisitionCount]` | Number of HT/FL/BF acquisitions |
| `[AcquisitionSetting]` | Z-step, wavelengths, exposure times |
| `[AcquisitionPosition]` | Well index, XYZ stage positions |
| `[AcquisitionSize]` | Field of view dimensions |
| `[ImageInfo]` | ROI size, pixel offsets |
| `[TileInfo]` | Tile grid information |
| `[DeviceInfo]` | Pixel size, magnification, NA, device serial |
| `[ExternalFLInfo]` | External fluorescence module settings |

**Key Device Parameters:**

- Pixel Size: 5.48 µm
- Magnification: 40x
- Objective NA: 0.95
- Condenser NA: 0.54
- Medium RI: 1.337
- Device: TomoXP (Serial: XP58002)

---

### 6. Vessel Files (`.vessel`)

**Format:** JSON  
**Purpose:** Well plate/vessel configuration

**Structure:**

```json
{
    "vessel": {
        "AFOffset": 200,
        "MultiDish": false,
        "NA": 0.54,
        "model": "Ibidi-slide-8well",
        "name": "Ibidi μ-Slide 8well",
        "size": [75.5, 25.5]
    },
    "well": {
        "columns": 4,
        "imagingArea": { ... }
    }
}
```

---

### 7. Project/Experiment Definition Files

#### `.tcxpro` (Project File)

**Format:** JSON  
**Purpose:** Top-level project container

```json
{
    "projectDescription": "",
    "projectTitle": "2052 lab"
}
```

#### `.tcxexp` (Experiment File)

**Format:** JSON  
**Purpose:** Experiment definition with imaging conditions (similar to `.experiment` but at experiment level)

---

### 8. Parent Reference Files (`.parent`)

**Format:** Plain text  
**Purpose:** Stores parent experiment path reference

**Example Content:**

```text
2052 lab/melanoma b16
```

---

## Data Relationships

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Project (.tcxpro)                        │
│                          "2052 lab"                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Experiment (.tcxexp)                       │
│                       "melanoma b16"                            │
│              + Shared Profiles (Cell.*.prf)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │  Session 1  │    │  Session 2  │    │  Session N  │
   │─────────────│    │─────────────│    │─────────────│
   │ .TCF        │    │ .TCF        │    │ .TCF        │
   │ config.dat  │    │ config.dat  │    │ config.dat  │
   │ .experiment │    │ .experiment │    │ .experiment │
   │ .vessel     │    │ .vessel     │    │ .vessel     │
   │ bgImages/   │    │ bgImages/   │    │ bgImages/   │
   │ thumbnail/  │    │ thumbnail/  │    │ thumbnail/  │
   │ profile/    │    │ profile/    │    │ profile/    │
   └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Sessions Summary

This dataset contains **10 acquisition sessions** from a melanoma B16 cell imaging experiment:

| Session | Well | Has TCF | Has Config |
| --------- | ------ | --------- | ------------ |
| S001 | A1 | ✓ | ✓ |
| S002 | A1 | ✓ | ✓ |
| S003 | A2 | ✓ | ✓ |
| S004 | A4 | ✓ | ✓ |
| S005 | A4 | ✓ | ✓ |
| S006 | A1 | ✓ | ✓ |
| S007 | A1 | ✓ | ✓ |
| S008 | A1 | ✓ | ✓ |
| S009 | A1 | ✓ | ✓ |
| S010 | A1 | ✓ | ✓ |

**Note:** All 10 sessions now have complete TCF files.

---

## Recommendations for Processing

### Available Tools

```bash
# Interactive 3D viewer with FL overlay
python -m tomocube view "path/to/file.TCF"

# Side-by-side HT/FL comparison
python -m tomocube slice "path/to/file.TCF"

# Show file metadata
python -m tomocube info "path/to/file.TCF"
```

### Using the Package (Python)

```python
from tomocube import TCFFile, TCFFileLoader, register_fl_to_ht
import h5py

# Load metadata
with h5py.File('path/to/file.TCF', 'r') as f:
    tcf = TCFFile.from_hdf5(f)
    print(f"Shape: {tcf.ht_shape}")
    print(f"Has FL: {tcf.has_fluorescence}")

# Load and process data
with TCFFileLoader('path/to/file.TCF') as loader:
    loader.load_timepoint(0)
    ht_data = loader.data_3d
    fl_data = loader.fl_data.get('CH0')

    if fl_data is not None:
        fl_registered = register_fl_to_ht(fl_data, ht_data.shape, loader.reg_params)
```

### Reading Config/Profile Files (Python)

```python
import configparser

config = configparser.ConfigParser()
config.read('config.dat')

# Access values
pixel_size = config.get('DeviceInfo', 'Pixel_Size_Micrometer')
```

### Reading JSON Files (Python)

```python
import json

with open('.experiment', 'r') as f:
    experiment = json.load(f)
```

---

## File Format Summary Table

| Extension | Format | Encoding | Parser |
| ----------- | -------- | ---------- | -------- |
| `.TCF` | HDF5 | Binary | h5py, HDFView |
| `.PNG` | PNG | Binary | PIL, OpenCV |
| `.experiment` | JSON | UTF-8 | json |
| `.vessel` | JSON | UTF-8 | json |
| `.tcxexp` | JSON | UTF-8 | json |
| `.tcxpro` | JSON | UTF-8 | json |
| `.prf` | INI | UTF-8 | configparser |
| `.dat` | INI | UTF-8 | configparser |
| `.parent` | Plain Text | UTF-8 | readline |
