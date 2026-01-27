"""
3D Volume Viewer for Tomocube TCF files.

Provides interactive 3D visualization using napari or PyVista.
Supports volume rendering, isosurface extraction, and fluorescence overlay.

Usage:
    python -m tomocube view3d sample.TCF
    python -m tomocube view3d sample.TCF --backend pyvista
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from tomocube.core.file import TCFFileLoader

logger = logging.getLogger(__name__)


def view_3d_napari(loader: TCFFileLoader, title: str = "TCF 3D Viewer") -> None:
    """
    Launch napari-based 3D viewer.
    
    Args:
        loader: TCFFileLoader with loaded data
        title: Window title
    """
    try:
        import napari
    except ImportError:
        raise ImportError(
            "napari is required for 3D viewing. Install with:\n"
            "  pip install 'tomocube-tools[3d]'\n"
            "  or: pip install napari[all]"
        )
    
    from tomocube.processing.registration import register_fl_to_ht
    
    # Create viewer
    viewer = napari.Viewer(title=title)
    
    # Get physical scale from metadata if available
    # TCF files typically have 0.1-0.3 µm/pixel XY and ~0.5 µm/slice Z
    scale = (0.5, 0.1, 0.1)  # Z, Y, X in µm
    
    # Add HT volume
    ht_data = loader.data_3d
    p1, p99 = np.percentile(ht_data, [1, 99])
    
    viewer.add_image(
        ht_data,
        name="Holotomography (RI)",
        colormap="gray",
        contrast_limits=(p1, p99),
        scale=scale,
        blending="translucent",
        opacity=0.8,
        rendering="mip",  # Maximum intensity projection by default
    )
    
    # Add fluorescence channels if available
    if loader.has_fluorescence:
        fl_colormaps = ["green", "magenta", "cyan", "yellow", "red", "blue"]
        
        for idx, (ch_name, fl_data) in enumerate(loader.fl_data.items()):
            # Register FL to HT coordinates
            fl_registered = register_fl_to_ht(fl_data, ht_data.shape, loader.reg_params)
            
            # Get contrast for non-zero values
            fl_nonzero = fl_registered[fl_registered > 0]
            if len(fl_nonzero) > 0:
                fl_p1, fl_p99 = np.percentile(fl_nonzero, [5, 99.5])
            else:
                fl_p1, fl_p99 = 0, 1
            
            cmap = fl_colormaps[idx % len(fl_colormaps)]
            
            viewer.add_image(
                fl_registered,
                name=f"FL {ch_name}",
                colormap=cmap,
                contrast_limits=(fl_p1, fl_p99),
                scale=scale,
                blending="additive",
                opacity=0.7,
                visible=True,
                rendering="mip",
            )
    
    # Set up initial view
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (0, -30, 45)  # Nice default angle
    viewer.camera.zoom = 0.8
    
    # Add some helpful text
    print("\n  napari 3D Viewer Controls:")
    print("  ─────────────────────────────")
    print("  Mouse drag      Rotate view")
    print("  Scroll          Zoom in/out")
    print("  Shift+drag      Pan view")
    print("  2/3             Toggle 2D/3D mode")
    print("  [/]             Previous/next layer")
    print("  V               Toggle layer visibility")
    print("  Ctrl+Shift+E    Toggle layer controls")
    print()
    
    napari.run()


def view_3d_pyvista(
    loader: TCFFileLoader,
    title: str = "TCF 3D Viewer",
    show_isosurface: bool = True,
    isosurface_value: float | None = None,
) -> None:
    """
    Launch PyVista-based 3D viewer with volume rendering.
    
    Args:
        loader: TCFFileLoader with loaded data
        title: Window title
        show_isosurface: Whether to show isosurface
        isosurface_value: RI value for isosurface (auto if None)
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "PyVista is required for 3D viewing. Install with:\n"
            "  pip install 'tomocube-tools[3d]'\n"
            "  or: pip install pyvista"
        )
    
    from tomocube.processing.registration import register_fl_to_ht
    
    # Create PyVista uniform grid from numpy array
    ht_data = loader.data_3d
    
    # Spacing in µm
    spacing = (0.1, 0.1, 0.5)  # X, Y, Z
    
    # Create ImageData grid
    grid = pv.ImageData(
        dimensions=(ht_data.shape[2] + 1, ht_data.shape[1] + 1, ht_data.shape[0] + 1),
        spacing=spacing,
    )
    grid.cell_data["RI"] = ht_data.flatten(order="F")
    
    # Create plotter
    plotter = pv.Plotter(title=title)
    plotter.set_background("#1e1e1e")
    
    # Calculate isosurface value if not provided
    if isosurface_value is None:
        # Use a value slightly above median for cell membrane visualization
        p50, p85 = np.percentile(ht_data, [50, 85])
        isosurface_value = p50 + 0.3 * (p85 - p50)
    
    if show_isosurface:
        # Extract isosurface
        contour = grid.contour([isosurface_value], scalars="RI")
        if contour.n_points > 0:
            plotter.add_mesh(
                contour,
                color="white",
                opacity=0.6,
                smooth_shading=True,
                name="HT Isosurface",
            )
    
    # Add volume rendering
    p1, p99 = np.percentile(ht_data, [1, 99])
    plotter.add_volume(
        grid,
        scalars="RI",
        cmap="gray",
        clim=(p1, p99),
        opacity="sigmoid",
        opacity_unit_distance=spacing[2] * 5,
        name="HT Volume",
        show_scalar_bar=True,
    )
    
    # Add fluorescence if available
    if loader.has_fluorescence:
        fl_colors = ["green", "magenta", "cyan"]
        
        for idx, (ch_name, fl_data) in enumerate(loader.fl_data.items()):
            fl_registered = register_fl_to_ht(fl_data, ht_data.shape, loader.reg_params)
            
            # Create FL grid
            fl_grid = pv.ImageData(
                dimensions=(fl_registered.shape[2] + 1, fl_registered.shape[1] + 1, fl_registered.shape[0] + 1),
                spacing=spacing,
            )
            fl_grid.cell_data["FL"] = fl_registered.flatten(order="F")
            
            # Get threshold for FL (only show bright regions)
            fl_nonzero = fl_registered[fl_registered > 0]
            if len(fl_nonzero) > 0:
                fl_thresh = np.percentile(fl_nonzero, 80)
                fl_max = np.percentile(fl_nonzero, 99.5)
                
                # Threshold and add
                fl_threshed = fl_grid.threshold(fl_thresh, scalars="FL")
                if fl_threshed.n_points > 0:
                    color = fl_colors[idx % len(fl_colors)]
                    plotter.add_mesh(
                        fl_threshed,
                        scalars="FL",
                        cmap=color,
                        clim=(fl_thresh, fl_max),
                        opacity=0.7,
                        name=f"FL {ch_name}",
                        show_scalar_bar=False,
                    )
    
    # Add axes and scale
    plotter.add_axes()
    plotter.add_scalar_bar(title="Refractive Index", n_labels=5)
    
    # Set initial camera position
    plotter.camera_position = "iso"
    plotter.camera.azimuth = 45
    plotter.camera.elevation = 30
    
    print("\n  PyVista 3D Viewer Controls:")
    print("  ────────────────────────────")
    print("  Left mouse      Rotate view")
    print("  Middle mouse    Pan view")
    print("  Right mouse     Zoom in/out")
    print("  R               Reset camera")
    print("  W               Wireframe mode")
    print("  S               Surface mode")
    print("  Q               Quit")
    print()
    
    plotter.show()


def view_3d(
    tcf_path: str | Path,
    backend: Literal["napari", "pyvista"] = "napari",
    **kwargs,
) -> None:
    """
    Open 3D viewer for a TCF file.
    
    Args:
        tcf_path: Path to TCF file
        backend: "napari" or "pyvista"
        **kwargs: Additional arguments passed to viewer
    """
    from tomocube.core.file import TCFFileLoader
    
    tcf_path = Path(tcf_path)
    
    with TCFFileLoader(str(tcf_path)) as loader:
        loader.load_timepoint(0)
        
        title = f"TCF 3D: {tcf_path.stem}"
        
        if backend == "napari":
            view_3d_napari(loader, title=title, **kwargs)
        elif backend == "pyvista":
            view_3d_pyvista(loader, title=title, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'napari' or 'pyvista'.")
