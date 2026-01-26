"""
Measurement Tools for TCF Viewer.

Provides interactive measurement capabilities:
    - Distance measurement (line tool)
    - Area measurement (polygon/rectangle tool)
    - Profile extraction along a line

All measurements are in physical units (micrometers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass
class Measurement:
    """Base class for measurements."""
    points: list[tuple[float, float]] = field(default_factory=list)
    label: str = ""


@dataclass
class DistanceMeasurement(Measurement):
    """Distance measurement between two points."""
    distance_um: float = 0.0

    def calculate(self) -> float:
        """Calculate distance in micrometers."""
        if len(self.points) >= 2:
            p1, p2 = self.points[0], self.points[1]
            self.distance_um = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return self.distance_um


@dataclass
class AreaMeasurement(Measurement):
    """Area measurement from polygon vertices."""
    area_um2: float = 0.0
    perimeter_um: float = 0.0

    def calculate(self) -> tuple[float, float]:
        """Calculate area and perimeter in um^2 and um."""
        if len(self.points) < 3:
            return 0.0, 0.0

        # Shoelace formula for polygon area
        n = len(self.points)
        area = 0.0
        perimeter = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i][0] * self.points[j][1]
            area -= self.points[j][0] * self.points[i][1]
            perimeter += np.sqrt(
                (self.points[j][0] - self.points[i][0])**2 +
                (self.points[j][1] - self.points[i][1])**2
            )

        self.area_um2 = abs(area) / 2.0
        self.perimeter_um = perimeter
        return self.area_um2, self.perimeter_um


class MeasurementTool:
    """
    Interactive measurement tool for matplotlib axes.

    Usage:
        tool = MeasurementTool(ax, fig)
        tool.start_distance()  # Start distance measurement mode
        # Click two points on the image
        # Tool displays the distance in micrometers

        tool.start_area()  # Start area measurement mode
        # Click multiple points to define polygon
        # Double-click or press Enter to finish
    """

    def __init__(self, ax: Axes, fig: Figure, status_callback=None):
        """
        Initialize measurement tool.

        Args:
            ax: Matplotlib axes to measure on
            fig: Figure containing the axes
            status_callback: Function to call with status updates (e.g., distance)
        """
        self.ax = ax
        self.fig = fig
        self.status_callback = status_callback

        self._mode: str | None = None  # "distance", "area", or None
        self._current_points: list[tuple[float, float]] = []
        self._temp_line = None
        self._temp_polygon = None
        self._measurement_artists: list = []
        self._measurements: list[Measurement] = []

        self._cid_click = None
        self._cid_motion = None
        self._cid_key = None

    def start_distance(self) -> None:
        """Start distance measurement mode."""
        self._clear_temp()
        self._mode = "distance"
        self._current_points = []
        self._connect_events()
        self._update_status("Distance mode: Click two points")

    def start_area(self) -> None:
        """Start area measurement mode."""
        self._clear_temp()
        self._mode = "area"
        self._current_points = []
        self._connect_events()
        self._update_status("Area mode: Click points, double-click to finish")

    def cancel(self) -> None:
        """Cancel current measurement."""
        self._clear_temp()
        self._mode = None
        self._current_points = []
        self._disconnect_events()
        self._update_status("")

    def clear_all(self) -> None:
        """Clear all measurements."""
        self.cancel()
        for artist in self._measurement_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._measurement_artists = []
        self._measurements = []
        self.fig.canvas.draw_idle()

    def get_measurements(self) -> list[Measurement]:
        """Get list of all measurements."""
        return self._measurements.copy()

    def _connect_events(self) -> None:
        """Connect matplotlib events."""
        if self._cid_click is None:
            self._cid_click = self.fig.canvas.mpl_connect(
                "button_press_event", self._on_click
            )
        if self._cid_motion is None:
            self._cid_motion = self.fig.canvas.mpl_connect(
                "motion_notify_event", self._on_motion
            )
        if self._cid_key is None:
            self._cid_key = self.fig.canvas.mpl_connect(
                "key_press_event", self._on_key
            )

    def _disconnect_events(self) -> None:
        """Disconnect matplotlib events."""
        if self._cid_click is not None:
            self.fig.canvas.mpl_disconnect(self._cid_click)
            self._cid_click = None
        if self._cid_motion is not None:
            self.fig.canvas.mpl_disconnect(self._cid_motion)
            self._cid_motion = None
        if self._cid_key is not None:
            self.fig.canvas.mpl_disconnect(self._cid_key)
            self._cid_key = None

    def _on_click(self, event) -> None:
        """Handle mouse click."""
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        point = (event.xdata, event.ydata)

        if self._mode == "distance":
            self._current_points.append(point)
            if len(self._current_points) == 1:
                self._update_status(f"Distance: Click second point")
            elif len(self._current_points) >= 2:
                self._finish_distance()

        elif self._mode == "area":
            # Double-click to finish
            if event.dblclick and len(self._current_points) >= 3:
                self._finish_area()
            else:
                self._current_points.append(point)
                n = len(self._current_points)
                self._update_status(f"Area: {n} points (double-click to finish)")
                self._update_temp_polygon()

    def _on_motion(self, event) -> None:
        """Handle mouse motion."""
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        if self._mode == "distance" and len(self._current_points) == 1:
            self._update_temp_line(self._current_points[0], (event.xdata, event.ydata))
            # Show live distance
            p1 = self._current_points[0]
            dist = np.sqrt((event.xdata - p1[0])**2 + (event.ydata - p1[1])**2)
            self._update_status(f"Distance: {dist:.2f} um")

        elif self._mode == "area" and len(self._current_points) >= 1:
            self._update_temp_polygon(preview_point=(event.xdata, event.ydata))

    def _on_key(self, event) -> None:
        """Handle key press."""
        if event.key == "escape":
            self.cancel()
        elif event.key == "enter" and self._mode == "area":
            if len(self._current_points) >= 3:
                self._finish_area()

    def _update_temp_line(self, p1: tuple, p2: tuple) -> None:
        """Update temporary line during distance measurement."""
        if self._temp_line is not None:
            self._temp_line.remove()
        self._temp_line, = self.ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            "c-", lw=2, alpha=0.8
        )
        self.fig.canvas.draw_idle()

    def _update_temp_polygon(self, preview_point=None) -> None:
        """Update temporary polygon during area measurement."""
        if self._temp_polygon is not None:
            self._temp_polygon.remove()
            self._temp_polygon = None

        points = self._current_points.copy()
        if preview_point:
            points.append(preview_point)

        if len(points) >= 2:
            # Draw lines between points
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if len(points) >= 3:
                xs.append(xs[0])
                ys.append(ys[0])
            self._temp_polygon, = self.ax.plot(xs, ys, "c-", lw=2, alpha=0.8)
            self.fig.canvas.draw_idle()

    def _clear_temp(self) -> None:
        """Clear temporary drawing elements."""
        if self._temp_line is not None:
            try:
                self._temp_line.remove()
            except Exception:
                pass
            self._temp_line = None
        if self._temp_polygon is not None:
            try:
                self._temp_polygon.remove()
            except Exception:
                pass
            self._temp_polygon = None
        self.fig.canvas.draw_idle()

    def _finish_distance(self) -> None:
        """Finish distance measurement and draw permanent line."""
        self._clear_temp()

        p1, p2 = self._current_points[0], self._current_points[1]
        measurement = DistanceMeasurement(points=[p1, p2])
        measurement.calculate()

        # Draw permanent line
        line, = self.ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            "y-", lw=2, alpha=0.9
        )
        self._measurement_artists.append(line)

        # Draw endpoints
        for p in [p1, p2]:
            marker, = self.ax.plot(p[0], p[1], "yo", markersize=6)
            self._measurement_artists.append(marker)

        # Draw label at midpoint
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        text = self.ax.text(
            mid_x, mid_y, f"{measurement.distance_um:.1f} um",
            color="yellow", fontsize=9, fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7)
        )
        self._measurement_artists.append(text)

        self._measurements.append(measurement)
        self._update_status(f"Distance: {measurement.distance_um:.2f} um")

        # Reset for next measurement
        self._current_points = []
        self._mode = None
        self._disconnect_events()
        self.fig.canvas.draw_idle()

    def _finish_area(self) -> None:
        """Finish area measurement and draw permanent polygon."""
        self._clear_temp()

        if len(self._current_points) < 3:
            return

        measurement = AreaMeasurement(points=self._current_points.copy())
        measurement.calculate()

        # Draw permanent polygon
        xs = [p[0] for p in self._current_points] + [self._current_points[0][0]]
        ys = [p[1] for p in self._current_points] + [self._current_points[0][1]]
        line, = self.ax.plot(xs, ys, "y-", lw=2, alpha=0.9)
        self._measurement_artists.append(line)

        # Fill polygon
        from matplotlib.patches import Polygon
        poly = Polygon(
            self._current_points,
            closed=True,
            fill=True,
            facecolor="yellow",
            alpha=0.2,
            edgecolor="yellow",
            linewidth=2
        )
        self.ax.add_patch(poly)
        self._measurement_artists.append(poly)

        # Draw vertices
        for p in self._current_points:
            marker, = self.ax.plot(p[0], p[1], "yo", markersize=5)
            self._measurement_artists.append(marker)

        # Draw label at centroid
        centroid_x = np.mean([p[0] for p in self._current_points])
        centroid_y = np.mean([p[1] for p in self._current_points])
        text = self.ax.text(
            centroid_x, centroid_y,
            f"{measurement.area_um2:.1f} um²",
            color="yellow", fontsize=9, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7)
        )
        self._measurement_artists.append(text)

        self._measurements.append(measurement)
        self._update_status(
            f"Area: {measurement.area_um2:.2f} um², Perimeter: {measurement.perimeter_um:.2f} um"
        )

        # Reset
        self._current_points = []
        self._mode = None
        self._disconnect_events()
        self.fig.canvas.draw_idle()

    def _update_status(self, message: str) -> None:
        """Update status message."""
        if self.status_callback:
            self.status_callback(message)
        else:
            print(message)


def extract_line_profile(
    data: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    res_xy: float,
    num_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract intensity profile along a line.

    Args:
        data: 2D image data
        p1: Start point in physical coordinates (um)
        p2: End point in physical coordinates (um)
        res_xy: Resolution in um/pixel
        num_points: Number of sample points (default: pixel distance)

    Returns:
        (distances, values): Arrays of distance from p1 and corresponding values
    """
    from scipy.ndimage import map_coordinates

    # Convert to pixel coordinates
    x1, y1 = p1[0] / res_xy, p1[1] / res_xy
    x2, y2 = p2[0] / res_xy, p2[1] / res_xy

    # Calculate number of points
    pixel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if num_points is None:
        num_points = max(2, int(pixel_dist))

    # Sample along line
    xs = np.linspace(x1, x2, num_points)
    ys = np.linspace(y1, y2, num_points)

    # Extract values using interpolation
    values = map_coordinates(data, [ys, xs], order=1)

    # Calculate physical distances
    physical_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    distances = np.linspace(0, physical_dist, num_points)

    return distances, values
