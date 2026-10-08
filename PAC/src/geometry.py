"""Geometric feature analysis of a binary character mask."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .topology import TopologyResult


@dataclass
class GeometryResult:
    """Geometric descriptors of a character mask (normalized canvas)."""

    width: int
    height: int
    aspect_ratio: float          # width / height
    area: int
    area_ratio: float            # ink area / bounding-box area
    perimeter: float
    circularity: float           # 4*pi*A / P^2
    compactness: float           # P^2 / A
    convex_hull_area: float
    solidity: float              # A / hull area
    centroid_x: float
    centroid_y: float
    centroid_x_norm: float       # centroid position inside the bbox (0..1)
    centroid_y_norm: float
    orientation_deg: float
    eccentricity: float
    hu_moments: List[float]      # -sign(h)*log10|h|
    quadrant_density: List[float]  # ink share in TL, TR, BL, BR of the bbox
    radial_histogram: List[float]
    skeleton_length: int
    endpoints: int
    branch_points: int
    hull_points: Optional[List[List[int]]] = field(default=None, repr=False)

    def summary(self) -> Dict[str, object]:
        """JSON-serializable summary."""
        data = asdict(self)
        data.pop("hull_points", None)
        return {k: (round(v, 5) if isinstance(v, float) else
                    [round(x, 5) for x in v] if isinstance(v, list) else v)
                for k, v in data.items()}


def _eccentricity_orientation(mask: np.ndarray) -> tuple:
    """Eccentricity and orientation (deg) from second-order central moments."""
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return 0.0, 0.0
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    common = math.sqrt(max(0.0, ((mu20 - mu02) / 2.0) ** 2 + mu11 ** 2))
    l1 = (mu20 + mu02) / 2.0 + common
    l2 = (mu20 + mu02) / 2.0 - common
    ecc = math.sqrt(max(0.0, 1.0 - l2 / l1)) if l1 > 0 else 0.0
    orient = 0.5 * math.degrees(math.atan2(2 * mu11, mu20 - mu02))
    return ecc, orient


def analyze_geometry(mask: np.ndarray, topo: TopologyResult, radial_bins: int = 8) -> GeometryResult:
    """Compute geometric features of a boolean ink mask.

    Args:
        mask: bool array, True = ink.
        topo: topology result of the same mask (skeleton statistics are reused).
        radial_bins: number of bins of the radial distance histogram.
    """
    mask = mask.astype(bool)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("Empty mask: cannot compute geometry.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    width, height = int(x1 - x0), int(y1 - y0)
    area = int(mask.sum())
    m8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(m8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    points = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(points)
    hull_area = max(float(cv2.contourArea(hull)), float(area))
    circularity = 4.0 * math.pi * area / perimeter ** 2 if perimeter > 0 else 0.0
    ecc, orient = _eccentricity_orientation(mask)

    hu = cv2.HuMoments(cv2.moments(m8, binaryImage=True)).ravel()
    hu_log = [float(-math.copysign(1.0, h) * math.log10(abs(h))) if h != 0 else 0.0 for h in hu]

    cx, cy = float(xs.mean()), float(ys.mean())
    my, mx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    quads = [
        mask[y0:int(my), x0:int(mx)].sum(), mask[y0:int(my), int(mx):x1].sum(),
        mask[int(my):y1, x0:int(mx)].sum(), mask[int(my):y1, int(mx):x1].sum(),
    ]
    quad_density = [float(q) / area for q in quads]

    dist = np.hypot(xs - cx, ys - cy)
    hist, _ = np.histogram(dist / (dist.max() + 1e-9), bins=radial_bins, range=(0.0, 1.0))
    radial = (hist / hist.sum()).tolist()

    return GeometryResult(
        width=width,
        height=height,
        aspect_ratio=width / float(height),
        area=area,
        area_ratio=area / float(width * height),
        perimeter=perimeter,
        circularity=circularity,
        compactness=perimeter ** 2 / area,
        convex_hull_area=hull_area,
        solidity=area / hull_area,
        centroid_x=cx,
        centroid_y=cy,
        centroid_x_norm=(cx - x0) / max(width, 1),
        centroid_y_norm=(cy - y0) / max(height, 1),
        orientation_deg=orient,
        eccentricity=ecc,
        hu_moments=hu_log,
        quadrant_density=quad_density,
        radial_histogram=radial,
        skeleton_length=topo.skeleton_length,
        endpoints=topo.endpoints,
        branch_points=topo.branch_points,
        hull_points=hull.reshape(-1, 2).tolist(),
    )


def geometry_feature_dict(geo: GeometryResult) -> Dict[str, float]:
    """Normalized (scale-free) geometry features used for comparison."""
    return {
        "aspect_ratio": geo.aspect_ratio,
        "area_ratio": geo.area_ratio,
        "circularity": geo.circularity,
        "solidity": geo.solidity,
        "eccentricity": geo.eccentricity,
        "hu_1": geo.hu_moments[0],
        "hu_2": geo.hu_moments[1],
        "hu_3": geo.hu_moments[2],
        "hu_4": geo.hu_moments[3],
        "centroid_x_norm": geo.centroid_x_norm,
        "centroid_y_norm": geo.centroid_y_norm,
        "quadrant_tl": geo.quadrant_density[0],
        "quadrant_tr": geo.quadrant_density[1],
        "quadrant_bl": geo.quadrant_density[2],
        "quadrant_br": geo.quadrant_density[3],
        "perimeter_norm": geo.perimeter / float(max(geo.width, geo.height)),
    }
