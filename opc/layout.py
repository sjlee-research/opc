"""Layout: geometry, rasterization, GDS/PNG input-output and the standard test patterns.

Sections
  1. Geometry        Grid, polygon utilities, exact area-coverage rasterization, contours
  2. Layout I/O      read_layout (GDS / PNG), write_gds, KLayout layer properties
  3. Test patterns   Pattern, Gauge, standard_patterns (iso/dense lines, line end, L, T, contacts)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import gdstk
import numpy as np
from scipy.fft import next_fast_len

from .config import PatternConfig


# ==================================================================================================
# 1. Geometry
# ==================================================================================================
#
# Polygon and pixel-grid utilities.
#
# Conventions
# -----------
# * Coordinates are in nm. A polygon is an (N, 2) float array, counter-clockwise,
#   without the closing point repeated.
# * An image array has shape (ny, nx); row index i grows with y, so plots use
#   ``origin="lower"``. Pixel (i, j) covers
#   [x0 + j*p, x0 + (j+1)*p] x [y0 + i*p, y0 + (i+1)*p] and its sample sits at
#   the pixel center.


Polygon = np.ndarray


# --------------------------------------------------------------------------- grid
@dataclass(frozen=True)
class Grid:
    x0: float
    y0: float
    pixel: float
    nx: int
    ny: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) of the pixel area, for ``imshow(extent=...)``."""
        return (self.x0, self.x0 + self.nx * self.pixel, self.y0, self.y0 + self.ny * self.pixel)

    def x_centers(self) -> np.ndarray:
        return self.x0 + (np.arange(self.nx) + 0.5) * self.pixel

    def y_centers(self) -> np.ndarray:
        return self.y0 + (np.arange(self.ny) + 0.5) * self.pixel

    def to_index(self, pts: np.ndarray) -> np.ndarray:
        """nm (x, y) -> fractional (row, col) of pixel centers."""
        pts = np.asarray(pts, dtype=float)
        col = (pts[..., 0] - self.x0) / self.pixel - 0.5
        row = (pts[..., 1] - self.y0) / self.pixel - 0.5
        return np.stack([row, col], axis=-1)

    def to_nm(self, rc: np.ndarray) -> np.ndarray:
        """fractional (row, col) -> nm (x, y)."""
        rc = np.asarray(rc, dtype=float)
        x = self.x0 + (rc[..., 1] + 0.5) * self.pixel
        y = self.y0 + (rc[..., 0] + 0.5) * self.pixel
        return np.stack([x, y], axis=-1)

    @classmethod
    def around(cls, polygons: list[Polygon], pixel: float, margin: float, align: int = 32) -> "Grid":
        """Grid covering the polygons' bounding box plus ``margin`` on every side.

        The size is rounded up to an FFT-friendly number and the origin is
        snapped to the pixel lattice, so edges on integer multiples of the pixel
        size fall exactly on pixel boundaries.
        """
        xmin, ymin, xmax, ymax = bbox(polygons)

        def axis(lo: float, hi: float) -> tuple[float, int]:
            n = math.ceil((hi - lo + 2 * margin) / pixel)
            n = next_fast_len(align * math.ceil(n / align))
            start = math.floor(((lo + hi) / 2 - n * pixel / 2) / pixel) * pixel
            return start, n

        x0, nx = axis(xmin, xmax)
        y0, ny = axis(ymin, ymax)
        return cls(x0, y0, pixel, nx, ny)


# ----------------------------------------------------------------------- polygons
def rect(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def signed_area(poly: Polygon) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_ccw(poly: Polygon) -> Polygon:
    poly = np.asarray(poly, dtype=float)
    return poly if signed_area(poly) >= 0 else poly[::-1].copy()


def clean(poly: Polygon, tol: float = 1e-6) -> Polygon:
    """Drop repeated and collinear vertices."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) > 1 and np.allclose(poly[0], poly[-1], atol=tol):
        poly = poly[:-1]
    changed = True
    while changed and len(poly) >= 3:
        prev, nxt = np.roll(poly, 1, axis=0), np.roll(poly, -1, axis=0)
        cross = (poly[:, 0] - prev[:, 0]) * (nxt[:, 1] - poly[:, 1]) - (poly[:, 1] - prev[:, 1]) * (nxt[:, 0] - poly[:, 0])
        dup = np.linalg.norm(poly - prev, axis=1) < tol
        keep = ~(dup | (np.abs(cross) < tol))
        changed = not keep.all()
        poly = poly[keep]
    return poly


def bbox(polygons: list[Polygon]) -> tuple[float, float, float, float]:
    pts = np.concatenate([np.asarray(p) for p in polygons])
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def is_manhattan(poly: Polygon, tol: float = 1e-6) -> bool:
    d = np.roll(poly, -1, axis=0) - poly
    return bool(np.all((np.abs(d[:, 0]) < tol) | (np.abs(d[:, 1]) < tol)))


def translate(polygons: list[Polygon], dx: float, dy: float) -> list[Polygon]:
    return [np.asarray(p) + np.array([dx, dy]) for p in polygons]


def union(polygons: list[Polygon], precision: float = 1e-3) -> list[Polygon]:
    """Merge overlapping or touching polygons (gdstk/Clipper). Holes come back as keyhole polygons.

    A single gdstk "or" pass can leave some touching/overlapping pieces unmerged when many shapes meet, so the
    pass is repeated until the polygon count stops changing.
    """
    if not polygons:
        return []
    res = [gdstk.Polygon(p) for p in polygons]
    for _ in range(10):
        nxt = gdstk.boolean(res, [], "or", precision=precision)
        done = len(nxt) == len(res)
        res = nxt
        if done:
            break
    return [ensure_ccw(clean(p.points)) for p in res]


def even_odd_fill(loops: list[Polygon], simplify_nm: float = 0.0, precision: float = 1e-3) -> list[Polygon]:
    """Turn closed contour loops (outer boundaries and holes, any orientation) into filled polygons.

    ``simplify_nm`` > 0 drops contour vertices that deviate less than that from a straight line
    (Douglas-Peucker), which keeps GDS output small.
    """
    from skimage.measure import approximate_polygon

    acc: list = []
    for loop in loops:
        if simplify_nm > 0:
            loop = approximate_polygon(np.vstack([loop, loop[:1]]), simplify_nm)[:-1]
        if len(loop) >= 3:
            acc = gdstk.boolean(acc, [gdstk.Polygon(loop)], "xor", precision=precision)
    return [ensure_ccw(clean(p.points)) for p in acc]


# --------------------------------------------------------------------- rasterize
def rasterize(polygons: list[Polygon], grid: Grid, supersample: int = 8) -> np.ndarray:
    """Area-coverage rasterization: each pixel gets the fraction of its area inside the mask.

    Manhattan polygons are rasterized exactly (see ``_coverage_manhattan``);
    other polygons fall back to supersampling.
    """
    if len(polygons) > 1:
        polygons = union(polygons)
    manh = [p for p in polygons if is_manhattan(p)]
    other = [p for p in polygons if not is_manhattan(p)]
    cov = _coverage_manhattan(manh, grid) if manh else np.zeros(grid.shape)
    for p in other:
        _add_supersampled(cov, p, grid, supersample)
    return np.clip(cov, 0.0, 1.0)


def _coverage_manhattan(polygons: list[Polygon], grid: Grid, chunk: int = 4096) -> np.ndarray:
    """Exact pixel coverage of Manhattan polygons as one matrix product.

    For a CCW polygon the inside indicator is a sum over vertical edges e:
        1_inside(x, y) = sum_e s_e * H(x - x_e) * 1[y_lo,e <= y < y_hi,e],
    with s_e = +1 for downward and -1 for upward edges (winding number of a ray
    toward -x). Integrating over a pixel factorizes into a row term and a column
    term, so the coverage image is  Cy^T diag(s) Cx  with
        Cx[e, j] = clip((X_j + p - x_e) / p, 0, 1)        (column coverage right of the edge)
        Cy[e, i] = |[y_lo,e, y_hi,e] ∩ [Y_i, Y_i + p]| / p  (row overlap).
    The result is exact and piecewise linear in every edge position.
    """
    xe, ylo, yhi, sgn = [], [], [], []
    for poly in polygons:
        poly = ensure_ccw(poly)
        nxt = np.roll(poly, -1, axis=0)
        vert = (np.abs(poly[:, 0] - nxt[:, 0]) < 1e-9) & (np.abs(poly[:, 1] - nxt[:, 1]) > 1e-9)
        xe.append(poly[vert, 0])
        ylo.append(np.minimum(poly[vert, 1], nxt[vert, 1]))
        yhi.append(np.maximum(poly[vert, 1], nxt[vert, 1]))
        sgn.append(np.where(nxt[vert, 1] < poly[vert, 1], 1.0, -1.0))
    xe, ylo, yhi, sgn = (np.concatenate(a) for a in (xe, ylo, yhi, sgn))

    p = grid.pixel
    X = grid.x0 + np.arange(grid.nx) * p
    Y = grid.y0 + np.arange(grid.ny) * p
    cov = np.zeros(grid.shape)
    for k in range(0, len(xe), chunk):
        sl = slice(k, k + chunk)
        cx = np.clip((X[None, :] + p - xe[sl, None]) / p, 0.0, 1.0)
        cy = np.clip((np.minimum(yhi[sl, None], Y[None, :] + p) - np.maximum(ylo[sl, None], Y[None, :])) / p, 0.0, 1.0)
        cov += (cy * sgn[sl, None]).T @ cx
    return cov


def _add_supersampled(cov: np.ndarray, poly: Polygon, grid: Grid, ss: int) -> None:
    from skimage.draw import polygon as draw_polygon

    rc = grid.to_index(poly) + 0.5                       # pixel-edge coordinates
    r0 = max(int(np.floor(rc[:, 0].min())), 0)
    r1 = min(int(np.ceil(rc[:, 0].max())), grid.ny)
    c0 = max(int(np.floor(rc[:, 1].min())), 0)
    c1 = min(int(np.ceil(rc[:, 1].max())), grid.nx)
    if r1 <= r0 or c1 <= c0:
        return
    fine = np.zeros(((r1 - r0) * ss, (c1 - c0) * ss))
    rr, cc = draw_polygon((rc[:, 0] - r0) * ss - 0.5, (rc[:, 1] - c0) * ss - 0.5, fine.shape)
    fine[rr, cc] = 1.0
    cov[r0:r1, c0:c1] += fine.reshape(r1 - r0, ss, c1 - c0, ss).mean(axis=(1, 3))


# ---------------------------------------------------------------------- contours
def extract_contours(image: np.ndarray, level: float, grid: Grid, simplify_nm: float = 0.0) -> list[Polygon]:
    """Sub-pixel iso-contours of ``image`` at ``level``, in nm (marching squares)."""
    from skimage.measure import approximate_polygon, find_contours

    loops = []
    for rc in find_contours(image, level):
        if len(rc) < 4:
            continue
        xy = grid.to_nm(rc)
        if simplify_nm > 0:
            xy = approximate_polygon(xy, simplify_nm)
        if np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        if len(xy) >= 3:
            loops.append(xy)
    return loops


def bitmap_to_polygons(bitmap: np.ndarray, pixel: float) -> list[Polygon]:
    """Binary image (row 0 = top, as stored in image files) -> merged Manhattan polygons in nm."""
    bitmap = np.asarray(bitmap, dtype=bool)
    ny = bitmap.shape[0]
    rects: list[Polygon] = []
    prev_runs: dict[tuple[int, int], int] = {}            # run -> starting row
    for r in range(ny + 1):
        runs: set[tuple[int, int]] = set()
        if r < ny:
            row = np.concatenate([[0], bitmap[r].astype(np.int8), [0]])
            d = np.diff(row)
            runs = set(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
        for run, r_start in list(prev_runs.items()):
            if run not in runs:                            # run ended: emit rectangle rows [r_start, r)
                c0, c1 = run
                rects.append(rect(c0 * pixel, (ny - r) * pixel, c1 * pixel, (ny - r_start) * pixel))
                del prev_runs[run]
        for run in runs:
            prev_runs.setdefault(run, r)
    return union(rects) if rects else []


def raster_to_polygons(binary: np.ndarray, grid: Grid) -> list[Polygon]:
    """Binary pixel mask on ``grid`` (row 0 = bottom) -> Manhattan polygons in nm."""
    polys = bitmap_to_polygons(np.asarray(binary, dtype=bool)[::-1], grid.pixel)
    return [p + np.array([grid.x0, grid.y0]) for p in polys]


# ==================================================================================================
# 2. Layout I/O
# ==================================================================================================
#
# Layout input/output: GDS (gdstk), black-and-white PNG, KLayout layer properties.
#
# Result GDS layer map (datatype 0):
#     1  target
#     2  OPC-corrected mask
#     3  printed contour before correction
#     4  printed contour after correction


LAYER_TARGET = 1
LAYER_MASK = 2
LAYER_PRINT_BEFORE = 3
LAYER_PRINT_AFTER = 4
LAYER_LABEL = 10               # text labels (method / pattern names) in the combined result GDS

GDS_MAX_POINTS = 8190          # GDSII limit; gdstk would otherwise fracture contours at 199 points
CONTOUR_SIMPLIFY_NM = 0.05     # Douglas-Peucker tolerance for printed contours written to GDS

LAYER_STYLE = {   # layer -> (name, color, KLayout dither pattern: 1 = hollow, 2 = dotted, 5 = hatched)
    LAYER_TARGET: ("target", "#9a9a9a", 2),
    LAYER_MASK: ("opc_mask", "#4a3aa7", 5),
    LAYER_PRINT_BEFORE: ("printed_before_opc", "#eb6834", 1),
    LAYER_PRINT_AFTER: ("printed_after_opc", "#2a78d6", 1),
    LAYER_LABEL: ("labels", "#52514e", 0),
}


# --------------------------------------------------------------------- reading
def read_gds(path: str | Path, layer: int = LAYER_TARGET, datatype: int | None = None,
             cell: str | None = None) -> list[Polygon]:
    """Flattened polygons (nm) of one layer. All datatypes are read unless ``datatype`` is given."""
    lib = gdstk.read_gds(str(path), unit=1e-9)
    if cell is not None:
        cells = [c for c in lib.cells if c.name == cell]
        if not cells:
            raise KeyError(f"cell '{cell}' not in {path}; cells: {[c.name for c in lib.cells]}")
        top = cells[0]
    else:
        tops = lib.top_level()
        if len(tops) != 1:
            raise ValueError(f"{path} has {len(tops)} top cells {[c.name for c in tops]}; pass cell=")
        top = tops[0]
    polys = top.get_polygons(apply_repetitions=True, include_paths=True, depth=None)
    polys = [p for p in polys if p.layer == layer and (datatype is None or p.datatype == datatype)]
    if not polys:
        raise ValueError(f"no polygons on layer {layer} in cell '{top.name}' of {path}")
    return [ensure_ccw(np.asarray(p.points, dtype=float)) for p in polys]


def read_gds_cells(path: str | Path, layer: int = LAYER_TARGET) -> list[tuple[str, list[Polygon]]]:
    """(cell name, polygons in the cell's own coordinates) for every cell placed in the top cell, in placement order."""
    lib = gdstk.read_gds(str(path), unit=1e-9)
    out = []
    for ref in lib.top_level()[0].references:
        polys = [p for p in ref.cell.get_polygons() if p.layer == layer]
        if polys:
            out.append((ref.cell.name, union([np.asarray(p.points, dtype=float) for p in polys])))
    return out


def read_png(path: str | Path, pixel_nm: float, feature: str = "auto") -> list[Polygon]:
    """Black-and-white image -> Manhattan polygons (nm), lower-left image corner at the origin.

    feature: "dark" (black shapes on white), "light", or "auto" (the minority color is the feature).
    """
    from PIL import Image

    img = np.asarray(Image.open(path).convert("L"), dtype=float)
    dark = img < 128
    if feature == "auto":
        feature = "dark" if dark.mean() <= 0.5 else "light"
    bitmap = dark if feature == "dark" else ~dark
    polys = bitmap_to_polygons(bitmap, pixel_nm)
    if not polys:
        raise ValueError(f"no {feature} features found in {path}")
    return polys


def read_layout(path: str | Path, layer: int = LAYER_TARGET, pixel_nm: float | None = None,
                cell: str | None = None, feature: str = "auto") -> list[Polygon]:
    """GDS or image -> merged target polygons (overlapping boxes drawn by hand are united)."""
    path = Path(path)
    if path.suffix.lower() in (".gds", ".gds2", ".gdsii"):
        return union(read_gds(path, layer, cell=cell))
    if path.suffix.lower() in (".png", ".bmp", ".tif", ".tiff"):
        if pixel_nm is None:
            raise ValueError("image input needs pixel_nm (nm per image pixel)")
        return read_png(path, pixel_nm, feature)
    raise ValueError(f"unsupported layout format: {path.suffix}")


# --------------------------------------------------------------------- writing
def _cell(name: str, layers: dict[int, list[Polygon]]) -> gdstk.Cell:
    c = gdstk.Cell(name)
    for layer, polys in layers.items():
        for p in polys:
            if len(p) >= 3:
                c.add(gdstk.Polygon(np.asarray(p) / 1000.0, layer=layer, datatype=0))   # nm -> um
    return c


def write_gds(path: str | Path, layers: dict[int, list[Polygon]], cell: str = "TOP", dbu_nm: float = 0.1) -> None:
    """Single flat cell. Database unit 0.1 nm keeps sub-nm OPC moves and contours exact enough."""
    lib = gdstk.Library(unit=1e-6, precision=dbu_nm * 1e-9)
    lib.add(_cell(cell, layers))
    lib.write_gds(str(path), max_points=GDS_MAX_POINTS)


def write_gds_grid(path: str | Path, rows: list[tuple[str, list[tuple[str, dict[int, list[Polygon]]]]]],
                   gap_nm: float = 1600.0, label_nm: float = 120.0, dbu_nm: float = 0.1, label_all: bool = False) -> None:
    """Many results in one GDS: rows x columns of cells, with text labels on LAYER_LABEL.

    rows: [(row label, [(column name, {layer: polygons}), ...]), ...]. Every cell is centred on its grid
    slot; the row label is written left of the row and the column names above the first row (above every
    cell with ``label_all``). An empty row label writes no row text.
    """
    boxes = [bbox([p for ps in layers.values() for p in ps]) for _, cols in rows for _, layers in cols]
    pitch_x = max(b[2] - b[0] for b in boxes) + gap_nm
    pitch_y = max(b[3] - b[1] for b in boxes) + gap_nm
    lib = gdstk.Library(unit=1e-6, precision=dbu_nm * 1e-9)
    top = gdstk.Cell("TOP")
    um = 1e-3
    for r, (row_label, cols) in enumerate(rows):
        y = -r * pitch_y
        if row_label:
            top.add(*gdstk.text(row_label, label_nm * um, ((-0.5 * pitch_x - 800) * um, y * um), layer=LAYER_LABEL))
        for c, (col_name, layers) in enumerate(cols):
            x = c * pitch_x
            cell = _cell(f"{row_label}__{col_name}" if row_label else col_name, layers)
            xmin, ymin, xmax, ymax = bbox([p for ps in layers.values() for p in ps])
            lib.add(cell)
            top.add(gdstk.Reference(cell, ((x - (xmin + xmax) / 2) * um, (y - (ymin + ymax) / 2) * um)))
            if r == 0 or label_all:
                top.add(*gdstk.text(col_name, label_nm * um, ((x - 0.4 * pitch_x) * um, (y + 0.5 * pitch_y - 300) * um),
                                    layer=LAYER_LABEL))
    lib.add(top)
    lib.write_gds(str(path), max_points=GDS_MAX_POINTS)


def write_klayout_lyp(path: str | Path) -> None:
    """KLayout layer-properties file with readable names/colors for the result layers."""
    entries = []
    for layer, (name, color, dither) in LAYER_STYLE.items():
        entries.append(f"""  <properties>
   <frame-color>{color}</frame-color>
   <fill-color>{color}</fill-color>
   <dither-pattern>I{dither}</dither-pattern>
   <visible>true</visible>
   <width>{2 if dither == 1 else 1}</width>
   <name>{name}</name>
   <source>{layer}/0@1</source>
  </properties>""")
    Path(path).write_text('<?xml version="1.0" encoding="utf-8"?>\n<layer-properties>\n'
                          + "\n".join(entries) + "\n</layer-properties>\n")


# ==================================================================================================
# 3. Test patterns
# ==================================================================================================
#
# Standard test patterns with exactly controlled dimensions.
#
# Every pattern is centered on the origin and comes with measurement gauges
# (CD across lines, line-end and corner edge positions) defined on the target.


@dataclass
class Gauge:
    """A measurement site defined on the target.

    kind="cd":   ``point`` is the feature center, ``direction`` the measurement
                 axis, ``target`` the target width; measures printed width.
    kind="edge": ``point`` lies on the target edge, ``direction`` points
                 outward; measures signed edge displacement (EPE).
    """
    name: str
    kind: str
    point: tuple[float, float]
    direction: tuple[float, float]
    target: float = 0.0


S2 = np.sqrt(0.5)


@dataclass
class Pattern:
    name: str
    polygons: list[Polygon]
    gauges: list[Gauge] = field(default_factory=list)
    description: str = ""


def _centered(p: Pattern) -> Pattern:
    xmin, ymin, xmax, ymax = bbox(p.polygons)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    p.polygons = union(translate(p.polygons, -cx, -cy))          # merge overlaps, CCW
    for g in p.gauges:
        g.point = (g.point[0] - cx, g.point[1] - cy)
    return p


def iso_line(cd: float, length: float) -> Pattern:
    return _centered(Pattern(
        "iso_line", [rect(-cd / 2, -length / 2, cd / 2, length / 2)],
        [Gauge("cd_center", "cd", (0, 0), (1, 0), cd),
         Gauge("line_end", "edge", (0, length / 2), (0, 1))],
        f"isolated line, CD {cd:g} nm"))


def dense_lines(cd: float, pitch: float, n: int, length: float) -> Pattern:
    xs = (np.arange(n) - (n - 1) / 2) * pitch
    return _centered(Pattern(
        f"dense_p{pitch:g}", [rect(x - cd / 2, -length / 2, x + cd / 2, length / 2) for x in xs],
        [Gauge("cd_center", "cd", (0, 0), (1, 0), cd),
         Gauge("cd_outer", "cd", (xs[-1], 0), (1, 0), cd),
         Gauge("line_end", "edge", (0, length / 2), (0, 1))],
        f"{n} lines, CD {cd:g} nm, pitch {pitch:g} nm"))


def line_end_pairs(cd: float, gap: float, pitch: float, n: int, length: float) -> Pattern:
    """n columns of tip-to-tip line pairs separated by ``gap``."""
    polys = []
    for x in (np.arange(n) - (n - 1) / 2) * pitch:
        polys += [rect(x - cd / 2, gap / 2, x + cd / 2, gap / 2 + length),
                  rect(x - cd / 2, -gap / 2 - length, x + cd / 2, -gap / 2)]
    return _centered(Pattern(
        "line_end", polys,
        [Gauge("tip_upper", "edge", (0, gap / 2), (0, -1)),
         Gauge("tip_lower", "edge", (0, -gap / 2), (0, 1)),
         Gauge("cd_center", "cd", (0, gap / 2 + length / 2), (1, 0), cd)],
        f"{n} tip-to-tip pairs, CD {cd:g}, gap {gap:g}, pitch {pitch:g} nm"))


def l_shape(cd: float, arm: float) -> Pattern:
    poly = np.array([[0, 0], [arm, 0], [arm, cd], [cd, cd], [cd, arm], [0, arm]], dtype=float)
    return _centered(Pattern(
        "l_shape", [poly],
        [Gauge("outer_corner", "edge", (0, 0), (-S2, -S2)),
         Gauge("inner_corner", "edge", (cd, cd), (S2, S2)),
         Gauge("cd_h_arm", "cd", (arm / 2 + cd / 2, cd / 2), (0, 1), cd),
         Gauge("cd_v_arm", "cd", (cd / 2, arm / 2 + cd / 2), (1, 0), cd),
         Gauge("end_h_arm", "edge", (arm, cd / 2), (1, 0)),
         Gauge("end_v_arm", "edge", (cd / 2, arm), (0, 1))],
        f"L-shape, CD {cd:g} nm, arms {arm:g} nm"))


def t_shape(cd: float, bar: float, stem: float) -> Pattern:
    b, c = bar / 2, cd / 2
    poly = np.array([[-b, 0], [-c, 0], [-c, -stem], [c, -stem], [c, 0], [b, 0], [b, cd], [-b, cd]], dtype=float)
    return _centered(Pattern(
        "t_shape", [poly],
        [Gauge("inner_corner_left", "edge", (-c, 0), (-S2, -S2)),
         Gauge("inner_corner_right", "edge", (c, 0), (S2, -S2)),
         Gauge("outer_corner", "edge", (b, cd), (S2, S2)),
         Gauge("cd_bar", "cd", (b / 2, cd / 2), (0, 1), cd),
         Gauge("cd_stem", "cd", (0, -stem / 2), (1, 0), cd),
         Gauge("end_stem", "edge", (0, -stem), (0, -1)),
         Gauge("end_bar", "edge", (b, cd / 2), (1, 0))],
        f"T-shape, CD {cd:g} nm, bar {bar:g} nm, stem {stem:g} nm"))


def contact_iso(size: float) -> Pattern:
    h = size / 2
    return _centered(Pattern(
        "contact_iso", [rect(-h, -h, h, h)],
        [Gauge("cd_x", "cd", (0, 0), (1, 0), size), Gauge("cd_y", "cd", (0, 0), (0, 1), size),
         Gauge("corner", "edge", (h, h), (S2, S2))],
        f"isolated contact {size:g} nm"))


def contact_array(size: float, pitch: float, n: int = 3) -> Pattern:
    h = size / 2
    cs = (np.arange(n) - (n - 1) / 2) * pitch
    return _centered(Pattern(
        f"contact_array_p{pitch:g}", [rect(x - h, y - h, x + h, y + h) for x in cs for y in cs],
        [Gauge("cd_x_center", "cd", (0, 0), (1, 0), size), Gauge("cd_y_center", "cd", (0, 0), (0, 1), size),
         Gauge("cd_x_corner", "cd", (cs[-1], cs[-1]), (1, 0), size)],
        f"{n}x{n} contacts {size:g} nm, pitch {pitch:g} nm"))


def demo_clip(cd: float) -> Pattern:
    """A small random-logic-like Manhattan clip (stands in for a hand-drawn layout)."""
    w = cd
    polys = [
        rect(0, 0, 900, w),                                                     # horizontal rail
        rect(0, 3 * w, 380, 4 * w), rect(380 - w, 3 * w, 380, 9 * w),           # L-route
        np.array([[480, 2 * w], [900, 2 * w], [900, 3 * w], [480 + w, 3 * w],   # Z-route
                  [480 + w, 7 * w], [700, 7 * w], [700, 8 * w], [480, 8 * w]], dtype=float),
        rect(0, 5 * w, w, 12 * w),                                              # vertical line
        rect(2 * w, 10 * w, 900, 11 * w),                                       # top wire with a jog below
        rect(800, 4.5 * w, 800 + w, 9 * w),
        rect(160, 6 * w, 160 + 1.5 * w, 7.5 * w),                               # contact-like pad
    ]
    return _centered(Pattern("demo_clip", polys, [], "random-logic-like demo clip"))


def standard_patterns(pc: PatternConfig) -> list[Pattern]:
    cd, L = pc.cd_nm, pc.line_length_nm
    pats = [iso_line(cd, L)]
    pats += [dense_lines(cd, p, pc.num_lines, L) for p in pc.dense_pitches_nm]
    pats += [line_end_pairs(cd, pc.line_end_gap_nm, pc.line_end_pitch_nm, 3, L / 2),
             l_shape(cd, pc.arm_length_nm),
             t_shape(cd, 1.6 * pc.arm_length_nm, pc.arm_length_nm),
             contact_iso(pc.contact_nm),
             contact_array(pc.contact_nm, pc.contact_pitch_nm),
             demo_clip(cd)]
    return pats


def _random_routing(rng: np.random.Generator, w: float, tracks: int = 5, pitch: float = 160.0,
                    length: float = 1000.0, grid: float = 20.0) -> list[Polygon]:
    """Random metal-like clip: wire segments on horizontal tracks plus a few vertical connectors."""
    segs: list[list[tuple[float, float]]] = []
    polys = []
    for t in range(tracks):
        y, x, row = t * pitch, rng.integers(0, 4) * grid * 2, []
        while x < length - 120:
            L = float(min(rng.integers(3, 9) * 40, length - x))
            row.append((x, x + L))
            polys.append(rect(x, y - w / 2, x + L, y + w / 2))
            x += L + float(rng.integers(2, 5) * 40)
        segs.append(row)
    for t in range(tracks - 1):                                   # connectors where both tracks have metal
        used: list[float] = []
        for _ in range(3):
            xc = float(rng.integers(2, int(length / grid) - 2) * grid)
            ok = lambda row: any(a + w <= xc <= b - w for a, b in row)
            if ok(segs[t]) and ok(segs[t + 1]) and all(abs(xc - u) >= 2 * w + 60 for u in used):
                polys.append(rect(xc - w / 2, t * pitch, xc + w / 2, (t + 1) * pitch))
                used.append(xc)
    return polys


def example_library(cd: float = 60.0, seed: int = 7) -> list[Pattern]:
    """100 Manhattan example targets for batch runs (nm; features >= cd wide, spaces >= cd).

    squares, rectangles, isolated / dense lines, tip-to-tip line ends, L, T, +, U, H, Z jogs, contact arrays,
    combs, the textbook "T + line" figure and random routing clips, each swept over a few dimensions.
    """
    ex: list[tuple[str, list[Polygon]]] = []
    for s in (90, 110, 130, 150, 180, 200, 240, 300, 360, 450):
        ex.append((f"square_{s}", [rect(-s / 2, -s / 2, s / 2, s / 2)]))
    for w, h in ((60, 200), (60, 400), (80, 300), (90, 180), (100, 500), (120, 240), (150, 300), (200, 400),
                 (250, 120), (300, 150)):
        ex.append((f"rect_{w}x{h}", [rect(-w / 2, -h / 2, w / 2, h / 2)]))
    for w, L in ((60, 500), (60, 800), (70, 600), (80, 800), (90, 1000), (100, 700), (120, 900), (60, 1200)):
        ex.append((f"line_w{w}_L{L}", iso_line(w, L).polygons))
    for w, p, n in ((60, 120, 5), (60, 150, 5), (60, 180, 5), (60, 240, 5), (60, 300, 3), (70, 140, 5), (80, 160, 5),
                    (80, 240, 3), (90, 180, 4), (100, 200, 3)):
        ex.append((f"lines_w{w}_p{p}_n{n}", dense_lines(w, p, n, 700).polygons))
    for w, g, p in ((60, 60, 180), (60, 80, 180), (60, 100, 180), (60, 150, 180), (60, 80, 240), (80, 80, 200),
                    (80, 120, 240), (100, 100, 250)):
        ex.append((f"lineend_w{w}_gap{g}_p{p}", line_end_pairs(w, g, p, 3, 400).polygons))
    for w, a in ((60, 200), (60, 300), (60, 500), (70, 600), (80, 400), (90, 300), (100, 400), (120, 500)):
        ex.append((f"L_w{w}_arm{a}", l_shape(w, a).polygons))
    for w, bar, stem in ((60, 400, 200), (60, 500, 300), (60, 800, 500), (70, 900, 600), (80, 600, 400),
                         (90, 600, 300), (100, 700, 400), (120, 800, 500)):
        ex.append((f"T_w{w}_bar{bar}_stem{stem}", t_shape(w, bar, stem).polygons))
    for w, a in ((60, 300), (60, 500), (80, 400), (90, 300), (100, 400), (120, 600)):
        ex.append((f"cross_w{w}_arm{a}", [rect(-w / 2, -a / 2, w / 2, a / 2), rect(-a / 2, -w / 2, a / 2, w / 2)]))
    for w, W, H in ((60, 240, 300), (60, 300, 400), (60, 400, 500), (80, 400, 400), (90, 360, 600), (100, 500, 500)):
        ex.append((f"U_w{w}_{W}x{H}", [rect(-W / 2, 0, -W / 2 + w, H), rect(W / 2 - w, 0, W / 2, H), rect(-W / 2, 0, W / 2, w)]))
    for w, W, H in ((60, 300, 500), (60, 500, 400), (80, 400, 600), (100, 500, 600)):
        ex.append((f"H_w{w}_{W}x{H}", [rect(-W / 2, -H / 2, -W / 2 + w, H / 2), rect(W / 2 - w, -H / 2, W / 2, H / 2),
                                         rect(-W / 2, -w / 2, W / 2, w / 2)]))
    for w, L, j in ((60, 250, 120), (60, 300, 200), (60, 400, 300), (80, 400, 240), (90, 600, 400), (100, 500, 300)):
        ex.append((f"Z_w{w}_L{L}_jog{j}", [rect(-L, j / 2 - w / 2, w / 2, j / 2 + w / 2),
                                           rect(-w / 2, -j / 2 - w / 2, w / 2, j / 2 + w / 2),
                                           rect(-w / 2, -j / 2 - w / 2, L, -j / 2 + w / 2)]))
    for size, p, n in ((90, 180, 2), (90, 180, 3), (80, 160, 4), (90, 240, 3), (100, 200, 3), (120, 240, 2)):
        ex.append((f"contacts_{size}_p{p}_{n}x{n}", contact_array(size, p, n).polygons))
    for w, fl, nf, p in ((60, 300, 4, 180), (60, 400, 5, 150), (80, 300, 4, 200), (100, 500, 3, 250)):
        ex.append((f"comb_w{w}_{nf}fingers_p{p}", [rect(0, 0, (nf - 1) * p + w, w)]
                   + [rect(i * p, 0, i * p + w, w + fl) for i in range(nf)]))
    for w, stub, gap in ((60, 200, 120), (80, 250, 150), (100, 300, 200)):
        H = 800.0
        ex.append((f"Tline_w{w}_stub{stub}_gap{gap}", [rect(0, 0, w, H), rect(0, H / 2 - w / 2, w + stub, H / 2 + w / 2),
                                                       rect(w + stub + gap, 0, 2 * w + stub + gap, H)]))
    rng = np.random.default_rng(seed)
    for k in range(3):
        ex.append((f"routing_{k + 1}", _random_routing(rng, cd)))
    return [_centered(Pattern(f"ex{i + 1:03d}_{name}", polys, [], name)) for i, (name, polys) in enumerate(ex)]



# ----------------------------------------------------------------- assignment cells
# Metal1 of the INV_X1 and NAND2_X4 FinFET standard cells (teaching geometry, layout unit lambda = 3.5 nm,
# uniform line width 10.5 nm). Rectangles are (x0, y0, x1, y1) in nm in the figure's frame: origin at the top-left
# of the cell, y pointing down. They were read from the reference figure on the lambda grid (every edge on a
# multiple of 3.5 nm) and are converted to y-up coordinates below.
CELL_HEIGHT_NM = 189.0
CELL_RECTS = {
    "INV_X1": [(0, 0, 63, 10.5), (10.5, 10.5, 21, 73.5), (42, 63, 52.5, 126), (10.5, 115.5, 21, 178.5),
               (0, 178.5, 63, 189)],
    "NAND2_X4": [(0, 0, 157.5, 10.5), (10.5, 10.5, 21, 42), (73.5, 10.5, 84, 42), (136.5, 10.5, 147, 42),
                 (42, 21, 52.5, 84), (105, 21, 115.5, 84), (42, 73.5, 115.5, 84), (73.5, 84, 84, 168),
                 (10.5, 105, 21, 178.5), (42, 105, 52.5, 168), (105, 105, 115.5, 168), (136.5, 105, 147, 178.5),
                 (0, 178.5, 157.5, 189)],
}
CELL_CD_GAUGES = {   # name, (x, y) in the figure frame, cut direction ("h" across a vertical wire, "v" across a bar)
    "INV_X1": [("top bar", (42, 5.25), "v"), ("upper stem", (15.75, 42), "h"), ("right bar", (47.25, 94.5), "h"),
               ("lower stem", (15.75, 147), "h"), ("bottom bar", (42, 183.75), "v")],
    "NAND2_X4": [("top bar", (31.5, 5.25), "v"), ("stub left", (15.75, 26.25), "h"), ("stub mid", (78.75, 26.25), "h"),
                 ("stub right", (141.75, 26.25), "h"), ("U left", (47.25, 47.25), "h"), ("U right", (110.25, 47.25), "h"),
                 ("U bar", (63, 78.75), "v"), ("center stem", (78.75, 126), "h"), ("lower 1", (15.75, 136.5), "h"),
                 ("lower 2", (47.25, 136.5), "h"), ("lower 3", (110.25, 136.5), "h"), ("lower 4", (141.75, 136.5), "h"),
                 ("bottom bar", (31.5, 183.75), "v")],
}
CELL_LINE_WIDTH_NM = 10.5


def fig_to_layout(x: float, y: float) -> tuple[float, float]:
    """Figure frame (y down from the cell top) -> layout frame (y up from the cell bottom)."""
    return x, CELL_HEIGHT_NM - y


def cell_targets() -> list[Pattern]:
    """INV_X1 and NAND2_X4 Metal1 targets (y-up, cell origin at the lower-left corner) with CD gauges."""
    out = []
    for name, rects in CELL_RECTS.items():
        polys = [rect(x0, CELL_HEIGHT_NM - y1, x1, CELL_HEIGHT_NM - y0) for x0, y0, x1, y1 in rects]
        gauges = [Gauge(g, "cd", fig_to_layout(*xy), (1, 0) if d == "h" else (0, 1), CELL_LINE_WIDTH_NM)
                  for g, xy, d in CELL_CD_GAUGES[name]]
        out.append(Pattern(name, union(polys), gauges, f"{name} Metal1"))
    return out
