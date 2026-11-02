"""Mask correction: fragmentation, evaluation, rule-based OPC, model-based OPC and pixel ILT.

Sections
  1. Fragmentation    target edges -> segments + control sites; segment offsets -> Manhattan mask
  2. Evaluation       EPE at control sites, PV band, MRC check (evaluate_mask)
  3. Rule-based OPC   space-dependent bias table + serif + hammerhead (RuleBasedOPC)
  4. Model-based OPC  segment-wise EPE feedback (+ adaptive damping / rule init / PW-aware) (run_mbopc)
  5. Pixel ILT        sigmoid-relaxed inverse lithography, reference only (run_ilt)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product

import gdstk
import numpy as np
import torch
from scipy.optimize import brentq

from .config import Config, FragmentConfig, ILTConfig, MBOPCConfig, MRCConfig
from .layout import (Grid, Polygon, clean, ensure_ccw, even_odd_fill, extract_contours, is_manhattan, raster_to_polygons,
                     rasterize, rect, signed_area, union)
from .litho import (ImageSampler, LithoSimulator, edge_displacement, epe_stats, measure_epe, measure_gauge, periodic_cd,
                    sample_edges, signed_contour_distance)


# ==================================================================================================
# 1. Fragmentation
# ==================================================================================================
#
# Edge fragmentation of Manhattan targets and mask reconstruction from segment offsets.
#
# Every target edge is cut into segments (corner segments next to each vertex,
# interior segments of ~segment_len, a single segment for line ends). The OPC
# state is one scalar per segment: its offset d along the outward normal. The
# mask is rebuilt from the offsets as a Manhattan polygon:
#
# * two collinear neighbours with different offsets are joined by a jog at their
#   shared end point;
# * two perpendicular neighbours meet at  corner + d_i n_i + d_j n_j , which is
#   the intersection of the two moved edge lines (convex or concave alike).


NORMAL, CORNER_CONVEX, CORNER_CONCAVE, LINE_END = 0, 1, 2, 3
KIND_NAMES = {NORMAL: "normal", CORNER_CONVEX: "convex corner", CORNER_CONCAVE: "concave corner",
              LINE_END: "line end"}


@dataclass
class Fragmentation:
    polygons: list[Polygon]   # target polygons (CCW, cleaned)
    a: np.ndarray             # (S, 2) segment start, CCW order inside each polygon
    b: np.ndarray             # (S, 2) segment end
    normal: np.ndarray        # (S, 2) outward unit normal
    kind: np.ndarray          # (S,) NORMAL / CORNER_CONVEX / CORNER_CONCAVE / LINE_END
    poly: np.ndarray          # (S,) polygon index
    sites: np.ndarray         # (S, 2) EPE control sites (on the target edge)
    width: np.ndarray         # (S,) local target width (inward ray-cast)
    space: np.ndarray         # (S,) local target space (outward ray-cast)
    next_collinear: np.ndarray  # (S,) segment i and its successor lie on the same edge
    beside_line_end: np.ndarray  # (S,) corner segment adjacent to a line-end segment

    def __len__(self) -> int:
        return len(self.a)

    @property
    def length(self) -> np.ndarray:
        return np.linalg.norm(self.b - self.a, axis=1)

    def mask_polygons(self, offsets: np.ndarray) -> list[Polygon]:
        """Manhattan mask polygons for the given per-segment offsets (nm, + = outward)."""
        offsets = np.asarray(offsets, dtype=float)
        out = []
        for p in range(len(self.polygons)):
            idx = np.flatnonzero(self.poly == p)
            verts = []
            for k, i in enumerate(idx):
                j = idx[(k + 1) % len(idx)]
                pi = self.b[i] + offsets[i] * self.normal[i]
                if self.next_collinear[i]:
                    verts += [pi, self.b[i] + offsets[j] * self.normal[j]]
                else:
                    verts.append(pi + offsets[j] * self.normal[j])
            poly = clean(np.array(verts))
            if len(poly) >= 3:
                out.append(poly)
        return union(out)

    def bounds(self, mrc: MRCConfig, max_in: float, max_out: float) -> tuple[np.ndarray, np.ndarray]:
        """Offset limits so the mask keeps min width / min space even if every neighbour moves too."""
        lo = -np.clip(np.minimum(max_in, (self.width - mrc.min_width_nm) / 2), 0.0, None)
        hi = np.clip(np.minimum(max_out, (self.space - mrc.min_space_nm) / 2), 0.0, None)
        return lo, hi


# ----------------------------------------------------------------- ray casting
def _edge_table(polygons: list[Polygon]):
    v, h = [], []
    for p in polygons:
        q = np.roll(p, -1, axis=0)
        for a, b in zip(p, q):
            if abs(a[0] - b[0]) < 1e-9:
                v.append((a[0], min(a[1], b[1]), max(a[1], b[1])))
            else:
                h.append((a[1], min(a[0], b[0]), max(a[0], b[0])))
    return np.array(v).reshape(-1, 3), np.array(h).reshape(-1, 3)


def raycast(points: np.ndarray, dirs: np.ndarray, polygons: list[Polygon], max_range: float) -> np.ndarray:
    """Distance from each point along its axis-aligned direction to the first polygon edge (<= max_range)."""
    vert, hor = _edge_table(polygons)
    out = np.full(len(points), max_range)
    for axis, edges in ((0, vert), (1, hor)):           # rays along x hit vertical edges, along y horizontal
        sel = np.abs(dirs[:, axis]) > 0.5
        if not sel.any() or len(edges) == 0:
            continue
        p, s = points[sel], np.sign(dirs[sel, axis])
        along, across = p[:, axis], p[:, 1 - axis]
        dist = (edges[None, :, 0] - along[:, None]) * s[:, None]
        hit = (dist > 1e-6) & (edges[None, :, 1] <= across[:, None] + 1e-9) & (across[:, None] <= edges[None, :, 2] + 1e-9)
        d = np.where(hit, dist, np.inf).min(axis=1)
        out[sel] = np.minimum(d, max_range)
    return out


# --------------------------------------------------------------- fragmentation
def _cuts(L: float, start_corner: bool, end_corner: bool, fc: FragmentConfig) -> list[float]:
    c, m = fc.corner_len_nm, fc.min_segment_nm
    if L < 2 * m:
        return [0.0, L]
    if L < 2 * c + m:
        return [0.0, L / 2, L]
    inner = L - 2 * c
    n = max(1, int(round(inner / fc.segment_len_nm)))
    return [0.0] + list(c + inner * np.arange(n + 1) / n) + [L]


def fragment(polygons: list[Polygon], fc: FragmentConfig) -> Fragmentation:
    polys = [ensure_ccw(clean(p)) for p in polygons]
    for p in polys:
        if not is_manhattan(p):
            raise ValueError("model-based OPC fragmentation needs Manhattan targets")
    A, B, N, K, P, SITE, LE_ADJ = [], [], [], [], [], [], []
    collinear = []
    for pi, poly in enumerate(polys):
        n = len(poly)
        e = np.roll(poly, -1, axis=0) - poly                        # e[k]: v_k -> v_{k+1}
        e_prev = np.roll(e, 1, axis=0)
        convex = (e_prev[:, 0] * e[:, 1] - e_prev[:, 1] * e[:, 0]) > 0   # at vertex k
        first = len(A)
        for k in range(n):
            a, b = poly[k], poly[(k + 1) % n]
            L = float(np.hypot(*(b - a)))
            t = (b - a) / L
            nrm = np.array([t[1], -t[0]])
            c0, c1 = bool(convex[k]), bool(convex[(k + 1) % n])
            if c0 and c1 and L <= fc.line_end_max_nm:
                cuts, kinds = [0.0, L], [LINE_END]
            else:
                cuts = _cuts(L, True, True, fc)
                ns = len(cuts) - 1
                kinds = [NORMAL] * ns
                kinds[0] = CORNER_CONVEX if c0 else CORNER_CONCAVE
                if ns > 1:
                    kinds[-1] = CORNER_CONVEX if c1 else CORNER_CONCAVE
            for s in range(len(cuts) - 1):
                sa, sb = a + cuts[s] * t, a + cuts[s + 1] * t
                kind = kinds[s]
                if kind in (CORNER_CONVEX, CORNER_CONCAVE) and len(cuts) > 2:
                    corner, far = (sa, sb) if s == 0 else (sb, sa)
                    site = corner + fc.corner_site * (far - corner)
                else:
                    site = (sa + sb) / 2
                A.append(sa), B.append(sb), N.append(nrm), K.append(kind), P.append(pi), SITE.append(site)
                collinear.append(s < len(cuts) - 2)
        # corner segments touching a line-end segment (hammerhead sides)
        kk = np.array(K[first:])
        le = kk == LINE_END
        adj = (np.roll(le, 1) | np.roll(le, -1)) & (kk != LINE_END)
        LE_ADJ += list(adj)
    A, B, N, SITE = (np.array(x, dtype=float) for x in (A, B, N, SITE))

    # local width / space: min over three points of each segment
    fr = np.array([0.25, 0.5, 0.75])
    pts = (A[:, None, :] + fr[None, :, None] * (B - A)[:, None, :]).reshape(-1, 2)
    nn = np.repeat(N, 3, axis=0)
    space = raycast(pts, nn, polys, fc.max_range_nm).reshape(-1, 3).min(axis=1)
    width = raycast(pts, -nn, polys, fc.max_range_nm).reshape(-1, 3).min(axis=1)
    return Fragmentation(polys, A, B, N, np.array(K), np.array(P), SITE, width, space,
                         np.array(collinear), np.array(LE_ADJ, dtype=bool))


# ==================================================================================================
# 2. Evaluation
# ==================================================================================================
#
# Mask evaluation: EPE at control sites, PV band over the process window, MRC.


# ------------------------------------------------------------------------ EPE
class ProcessImages:
    """Latent images of one mask at several defocus values (dose only rescales the threshold)."""

    def __init__(self, sim: LithoSimulator, mask: np.ndarray, grid: Grid, defocus: list[float]):
        self.sim, self.grid = sim, grid
        self.latent: dict[float, np.ndarray] = {}
        m = sim.to_tensor(mask)
        with torch.no_grad():
            for f in sorted(set(float(x) for x in defocus)):
                a = sim.aerial(m, grid.pixel, f)
                self.latent[f] = sim.latent(a, grid.pixel).cpu().numpy().astype(float)
        self._samplers: dict[float, ImageSampler] = {}
        self._contours: dict[tuple, list] = {}

    def sampler(self, f: float) -> ImageSampler:
        if f not in self._samplers:
            self._samplers[f] = ImageSampler(self.latent[f], self.grid)
        return self._samplers[f]

    def contours(self, f: float, dose: float) -> list[Polygon]:
        key = (f, dose)
        if key not in self._contours:
            self._contours[key] = extract_contours(self.latent[f], self.sim.threshold / dose, self.grid)
        return self._contours[key]

    def site_epe(self, sites: np.ndarray, normals: np.ndarray, f: float = 0.0, dose: float = 1.0,
                 search_nm: float = 80.0) -> np.ndarray:
        """EPE at control sites; misses of the normal ray fall back to the signed contour distance."""
        level = self.sim.threshold / dose
        smp = self.sampler(f)
        epe, found = edge_displacement(smp, level, sites, normals, search_nm)
        if not found.all():
            epe[~found] = signed_contour_distance(sites[~found], self.contours(f, dose), smp, level)
        return epe


# ------------------------------------------------------------------------ MRC
def mrc_check(polygons: list[Polygon], min_width: float, min_space: float, area_tol: float = 4.0,
              eps: float = 0.05) -> dict:
    """Square-element morphological width / space check (exact for Manhattan masks).

    width violations = mask - opening(mask), space violations = closing(mask) - mask.
    Features exactly at the limit pass (the element is shrunk by ``eps``).
    """
    min_width, min_space = min_width - eps, min_space - eps
    if not polygons:
        return {"width_violations": [], "space_violations": [], "n_width": 0, "n_space": 0,
                "area_width": 0.0, "area_space": 0.0}
    P = [gdstk.Polygon(p) for p in polygons]
    er = gdstk.offset(P, -min_width / 2, join="miter", use_union=True)
    opened = gdstk.offset(er, min_width / 2, join="miter", use_union=True) if er else []
    wv = [q for q in gdstk.boolean(P, opened, "not") if q.area() > area_tol]
    di = gdstk.offset(P, min_space / 2, join="miter", use_union=True)
    closed = gdstk.offset(di, -min_space / 2, join="miter", use_union=True)
    sv = [q for q in gdstk.boolean(closed, P, "not") if q.area() > area_tol]
    return {"width_violations": [q.points for q in wv], "space_violations": [q.points for q in sv],
            "n_width": len(wv), "n_space": len(sv),
            "area_width": float(sum(q.area() for q in wv)), "area_space": float(sum(q.area() for q in sv))}


# ----------------------------------------------------------------- evaluation
@dataclass
class Evaluation:
    epe_sites: np.ndarray                    # nominal EPE at control sites
    stats: dict                              # nominal site EPE statistics
    dense_stats: dict                        # nominal EPE on 4 nm edge samples away from corners
    pvb_sites: np.ndarray                    # per-site PV band width (max - min EPE over conditions)
    pvb_mean: float
    pvb_area_width: float                    # (area(outer) - area(inner)) / target perimeter
    worst_epe_pw: float                      # max |EPE| over every site and condition
    mrc: dict
    mask_vertices: int
    contours: list[Polygon] = field(default_factory=list)       # nominal printed contour
    pv_outer: list[Polygon] = field(default_factory=list)
    pv_inner: list[Polygon] = field(default_factory=list)

    def row(self) -> dict:
        return {"epe_mean_abs": self.stats["mean_abs"], "epe_rms": self.stats["rms"], "epe_max_abs": self.stats["max_abs"],
                "dense_mean_abs": self.dense_stats["mean_abs"], "dense_max_abs": self.dense_stats["max_abs"],
                "pvb_mean": self.pvb_mean, "pvb_area_width": self.pvb_area_width, "worst_epe_pw": self.worst_epe_pw,
                "mrc_width": self.mrc["n_width"], "mrc_space": self.mrc["n_space"], "mask_vertices": self.mask_vertices}


def evaluate_mask(sim: LithoSimulator, cfg: Config, target: list[Polygon], frag: Fragmentation, grid: Grid,
                  mask_polygons: list[Polygon] | None = None, mask_raster: np.ndarray | None = None,
                  corner_exclusion_nm: float = 15.0) -> Evaluation:
    if mask_raster is None:
        mask_raster = rasterize(mask_polygons, grid)
    pc = cfg.process
    focus = sorted(set([0.0] + [float(f) for f in pc.defocus_nm]))
    imgs = ProcessImages(sim, mask_raster, grid, focus)

    nominal = imgs.site_epe(frag.sites, frag.normal)
    conds = list(product(focus, [float(d) for d in pc.doses]))
    all_epe = np.array([imgs.site_epe(frag.sites, frag.normal, f, d) for f, d in conds])
    pvb = all_epe.max(axis=0) - all_epe.min(axis=0)

    th = sim.threshold
    stack = np.array([d * imgs.latent[f] for f, d in conds])
    outer = extract_contours(stack.max(axis=0), th, grid)
    inner = extract_contours(stack.min(axis=0), th, grid)
    area = lambda loops: sum(signed_area(p) for p in even_odd_fill(loops))
    perim = sum(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum() for p in target)

    samples = sample_edges(target, 4.0)
    dense, _ = measure_epe(samples, imgs.sampler(0.0), th, imgs.contours(0.0, 1.0))
    mrc = mrc_check(mask_polygons, cfg.mrc.min_width_nm, cfg.mrc.min_space_nm) if mask_polygons is not None \
        else {"n_width": -1, "n_space": -1}
    return Evaluation(
        epe_sites=nominal, stats=epe_stats(nominal),
        dense_stats=epe_stats(dense, samples.corner_dist >= corner_exclusion_nm),
        pvb_sites=pvb, pvb_mean=float(pvb.mean()), pvb_area_width=float((area(outer) - area(inner)) / perim),
        worst_epe_pw=float(np.abs(all_epe).max()), mrc=mrc,
        mask_vertices=int(sum(len(p) for p in mask_polygons)) if mask_polygons is not None else -1,
        contours=imgs.contours(0.0, 1.0), pv_outer=outer, pv_inner=inner)


def topology(target_on: np.ndarray, printed_on: np.ndarray) -> dict:
    """Connectivity check on the pixel grid (8-connected components).

    bridges   printed shapes that join two or more separate target shapes
    opens     target shapes that print as two or more separate pieces
    missing   target shapes with no printed pixel at all
    spurious  printed shapes that touch no target shape (e.g. a printing assist feature)
    """
    from scipy import ndimage

    st = np.ones((3, 3), bool)
    lt, nt = ndimage.label(target_on, structure=st)
    lp, n_p = ndimage.label(printed_on, structure=st)
    pairs = {(a, b) for a, b in zip(lt[target_on & printed_on].ravel(), lp[target_on & printed_on].ravel())}
    per_p, per_t = {}, {}
    for t, q in pairs:
        per_p.setdefault(q, set()).add(t)
        per_t.setdefault(t, set()).add(q)
    return {"n_target": int(nt), "n_printed": int(n_p),
            "bridges": sum(1 for v in per_p.values() if len(v) > 1),
            "opens": sum(1 for v in per_t.values() if len(v) > 1),
            "missing": int(nt - len(per_t)), "spurious": int(n_p - len(per_p))}


def fidelity_metrics(sim: LithoSimulator, cfg: Config, target: list[Polygon], grid: Grid,
                     mask_polygons: list[Polygon], gauges: list | None = None, dose: float = 1.0,
                     details: bool = False) -> dict:
    """How closely the nominal print matches the target.

    epe_mean / epe_max   |EPE| along the straight edges, sampled every 1 nm, outside the corner allowance
                         (corner_len from every vertex)
    corner               mean |distance| from each target corner to the printed contour, along the corner bisector
    area_nm2 / area_pct  mismatch area: where printed != target (XOR), and relative to the target area
    iou                  |printed AND target| / |printed OR target|
    cd_err_mean/_max     |printed width - target width| over the CD gauges (if given)
    bridges/opens/...    connectivity check (see ``topology``)
    With details=True the per-gauge widths and the exposure image are returned as well (keys "cd", "exposure").
    """
    imgs = ProcessImages(sim, rasterize(mask_polygons, grid), grid, [0.0])
    smp, th = imgs.sampler(0.0), sim.threshold / dose
    contours = imgs.contours(0.0, dose)

    samples = sample_edges(target, 1.0)
    epe, _ = measure_epe(samples, smp, th, contours)
    straight = samples.corner_dist >= cfg.fragment.corner_len_nm
    e = np.abs(epe[straight] if straight.any() else epe)

    pts, dirs = [], []
    for poly in (ensure_ccw(p) for p in target):
        e_in = poly - np.roll(poly, 1, axis=0)                     # edge arriving at each vertex
        e_out = np.roll(poly, -1, axis=0) - poly                   # edge leaving it
        n = lambda d: np.stack([d[:, 1], -d[:, 0]], axis=1) / np.linalg.norm(d, axis=1, keepdims=True)
        b = n(e_in) + n(e_out)                                     # outward bisector (convex and concave alike)
        pts.append(poly)
        dirs.append(b / np.linalg.norm(b, axis=1, keepdims=True))
    corner, _ = edge_displacement(smp, th, np.concatenate(pts), np.concatenate(dirs))

    P = [gdstk.Polygon(p) for p in even_odd_fill(contours)]
    T = [gdstk.Polygon(p) for p in target]
    area = lambda polys: sum(abs(q.area()) for q in polys)
    xor, inter = area(gdstk.boolean(P, T, "xor")), area(gdstk.boolean(P, T, "and"))
    t_area, p_area = area(T), area(P)
    out = {"epe_mean": float(e.mean()), "epe_max": float(e.max()), "corner": float(np.abs(corner).mean()),
           "area_pct": float(100 * xor / t_area), "area_nm2": float(xor),
           "iou": float(inter / max(t_area + p_area - inter, 1e-9))}

    cds = []
    if gauges:
        for g in gauges:
            width, _ = measure_gauge(g, smp, th)
            cds.append((g.name, width))
        err = np.abs(np.array([w for _, w in cds]) - np.array([g.target for g in gauges]))
        out.update(cd_err_mean=float(err.mean()), cd_err_max=float(err.max()))
    out.update(topology(rasterize(target, grid) > 0.5, imgs.latent[0.0] >= th))
    if details:
        out.update(cd=cds, exposure=imgs.latent[0.0], contours=contours)
    return out


# ==================================================================================================
# 3. Rule-based OPC
# ==================================================================================================
#
# Rule-based OPC: space-dependent edge bias + corner serifs + line-end hammerheads.
#
# The rule deck is expressed as per-segment offsets on the same fragmentation as
# model-based OPC, so a rule-based mask can directly seed the model-based loop.
#
#   * bias table b(width, space): for every target width found in the layout and
#     every space in the table, the edge bias that makes an infinite 1D
#     line/space array (line = width, pitch = width + space) print on target;
#   * convex corners: corner segments pushed out by ``serif_nm`` (serif);
#   * concave corners: corner segments pulled in by ``inner_serif_nm`` (anti-serif);
#   * line ends: end segment extended by ``line_end_ext_nm`` and the adjacent side
#     segments pushed out by ``hammer_nm`` (hammerhead). The extension is
#     calibrated so an isolated line end prints on target.


@dataclass
class BiasTable:
    width: float
    spaces: np.ndarray
    bias: np.ndarray

    def __call__(self, space: np.ndarray) -> np.ndarray:
        return np.interp(space, self.spaces, self.bias)


@dataclass
class RuleDeck:
    tables: dict[float, BiasTable] = field(default_factory=dict)
    line_end_ext: dict[float, float] = field(default_factory=dict)


def build_bias_table(sim: LithoSimulator, width: float, spaces: list[float], max_bias: float) -> BiasTable:
    lo_b = -min(max_bias, (width - sim.cfg.mrc.min_width_nm) / 2)
    out = []
    for s in spaces:
        pitch = width + s
        f = lambda b: periodic_cd(sim, width + 2 * b, pitch) - width
        hi_b = min(max_bias, (s - sim.cfg.mrc.min_space_nm) / 2)
        f_lo, f_hi = f(lo_b), f(hi_b)
        if f_lo > 0:
            out.append(lo_b)
        elif f_hi < 0:
            out.append(hi_b)
        else:
            out.append(brentq(f, lo_b, hi_b, xtol=0.01))
    return BiasTable(width, np.asarray(spaces, dtype=float), np.asarray(out))


def calibrate_line_end(sim: LithoSimulator, cfg: Config, width: float, table: BiasTable) -> float:
    """Line-end extension that puts the tip of an isolated, rule-biased line on target."""
    if cfg.rule.line_end_ext_nm is not None:
        return float(cfg.rule.line_end_ext_nm)
    if width > cfg.fragment.line_end_max_nm:          # ends this wide are not line-end segments
        return 0.0
    L = 800.0
    frag = fragment([rect(-width / 2, -L / 2, width / 2, L / 2)], cfg.fragment)
    tip = np.flatnonzero((frag.kind == LINE_END) & (frag.normal[:, 1] > 0))
    grid = None

    def tip_epe(ext: float) -> float:
        nonlocal grid
        d = _rule_offsets(frag, cfg, table_lookup={width: table}, ext={width: ext})
        polys = frag.mask_polygons(d)
        r = sim.simulate(polys, grid=grid, contours=False)
        grid = r.grid
        e, _ = edge_displacement(ImageSampler(r.latent, r.grid), r.level, frag.sites[tip], frag.normal[tip])
        return float(e[0])

    hi = cfg.mbopc.max_out_nm
    if tip_epe(0.0) >= 0:
        return 0.0
    if tip_epe(hi) < 0:
        return hi
    return float(brentq(tip_epe, 0.0, hi, xtol=0.05))


def _nearest_table(tables: dict[float, BiasTable], width: float) -> BiasTable:
    return tables[min(tables, key=lambda w: abs(w - width))]


def _rule_offsets(frag: Fragmentation, cfg: Config, table_lookup: dict[float, BiasTable],
                  ext: dict[float, float]) -> np.ndarray:
    rc = cfg.rule
    d = np.zeros(len(frag))
    side = frag.kind != LINE_END
    for i in np.flatnonzero(side):
        d[i] = _nearest_table(table_lookup, frag.width[i])(frag.space[i])
    convex, concave = frag.kind == CORNER_CONVEX, frag.kind == CORNER_CONCAVE
    d[convex & ~frag.beside_line_end] += rc.serif_nm
    d[concave] -= rc.inner_serif_nm
    d[convex & frag.beside_line_end] += rc.hammer_nm
    # line-end segments: extension calibrated for the closest line width (width = the end edge length)
    for i in np.flatnonzero(frag.kind == LINE_END):
        w = frag.length[i]
        d[i] = ext[min(ext, key=lambda k: abs(k - w))]
    return d


def layout_widths(frag: Fragmentation, step: float = 5.0, max_width: float = 400.0) -> list[float]:
    """Distinct target line widths (rounded): side-segment widths and line-end edge lengths."""
    side = (frag.kind != LINE_END) & (frag.width < max_width)
    w = np.concatenate([frag.width[side], frag.length[frag.kind == LINE_END]])
    return sorted({float(step * round(x / step)) for x in w})


class RuleBasedOPC:
    def __init__(self, sim: LithoSimulator, cfg: Config):
        self.sim, self.cfg = sim, cfg
        self.deck = RuleDeck()

    def ensure_width(self, width: float) -> None:
        if width in self.deck.tables:
            return
        t = build_bias_table(self.sim, width, self.cfg.rule.bias_spaces_nm, self.cfg.rule.max_bias_nm)
        self.deck.tables[width] = t
        self.deck.line_end_ext[width] = calibrate_line_end(self.sim, self.cfg, width, t)

    def offsets(self, frag: Fragmentation) -> np.ndarray:
        for w in layout_widths(frag):
            self.ensure_width(w)
        d = _rule_offsets(frag, self.cfg, self.deck.tables, self.deck.line_end_ext)
        lo, hi = frag.bounds(self.cfg.mrc, self.cfg.mbopc.max_in_nm, self.cfg.mbopc.max_out_nm)
        return np.clip(d, lo, hi)


def insert_srafs(target: list[Polygon], width: float, distance: float, min_length: float,
                 end_pullback: float = 0.0, max_range: float = 200.0) -> list[Polygon]:
    """Rule-based assist features (scattering bars): a bar of ``width`` parallel to every target edge that faces
    open space, ``distance`` away from it. Bars are kept only where they stay ``distance`` away from every target
    shape and remain at least ``min_length`` long."""
    polys = [ensure_ccw(p) for p in target]
    bars = []
    for poly in polys:
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            L = float(np.hypot(*(b - a)))
            if L < min_length + 2 * end_pullback:
                continue
            t = (b - a) / L
            nrm = np.array([t[1], -t[0]])
            pts = a + np.outer([0.25, 0.5, 0.75], b - a)
            space = raycast(pts, np.repeat(nrm[None], 3, axis=0), polys, max_range)
            if space.min() < 2 * distance + width:
                continue
            p0 = a + t * end_pullback + nrm * distance
            p1 = b - t * end_pullback + nrm * (distance + width)
            bars.append(rect(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])))
    if not bars:
        return []
    keepout = gdstk.offset([gdstk.Polygon(p) for p in polys], distance - 1e-6, join="miter", use_union=True)
    pieces = gdstk.boolean([gdstk.Polygon(p) for p in union(bars)], keepout, "not")
    out = []
    for q in pieces:
        x0, y0 = q.points.min(axis=0)
        x1, y1 = q.points.max(axis=0)
        if min(x1 - x0, y1 - y0) >= width - 1e-6 and max(x1 - x0, y1 - y0) >= min_length:
            out.append(ensure_ccw(clean(q.points)))
    return out


# ==================================================================================================
# 4. Model-based OPC
# ==================================================================================================
#
# Model-based OPC: segment-wise EPE feedback.
#
# Iteration k:
#     mask_k   = Manhattan polygon rebuilt from segment offsets d_k
#     EPE_k    = signed EPE at every control site (simulated, nominal condition)
#     d_{k+1}  = clip(d_k - lambda * EPE_k, lo, hi)          (lo/hi from max move and MRC)
#
# Options (the three candidate contributions):
#     adaptive  per-segment lambda_i: x adapt_up while the sign of EPE_i is kept,
#               x adapt_down when it flips (overshoot), clipped to [damping_min, damping_max]
#     init=rule start from the rule-based offsets instead of the target
#     pw_aware  feedback on EPE_eff = w * EPE_nominal + (1 - w) * mean(EPE at +-dose, +-defocus)


@dataclass
class MBResult:
    offsets: np.ndarray
    mask: list
    epe: np.ndarray                       # final nominal EPE at control sites
    history: list[dict] = field(default_factory=list)
    converged_at: int | None = None       # first iteration with rms < tol_rms and max|EPE| < tol_max over the
                                          # sites not pinned at a move limit or blocked by MRC (None = never)
    runtime_s: float = 0.0
    masks: list = field(default_factory=list)   # mask of every iteration (keep_masks=True)


def _stats(e: np.ndarray) -> dict:
    return {"rms": float(np.sqrt(np.mean(e**2))), "max_abs": float(np.abs(e).max()),
            "mean_abs": float(np.abs(e).mean())}


def _mrc_guard(frag: Fragmentation, cfg: Config, d_new: np.ndarray, d_old: np.ndarray,
               rounds: int = 4, fixed: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Revert the segments touching any MRC violation of the updated mask to their previous offsets.

    Returns (offsets, blocked) where blocked marks the reverted segments.
    """
    d = d_new.copy()
    blocked = np.zeros(len(frag), bool)
    for _ in range(rounds):
        m = mrc_check(frag.mask_polygons(d) + list(fixed or []), cfg.mrc.min_width_nm, cfg.mrc.min_space_nm)
        viol = m["width_violations"] + m["space_violations"]
        if not viol:
            break
        pa, pb = frag.a + d[:, None] * frag.normal, frag.b + d[:, None] * frag.normal
        smin, smax = np.minimum(pa, pb), np.maximum(pa, pb)
        hit = np.zeros(len(frag), bool)
        for v in viol:
            vmin, vmax = v.min(axis=0) - 1.0, v.max(axis=0) + 1.0
            hit |= np.all((smin <= vmax) & (smax >= vmin), axis=1)
        d[hit] = d_old[hit]
        blocked |= hit & (d_new != d_old)
    return d, blocked


def run_mbopc(sim: LithoSimulator, cfg: Config, frag: Fragmentation, grid: Grid,
              init_offsets: np.ndarray | None = None, opts: MBOPCConfig | None = None,
              fixed: list[Polygon] | None = None, keep_masks: bool = False, effective=None) -> MBResult:
    """fixed: polygons added to every mask but never moved (e.g. assist features);
    keep_masks: store the mask of every iteration in MBResult.masks;
    effective: maps the drawn mask to the mask the optics sees (e.g. EUV absorber shadowing); MRC checks the drawn mask."""
    o = opts or cfg.mbopc
    t0 = time.time()
    lo, hi = frag.bounds(cfg.mrc, o.max_in_nm, o.max_out_nm)
    d = np.zeros(len(frag)) if init_offsets is None else np.clip(init_offsets, lo, hi)
    lam = np.full(len(frag), o.damping)
    prev = None
    blocked = np.zeros(len(frag), bool)
    conds = [(0.0, 1.0)]
    if o.pw_aware:
        conds += [(0.0, 1 + o.pw_dose), (0.0, 1 - o.pw_dose), (o.pw_defocus_nm, 1.0), (-o.pw_defocus_nm, 1.0)]
    focus = sorted({f for f, _ in conds})

    res = MBResult(offsets=d, mask=[], epe=np.zeros(len(frag)))
    for k in range(o.iterations + 1):
        mask = frag.mask_polygons(d) + list(fixed or [])
        if keep_masks:
            res.masks.append(mask)
        imgs = ProcessImages(sim, rasterize(effective(mask) if effective else mask, grid), grid, focus)
        epes = [imgs.site_epe(frag.sites, frag.normal, f, dose) for f, dose in conds]
        nominal = epes[0]
        eff = nominal if len(epes) == 1 else o.pw_nominal_weight * nominal + (1 - o.pw_nominal_weight) * np.mean(epes[1:], axis=0)
        st = _stats(nominal)
        free = ((d > lo + 1e-3) & (d < hi - 1e-3) | (np.abs(d) < 1e-9)) & ~blocked   # not pinned by move/MRC limits
        ef = nominal[free] if free.any() else np.zeros(1)
        st.update(iteration=k, lam_mean=float(lam.mean()), move_rms=float(np.sqrt(np.mean(d**2))),
                  rms_free=float(np.sqrt(np.mean(ef**2))), max_abs_free=float(np.abs(ef).max()),
                  pinned=int((~free).sum()))
        if len(epes) > 1:
            st["pw_worst"] = float(np.abs(np.array(epes)).max())
        res.history.append(st)
        if res.converged_at is None and st["rms_free"] < o.tol_rms_nm and st["max_abs_free"] < o.tol_max_nm:
            res.converged_at = k
        res.offsets, res.mask, res.epe = d.copy(), mask, nominal
        if k == o.iterations:
            break
        if o.adaptive and prev is not None:
            s = np.sign(eff) * np.sign(prev)
            lam = np.where(s > 0, np.minimum(lam * o.adapt_up, o.damping_max),
                           np.where(s < 0, np.maximum(lam * o.adapt_down, o.damping_min), lam))
        d_new = np.clip(d - lam * eff, lo, hi)
        d, blocked = (_mrc_guard(frag, cfg, d_new, d, fixed=fixed) if o.mrc_guard
                      else (d_new, np.zeros(len(frag), bool)))
        prev = eff
    res.runtime_s = time.time() - t0
    return res


# ==================================================================================================
# 5. Pixel ILT
# ==================================================================================================
#
# Pixel-based inverse lithography (ILT), used as a reference against segment-based OPC.
#
#     m(theta) = sigmoid(beta * theta)                 relaxed mask transmission
#     Z        = sigmoid(alpha * (latent(m) - th))     relaxed constant-threshold resist
#     loss     = mean(W (Z - target)^2) + w_bin * mean(m (1 - m)) + w_tv * TV(m)
#
# W down-weights squares of half-size corner_len around target vertices (the same
# corner-rounding allowance the segment-based OPC uses), so that pixel-area error
# at unavoidably rounded corners does not bias the edges.
#
# The simulator is written in torch, so the gradient comes from autograd through
# the FFT/SOCS imaging chain. The final mask is binarized at m = 0.5.


@dataclass
class ILTResult:
    mask: np.ndarray                  # binary transmission
    mask_continuous: np.ndarray
    polygons: list[Polygon]
    history: list[dict] = field(default_factory=list)
    runtime_s: float = 0.0


def run_ilt(sim: LithoSimulator, cfg: Config, target: list[Polygon], grid: Grid,
            opts: ILTConfig | None = None, init: np.ndarray | None = None) -> ILTResult:
    o = opts or cfg.ilt
    t0 = time.time()
    tgt = sim.to_tensor(rasterize(target, grid))
    c = cfg.fragment.corner_len_nm
    zones = rasterize([rect(x - c, y - c, x + c, y + c) for p in target for x, y in p], grid)
    W = sim.to_tensor(1.0 - (1.0 - o.corner_weight) * zones)
    start = tgt if init is None else sim.to_tensor(init)
    theta = ((start - 0.5) * 2.0).clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=o.lr)
    th = sim.threshold
    hist = []
    for it in range(o.iterations):
        m = torch.sigmoid(o.mask_steepness * theta)
        lat = sim.latent(sim.aerial(m, grid.pixel), grid.pixel)
        z = torch.sigmoid(o.resist_steepness * (lat - th))
        l_img = torch.mean(W * (z - tgt) ** 2)
        l_bin = torch.mean(m * (1 - m))
        l_tv = torch.mean(torch.abs(m[1:, :] - m[:-1, :])) + torch.mean(torch.abs(m[:, 1:] - m[:, :-1]))
        loss = l_img + o.binarization_weight * l_bin + o.tv_weight * l_tv
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append({"iteration": it, "loss": loss.item(), "image": l_img.item(), "bin": l_bin.item(),
                     "tv": l_tv.item()})
    with torch.no_grad():
        mc = torch.sigmoid(o.mask_steepness * theta).cpu().numpy().astype(float)
    binary = (mc > 0.5).astype(float)
    return ILTResult(mask=binary, mask_continuous=mc, polygons=raster_to_polygons(binary > 0.5, grid),
                     history=hist, runtime_s=time.time() - t0)
