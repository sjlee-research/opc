"""Tool pipeline for any layout: split into clips, run one correction method, save GDS / figure / CSV.

    runner = Runner(sim, cfg)
    case = runner.case(name, target_polygons)   # fragmentation + simulation grid
    result = runner.run(case, "mb_adaptive")     # mask, evaluation (EPE, PV band, MRC), convergence history
    correct_file(cfg, "examples/square.gds", "mb", "outputs")          # what `run.py correct` does
"""
from __future__ import annotations

import copy
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .correction import (Evaluation, Fragmentation, RuleBasedOPC, evaluate_mask, fidelity_metrics, fragment,
                         run_ilt, run_mbopc)
from .layout import (CONTOUR_SIMPLIFY_NM, LAYER_MASK, LAYER_PRINT_AFTER, LAYER_PRINT_BEFORE, LAYER_TARGET, Grid,
                     Polygon, bbox, even_odd_fill, example_library, read_gds_cells, read_layout, write_gds,
                     write_gds_grid)
from .litho import LithoSimulator

METHODS = ["no_opc", "rule", "mb", "mb_adaptive", "mb_hybrid", "mb_adaptive_hybrid", "mb_pw", "ilt"]
MB_VARIANTS = {                     # model-based OPC variants = baseline + candidate contributions
    "mb": {},
    "mb_adaptive": {"adaptive": True},                      # A: adaptive damping
    "mb_hybrid": {"init": "rule"},                          # B: rule-based initial mask
    "mb_adaptive_hybrid": {"adaptive": True, "init": "rule"},
    "mb_pw": {"pw_aware": True},                            # C: process-window-aware EPE
}


@dataclass
class Case:
    name: str
    target: list[Polygon]
    frag: Fragmentation
    grid: Grid


@dataclass
class MethodResult:
    method: str
    mask: list[Polygon]
    evaluation: Evaluation
    history: list[dict] = field(default_factory=list)
    converged_at: int | None = None
    runtime_s: float = 0.0
    offsets: np.ndarray | None = None


class Runner:
    def __init__(self, sim: LithoSimulator, cfg: Config):
        self.sim, self.cfg = sim, cfg
        self.rule = RuleBasedOPC(sim, cfg)

    def case(self, name: str, target: list[Polygon]) -> Case:
        g = self.cfg.grid
        return Case(name, target, fragment(target, self.cfg.fragment), Grid.around(target, g.pixel_nm, g.margin_nm))

    def run(self, case: Case, method: str) -> MethodResult:
        cfg, sim = self.cfg, self.sim
        t0 = time.time()
        if method == "no_opc":
            mask, hist, conv, off = case.target, [], None, np.zeros(len(case.frag))
        elif method == "rule":
            off = self.rule.offsets(case.frag)
            mask, hist, conv = case.frag.mask_polygons(off), [], None
        elif method in MB_VARIANTS:
            o = copy.deepcopy(cfg.mbopc)
            o.adaptive, o.init, o.pw_aware = False, "none", False
            for k, v in MB_VARIANTS[method].items():
                setattr(o, k, v)
            init = self.rule.offsets(case.frag) if o.init == "rule" else None
            r = run_mbopc(sim, cfg, case.frag, case.grid, init_offsets=init, opts=o)
            mask, hist, conv, off = r.mask, r.history, r.converged_at, r.offsets
        elif method == "ilt":
            r = run_ilt(sim, cfg, case.target, case.grid)
            mask, hist, conv, off = r.polygons, r.history, None, None
        else:
            raise ValueError(f"unknown method '{method}' (choose from {METHODS})")
        runtime = time.time() - t0
        ev = evaluate_mask(sim, cfg, case.target, case.frag, case.grid, mask_polygons=mask)
        return MethodResult(method, mask, ev, hist, conv, runtime, off)


def result_row(case: Case, r: MethodResult) -> dict:
    """One CSV row of metrics for (pattern, method)."""
    d = {"pattern": case.name, "method": r.method, "segments": len(case.frag),
         "iterations_to_converge": r.converged_at if r.converged_at is not None else "",
         "rms_at_iter3": round(r.history[3]["rms"], 4) if r.method in MB_VARIANTS else "",
         "runtime_s": round(r.runtime_s, 2)}
    d.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.evaluation.row().items()})
    return d


def write_csv(path: str | Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))      # union of keys, first-seen order
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def printed_polygons(r: MethodResult) -> list[Polygon]:
    """Nominal printed contour of a result as GDS-ready filled polygons."""
    return even_odd_fill(r.evaluation.contours, CONTOUR_SIMPLIFY_NM)


def cluster_polygons(polygons: list[Polygon], distance: float) -> list[list[int]]:
    """Group polygons whose bounding boxes come closer than ``distance`` (optically coupled clips)."""
    n = len(polygons)
    boxes = np.array([bbox([p]) for p in polygons])
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lo, hi = boxes[:, :2] - distance / 2, boxes[:, 2:] + distance / 2
    for i in range(n):
        near = np.all((lo[i] <= hi[i + 1:]) & (lo[i + 1:] <= hi[i]), axis=1)
        for j in np.flatnonzero(near) + i + 1:
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def correct_file(cfg: Config, layout: str | Path, method: str, out: str | Path, layer: int = LAYER_TARGET,
                 cell: str | None = None, pixel_nm: float | None = None) -> Path:
    """Run ``method`` on every optically isolated clip of a GDS/PNG layout and write two files:

    <stem>_opc.gds   layer 1 target, 2 corrected mask, 3 printed contour before OPC, 4 after OPC
    <stem>_opc.png   left: original mask (= target) and its print, right: OPC mask and its print
    (method "mb" is the standard model-based OPC; other methods use their name instead of "opc";
    method "no_opc" only simulates and writes <stem>_sim.*).
    """
    from . import viz

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target = read_layout(layout, layer, pixel_nm, cell)
    sim = LithoSimulator(cfg)
    runner = Runner(sim, cfg)
    t0 = time.time()
    print(f"{len(target)} target polygons, method {method}, threshold {sim.threshold:.4f}")
    clips = []
    for i, idx in enumerate(cluster_polygons(target, 2 * cfg.grid.margin_nm)):
        c = runner.case(f"clip{i}", [target[j] for j in idx])
        before = runner.run(c, "no_opc")
        after = before if method == "no_opc" else runner.run(c, method)
        mb_, ma_ = (fidelity_metrics(sim, cfg, c.target, c.grid, r.mask) for r in (before, after))
        clips.append((c, before, after, mb_, ma_))
        print(f"  clip {i}: EPE on edges mean {mb_['epe_mean']:.2f} -> {ma_['epe_mean']:.2f} nm, "
              f"max {mb_['epe_max']:.2f} -> {ma_['epe_max']:.2f} nm, corner {mb_['corner']:.1f} -> {ma_['corner']:.1f} nm, "
              f"area error {mb_['area_pct']:.1f} -> {ma_['area_pct']:.1f} %"
              + (f", converged at iteration {after.converged_at}" if after.converged_at is not None else ""))

    tag = {"mb": "opc", "no_opc": "sim"}.get(method, method)
    stem = f"{Path(layout).stem}_{tag}"
    layers = {LAYER_TARGET: target, LAYER_PRINT_BEFORE: [p for _, b, _, _, _ in clips for p in printed_polygons(b)]}
    if method != "no_opc":
        layers[LAYER_MASK] = [p for _, _, a, _, _ in clips for p in a.mask]
        layers[LAYER_PRINT_AFTER] = [p for _, _, a, _, _ in clips for p in printed_polygons(a)]
    write_gds(out / f"{stem}.gds", layers)

    opc = method != "no_opc"
    label = {"mb": "model-based OPC"}.get(method, method)
    viz.fig_before_after([
        {"name": c.name, "target": c.target, "printed_before": b.evaluation.contours,
         "mask_after": a.mask if opc else None, "printed_after": a.evaluation.contours,
         "metrics_before": mb_, "metrics_after": ma_ if opc else None,
         "footer_after": f"{label}" + (f", converged in {a.converged_at} iterations" if a.converged_at is not None else "")}
        for c, b, a, mb_, ma_ in clips], path=out / f"{stem}.png",
        title=f"{Path(layout).stem}: " + (label if opc else "simulation without OPC"))
    print(f"done in {time.time() - t0:.1f}s -> {out / (stem + '.gds')}, {out / (stem + '.png')}")
    return out / f"{stem}.gds"


def write_example_library(path: str | Path, cd: float = 60.0) -> Path:
    """examples/examples_100.gds: the 100 example targets on layer 1, one cell each, names written on layer 10."""
    pats = example_library(cd)
    write_gds_grid(path, [("", [(p.name, {LAYER_TARGET: p.polygons}) for p in pats[i:i + 10]])
                          for i in range(0, len(pats), 10)], gap_nm=900.0, label_all=True)
    return Path(path)


def correct_examples(cfg: Config, library: str | Path, method: str, out: str | Path) -> Path:
    """Run ``method`` on every cell of a GDS library separately (e.g. examples/examples_100.gds).

    out/<cell>.png        before/after figure with fidelity metrics, one per cell
    out/<lib>_opc.gds     every result in one GDS (layers 1-4, cell names on layer 10)
    out/summary.csv       metrics before/after per cell
    out/overview.png      contact sheet of all OPC results
    out/metrics.png       distribution of each metric before vs after
    """
    from . import viz

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cells = read_gds_cells(library)
    sim = LithoSimulator(cfg)
    runner = Runner(sim, cfg)
    label = {"mb": "model-based OPC"}.get(method, method)
    done, rows, t_all = [], [], time.time()
    print(f"{len(cells)} cells in {library}, method {method}, threshold {sim.threshold:.4f}")
    for i, (name, target) in enumerate(cells):
        t0 = time.time()
        row = {"cell": name, "shape": name.split("_")[1] if "_" in name else name}
        try:
            c = runner.case(name, target)
            b, a = runner.run(c, "no_opc"), runner.run(c, method)
            mb_, ma_ = (fidelity_metrics(sim, cfg, c.target, c.grid, r.mask) for r in (b, a))
            viz.fig_before_after([{"name": name, "target": c.target, "printed_before": b.evaluation.contours,
                                   "mask_after": a.mask, "printed_after": a.evaluation.contours,
                                   "metrics_before": mb_, "metrics_after": ma_,
                                   "footer_after": label + (f", converged in {a.converged_at} iterations"
                                                            if a.converged_at is not None else "")}],
                                 path=out / f"{name}.png", title=f"{name}: {label}")
            done.append((c, b, a, mb_, ma_))
            row.update({f"{k}_before": round(v, 3) for k, v in mb_.items() if isinstance(v, float)})
            row.update({f"{k}_after": round(v, 3) for k, v in ma_.items() if isinstance(v, float)})
            row.update({"iterations": a.converged_at if a.converged_at is not None else "",
                        "pv_band_nm": round(a.evaluation.pvb_mean, 3),
                        "mrc_violations": a.evaluation.mrc["n_width"] + a.evaluation.mrc["n_space"],
                        "segments": len(c.frag), "runtime_s": round(time.time() - t0, 1), "error": ""})
            print(f"  [{i + 1:3d}/{len(cells)}] {name:36s} area error {mb_['area_pct']:5.1f} -> {ma_['area_pct']:4.1f} %, "
                  f"EPE mean {mb_['epe_mean']:5.2f} -> {ma_['epe_mean']:4.2f} nm  ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:                                    # keep going; the failure is listed in the CSV
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"  [{i + 1:3d}/{len(cells)}] {name:36s} FAILED: {row['error']}", flush=True)
        rows.append(row)

    write_csv(out / "summary.csv", rows)
    stem = Path(library).stem
    res = [(c.name, {LAYER_TARGET: c.target, LAYER_MASK: a.mask, LAYER_PRINT_BEFORE: printed_polygons(b),
                     LAYER_PRINT_AFTER: printed_polygons(a)}) for c, b, a, _, _ in done]
    write_gds_grid(out / f"{stem}_opc.gds", [("", res[i:i + 10]) for i in range(0, len(res), 10)], gap_nm=900.0,
                   label_all=True)
    viz.fig_thumbnails([{"name": c.name, "target": c.target, "mask": a.mask, "printed": a.evaluation.contours,
                         "caption": f"area error {mb_['area_pct']:.1f}→{ma_['area_pct']:.1f}%"}
                        for c, b, a, mb_, ma_ in done], path=out / "overview.png",
                       title=f"{len(done)} examples after {label} (gray target, violet mask, blue printed)")
    viz.fig_metric_distribution([d[3] for d in done], [d[4] for d in done], path=out / "metrics.png",
                                title=f"{len(done)} examples: print fidelity with the original mask vs the OPC mask")
    n_fail = len(rows) - len(done)
    print(f"done in {time.time() - t_all:.0f}s, {len(done)} ok, {n_fail} failed -> {out}")
    return out

