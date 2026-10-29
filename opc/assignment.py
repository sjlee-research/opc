"""INV_X1 / NAND2_X4 Metal1 OPC study (`python run.py assignment`) -> outputs/assignment/.

    figures/  paper figures, one column wide (Fig3 is a screenshot of the web app, added by hand)
        Fig1_targets.png             target patterns on the 3.5 nm lambda grid
        Fig2_light_distribution.png  2D Gaussian light distribution + light amount per lambda cell
        Fig4_INV_X1.png              without / with OPC: mask, exposure distribution, developed pattern vs target
        Fig5_NAND2_X4.png
        Fig6_dose.png                dose variation (kept separate from the OPC effect)
    appendix/ supplementary figures (GitHub appendix)
        A1_opc_progress.png          convergence of the model-based correction + mask snapshots
        A2_conditions.png            dose and Gaussian-width variation
        A3_model_compare.png         intensity superposition vs Hopkins interference model
        A4_sraf_focus.png            assist features through focus (Hopkins)
    tables/T1..T6 *.csv              conditions, geometry, metrics, CD per gauge, model comparison, conditions
    gds/targets.gds, gds/opc_result.gds (+ layers.lyp)
    conditions.yaml                  configuration + derived values (Gaussian sigma, threshold, grid per cell)
    metrics.json                     every number quoted in the paper

Before and after OPC use the same grid, light distribution, dose and threshold; condition sweeps and the model comparison
are reported separately from the correction effect.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import gdstk
import numpy as np
import yaml

from . import viz
from .config import Config
from .correction import LINE_END, ProcessImages, fidelity_metrics, insert_srafs, run_mbopc
from .layout import (CELL_RECTS, LAYER_MASK, LAYER_PRINT_AFTER, LAYER_PRINT_BEFORE, LAYER_TARGET, bbox, cell_targets,
                     even_odd_fill, fig_to_layout, rasterize, write_gds_grid, write_klayout_lyp)
from .litho import ImageSampler, LithoSimulator, edge_displacement, intensity_psf, psf_cell_table, psf_sigma
from .pipeline import Runner, printed_polygons, write_csv

DOSES = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
DEFOCUS = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
FIG4_TABLE = [[0, 0.5, 1.0, 0.5, 0], [0.5, 4.0, 5.0, 4.0, 0.5], [1.0, 5.0, 10.0, 5.0, 1.0],
              [0.5, 4.0, 5.0, 4.0, 0.5], [0, 0.5, 1.0, 0.5, 0]]      # weight-table PSF of the assignment sheet


def _scalar(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _mismatch(sim: LithoSimulator, case, mask, defocus: float = 0.0) -> float:
    imgs = ProcessImages(sim, rasterize(mask, case.grid), case.grid, [defocus])
    P = [gdstk.Polygon(p) for p in even_odd_fill(imgs.contours(defocus, 1.0))]
    T = [gdstk.Polygon(p) for p in case.target]
    area = lambda q: sum(abs(x.area()) for x in q)
    return 100 * area(gdstk.boolean(P, T, "xor")) / area(T)


def _line_end_pullback(case, exposure: np.ndarray, threshold: float) -> np.ndarray:
    """Distance the printed line ends fall short of the target ends (+ = shorter), at every line-end segment."""
    le = case.frag.kind == LINE_END
    disp, _ = edge_displacement(ImageSampler(exposure, case.grid), threshold, case.frag.sites[le],
                                case.frag.normal[le])
    return -disp


def _standard(cfg: Config):
    o = copy.deepcopy(cfg.mbopc)
    o.adaptive, o.init, o.pw_aware = False, "none", False
    return o


def run_assignment(cfg: Config, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    fig_dir, app_dir, tab_dir, gds_dir = out / "figures", out / "appendix", out / "tables", out / "gds"
    for d in (fig_dir, app_dir, tab_dir, gds_dir):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    sim = LithoSimulator(cfg)
    runner = Runner(sim, cfg)
    o = cfg.optics
    sigma = psf_sigma(o)
    radius = o.psf_radius_sigma * sigma
    th = sim.threshold
    cells = cell_targets()
    cases = {p.name: runner.case(p.name, p.polygons) for p in cells}
    gauges = {p.name: p.gauges for p in cells}
    M: dict = {"model": {"psf": o.psf, "sigma_nm": sigma, "radius_nm": radius, "threshold": th,
                         "pixel_nm": cfg.grid.pixel_nm, "margin_nm": cfg.grid.margin_nm,
                         "wavelength_nm": o.wavelength_nm, "NA": o.NA}, "cells": {}}
    print(f"[assignment] light distribution {o.psf} sigma {sigma:.3f} nm, cut-off {radius:.2f} nm, threshold {th}")

    # ---- conditions -----------------------------------------------------------------------------------
    grid_rows = []
    for name, c in cases.items():
        g = c.grid
        grid_rows.append({"cell": name, "pixel_nm": g.pixel, "pixels": f"{g.nx} x {g.ny}",
                          "area_nm": f"{g.nx * g.pixel:g} x {g.ny * g.pixel:g}",
                          "origin_nm": f"({g.x0:g}, {g.y0:g})"})
        M["cells"][name] = {"grid": grid_rows[-1]}
    cond = cfg.to_dict()
    cond["derived"] = {"psf_sigma_nm": sigma, "psf_cutoff_nm": radius, "threshold": th, "grids": grid_rows}
    (out / "conditions.yaml").write_text(yaml.safe_dump(cond, sort_keys=False, allow_unicode=True))
    write_csv(tab_dir / "T1_conditions.csv", [
        {"item": "grid", "value": f"{cfg.grid.pixel_nm:g} nm per cell (λ/7), computation area = cell + "
                                  f"{cfg.grid.margin_nm:g} nm on every side"},
        *[{"item": f"area {r['cell']}", "value": f"{r['pixels']} cells = {r['area_nm']} nm"} for r in grid_rows],
        {"item": "light distribution", "value": f"2D Gaussian, σ = 0.21·λ/NA = {sigma:.3f} nm "
                                                f"(λ = {o.wavelength_nm:g} nm, NA = {o.NA:g})"},
        {"item": "range / normalization", "value": f"cut off at r = {o.psf_radius_sigma:g}σ = {radius:.2f} nm, "
                                                   "weights normalized to sum 1 (open area -> exposure 1)"},
        {"item": "exposure", "value": "E(x) = Σ mask(x')·G(x − x'), G = normalized Gaussian (light amounts added, no phase)"},
        {"item": "resist / threshold", "value": f"positive resist, E ≥ {th:g} dissolves (trench → Metal1)"},
        {"item": "dose", "value": "1.0 for before/after comparison; 0.85–1.15 only in the condition study"},
        {"item": "OPC", "value": f"model-based, fragments {cfg.fragment.corner_len_nm:g} / {cfg.fragment.segment_len_nm:g}"
                                 f" nm, damping {cfg.mbopc.damping}, {cfg.mbopc.iterations} iterations, "
                                 f"moves −{cfg.mbopc.max_in_nm:g}…+{cfg.mbopc.max_out_nm:g} nm, "
                                 f"MRC {cfg.mrc.min_width_nm:g} nm"},
    ])
    rect_rows = [{"cell": n, "x0": r[0], "y0": r[1], "x1": r[2], "y1": r[3]} for n, rs in CELL_RECTS.items() for r in rs]
    write_csv(tab_dir / "T2_target_rectangles.csv", rect_rows)

    # ---- Fig1 targets, Fig2 light distribution -------------------------------------------------------
    table = psf_cell_table(o, 3.5, 5)
    viz.fig_light_distribution(intensity_psf(o, 0.25), 0.25, sigma, radius, table,
                               path=fig_dir / "Fig2_light_distribution.png")
    M["model"]["psf_cell_table"] = np.round(table, 3).tolist()
    L = lambda x, y: fig_to_layout(x, y)
    notes = {"INV_X1": [(L(56, 189), L(56, 178.5), "10.5 nm", "above"), (L(21, 120), L(42, 120), "21 nm", "above"),
                        (L(15.75, 73.5), L(15.75, 115.5), "42 nm", "right")],
             "NAND2_X4": [(L(47.25, 10.5), L(47.25, 21), "10.5 nm", "right"),
                          (L(47.25, 168), L(47.25, 178.5), "10.5 nm", "right"),
                          (L(21, 140), L(42, 140), "21 nm", "above"), (L(10.5, 95), L(42, 95), "31.5 nm pitch", "above")]}
    viz.fig_targets([{"name": p.name, "target": p.polygons, "notes": notes[p.name]} for p in cells],
                    path=fig_dir / "Fig1_targets.png")

    # ---- before / after OPC ---------------------------------------------------------------------------
    results = {}
    metric_rows, cd_rows = [], []
    sizes = [bbox(p.polygons) for p in cells]
    window = (max(b[2] - b[0] for b in sizes) + 16, max(b[3] - b[1] for b in sizes) + 16)   # one scale for both cells
    for p in cells:
        c = cases[p.name]
        before = runner.run(c, "no_opc")
        mb = run_mbopc(sim, cfg, c.frag, c.grid, opts=_standard(cfg), keep_masks=True)
        fb = fidelity_metrics(sim, cfg, c.target, c.grid, c.target, gauges[p.name], details=True)
        fa = fidelity_metrics(sim, cfg, c.target, c.grid, mb.mask, gauges[p.name], details=True)
        le_b, le_a = _line_end_pullback(c, fb["exposure"], th), _line_end_pullback(c, fa["exposure"], th)
        results[p.name] = dict(case=c, before=before, mb=mb, fb=fb, fa=fa)
        for tag, f, le in (("without OPC", fb, le_b), ("with OPC", fa, le_a)):
            row = {"cell": p.name, "mask": tag, **_scalar(f), "line_end_pullback_mean": round(float(le.mean()), 4),
                   "line_end_pullback_max": round(float(le.max()), 4), "line_ends": int(len(le))}
            metric_rows.append(row)
        for (g, wb), (_, wa) in zip(fb["cd"], fa["cd"]):
            cd_rows.append({"cell": p.name, "gauge": g, "target_nm": 10.5, "without_opc_nm": round(wb, 3),
                            "with_opc_nm": round(wa, 3)})
        M["cells"][p.name].update(before={**_scalar(fb), "line_end_pullback_mean": float(le_b.mean()),
                                          "line_end_pullback_max": float(le_b.max())},
                                  after={**_scalar(fa), "line_end_pullback_mean": float(le_a.mean()),
                                         "line_end_pullback_max": float(le_a.max())},
                                  iterations_to_converge=mb.converged_at, segments=len(c.frag),
                                  mask_vertices=int(sum(len(q) for q in mb.mask)),
                                  mrc_violations=_mrc_count(cfg, mb.mask),
                                  runtime_s=round(mb.runtime_s, 2))
        viz.fig_cell_result(p.name, c.target, [
            dict(label="without OPC", mask=c.target, grid=c.grid, exposure=fb["exposure"], threshold=th,
                 contours=fb["contours"], color=viz.C_PRINT_BEFORE, metrics=fb),
            dict(label="with OPC", mask=mb.mask, grid=c.grid, exposure=fa["exposure"], threshold=th,
                 contours=fa["contours"], color=viz.C_PRINT_AFTER, metrics=fa)],
            path=fig_dir / f"Fig{4 if p.name == 'INV_X1' else 5}_{p.name}.png",
            width_in=viz.PAGE_W_IN, window=window)
        print(f"  {p.name}: IoU {fb['iou']:.3f} -> {fa['iou']:.3f}, mismatch {fb['area_pct']:.2f} -> "
              f"{fa['area_pct']:.2f} %, converged at {mb.converged_at}", flush=True)
    write_csv(tab_dir / "T3_metrics.csv", metric_rows)
    write_csv(tab_dir / "T4_cd_gauges.csv", cd_rows)

    # ---- correction progress --------------------------------------------------------------------------
    curves, snaps = {}, []
    for name, r in results.items():
        c, mb = r["case"], r["mb"]
        fm = [fidelity_metrics(sim, cfg, c.target, c.grid, m) for m in mb.masks]
        curves[name] = dict(it=[h["iteration"] for h in mb.history], epe_rms=[h["rms"] for h in mb.history],
                            epe_max=[h["max_abs"] for h in mb.history], mism=[f["area_pct"] for f in fm],
                            iou=[f["iou"] for f in fm], conv=mb.converged_at)
        M["cells"][name]["progress"] = {"mismatch_pct": [round(f["area_pct"], 3) for f in fm],
                                        "iou": [round(f["iou"], 4) for f in fm],
                                        "epe_rms": [round(h["rms"], 4) for h in mb.history]}
    nand = results["NAND2_X4"]
    conv = nand["mb"].converged_at or cfg.mbopc.iterations
    for it in sorted({0, 1, 3, conv, cfg.mbopc.iterations}):
        mask = nand["mb"].masks[it]
        cont = ProcessImages(sim, rasterize(mask, nand["case"].grid), nand["case"].grid, [0.0]).contours(0.0, 1.0)
        snaps.append((it, mask, cont))
    viz.fig_opc_progress(curves, snaps, nand["case"].target, path=app_dir / "A1_opc_progress.png")

    # ---- condition variation (separate from the correction effect) ------------------------------------
    cond_rows, dose_lines, sig_lines = [], {}, {}
    sigmas = [4.0, 4.5, round(sigma, 3), 5.75, 6.5]
    sims_sigma = {}
    for s in sigmas:
        cs = copy.deepcopy(cfg)
        cs.optics.psf_sigma_nm = s
        sims_sigma[s] = LithoSimulator(cs)
    for name, r in results.items():
        c = r["case"]
        for tag, mask, col in (("without OPC", c.target, viz.C_PRINT_BEFORE), ("with OPC", r["mb"].mask, viz.C_PRINT_AFTER)):
            ls = "-" if name == "INV_X1" else "--"
            vals, flags = [], []
            for d in DOSES:
                f = fidelity_metrics(sim, cfg, c.target, c.grid, mask, dose=d)
                vals.append(f["area_pct"])
                flags.append(f["bridges"] + f["opens"] + f["missing"] > 0)
                cond_rows.append({"study": "dose", "value": d, "cell": name, "mask": tag, **_scalar(f)})
            dose_lines[f"{name}, {tag}"] = (vals, col, ls, flags)
            vals, flags = [], []
            for s in sigmas:
                f = fidelity_metrics(sims_sigma[s], cfg, c.target, c.grid, mask)
                vals.append(f["area_pct"])
                flags.append(f["bridges"] + f["opens"] + f["missing"] > 0)
                cond_rows.append({"study": "psf_sigma_nm", "value": s, "cell": name, "mask": tag, **_scalar(f)})
            sig_lines[f"{name}, {tag}"] = (vals, col, ls, flags)
    viz.fig_conditions({"x": DOSES, "lines": dose_lines, "nominal": 1.0},
                       {"x": sigmas, "lines": sig_lines, "nominal": sigma}, path=app_dir / "A2_conditions.png")
    viz.fig_dose({"x": DOSES, "lines": dose_lines, "nominal": 1.0}, path=fig_dir / "Fig6_dose.png")

    # weight-table PSF of the assignment sheet (cells of 3.5 nm)
    ca = copy.deepcopy(cfg)
    ca.optics.psf, ca.optics.psf_array, ca.optics.psf_array_cell_nm = "array", FIG4_TABLE, 3.5
    sim_a = LithoSimulator(ca)
    M["kernel_table"] = {}
    for name, r in results.items():
        c = r["case"]
        mb_a = run_mbopc(sim_a, ca, c.frag, c.grid, opts=_standard(ca))
        rows = {"original mask": c.target, "Gaussian-model OPC mask": r["mb"].mask, "table-PSF OPC mask": mb_a.mask}
        M["kernel_table"][name] = {}
        for tag, mask in rows.items():
            f = fidelity_metrics(sim_a, ca, c.target, c.grid, mask)
            cond_rows.append({"study": "table_psf", "value": "Fig.4 table, 3.5 nm cells", "cell": name, "mask": tag,
                              **_scalar(f)})
            M["kernel_table"][name][tag] = _scalar(f)
    write_csv(tab_dir / "T6_condition_study.csv", cond_rows)
    M["conditions"] = cond_rows

    # ---- Hopkins interference model -------------------------------------------------------------------
    ch = copy.deepcopy(cfg)
    ch.optics.model = "hopkins"
    ch.resist.threshold = None                     # edge rule: a large feature's edge prints on target
    sim_h = LithoSimulator(ch)
    th_h = sim_h.threshold
    M["hopkins"] = {"threshold": th_h, "illumination": f"{ch.optics.illumination} {ch.optics.sigma_in}/{ch.optics.sigma_out}"}
    cmp_rows, cmp_fig = [], []
    for name, r in results.items():
        c = r["case"]
        mb_h = run_mbopc(sim_h, ch, c.frag, c.grid, opts=_standard(ch))
        r["mb_h"] = mb_h
        masks = {"original mask": c.target, "intensity-model OPC mask": r["mb"].mask, "Hopkins-model OPC mask": mb_h.mask}
        conts = {}
        M["hopkins"][name] = {}
        for tag, mask in masks.items():
            f = fidelity_metrics(sim_h, ch, c.target, c.grid, mask, gauges[name], details=True)
            cmp_rows.append({"cell": name, "evaluated with": "Hopkins", "mask": tag, **_scalar(f)})
            M["hopkins"][name][tag] = _scalar(f)
            conts[tag] = f
        for tag, f in (("original mask", r["fb"]), ("intensity-model OPC mask", r["fa"])):
            cmp_rows.append({"cell": name, "evaluated with": "intensity", "mask": tag, **_scalar(f)})
        cmp_fig.append(dict(name=name, target=c.target, grid=c.grid, exp_int=r["fb"]["exposure"], thr_int=th,
                            exp_hop=conts["original mask"]["exposure"], thr_hop=th_h,
                            cont={"original mask": (conts["original mask"]["contours"], viz.C_PRINT_BEFORE),
                                  "intensity-model OPC mask": (conts["intensity-model OPC mask"]["contours"], viz.C_CROSS),
                                  "Hopkins-model OPC mask": (conts["Hopkins-model OPC mask"]["contours"], viz.C_PRINT_AFTER)}))
    write_csv(tab_dir / "T5_model_comparison.csv", cmp_rows)
    viz.fig_model_compare(cmp_fig, path=app_dir / "A3_model_compare.png")

    # ---- assist features through focus (Hopkins) and at nominal (intensity) ----------------------------
    curves_f, M["sraf"] = {}, {}
    for i, (name, r) in enumerate(results.items()):
        c = r["case"]
        sr = insert_srafs(c.target, cfg.mrc.min_width_nm, 14.0, 10.5, end_pullback=3.5)
        mb_hs = run_mbopc(sim_h, ch, c.frag, c.grid, opts=_standard(ch), fixed=sr)
        mb_is = run_mbopc(sim, cfg, c.frag, c.grid, opts=_standard(cfg), fixed=sr)
        no = [_mismatch(sim_h, c, r["mb_h"].mask, f) for f in DEFOCUS]
        yes = [_mismatch(sim_h, c, mb_hs.mask, f) for f in DEFOCUS]
        col = viz.C_SERIES[i]
        curves_f[f"{name}, OPC"] = (DEFOCUS, no, col, "-")
        curves_f[f"{name}, OPC + assist features"] = (DEFOCUS, yes, col, "--")
        fi = {d: (fidelity_metrics(sim, cfg, c.target, c.grid, r["mb"].mask, dose=d)["area_pct"],
                  fidelity_metrics(sim, cfg, c.target, c.grid, mb_is.mask, dose=d)["area_pct"]) for d in (0.95, 1.0, 1.05)}
        M["sraf"][name] = {"bars": len(sr), "defocus_nm": DEFOCUS, "hopkins_opc": no, "hopkins_opc_sraf": yes,
                           "intensity_dose": {str(d): {"opc": v[0], "opc_sraf": v[1]} for d, v in fi.items()}}
        if name == "NAND2_X4":
            view = {"name": name, "target": c.target, "mask": mb_hs.mask[:len(mb_hs.mask) - len(sr)], "srafs": sr}
    viz.fig_sraf_focus(curves_f, view, path=app_dir / "A4_sraf_focus.png")

    # ---- GDS --------------------------------------------------------------------------------------------
    write_gds_grid(gds_dir / "targets.gds", [("", [(p.name, {LAYER_TARGET: p.polygons}) for p in cells])],
                   gap_nm=120.0, label_nm=10.0, label_all=True)
    write_gds_grid(gds_dir / "opc_result.gds", [("", [(name, {
        LAYER_TARGET: r["case"].target, LAYER_MASK: r["mb"].mask, LAYER_PRINT_BEFORE: printed_polygons(r["before"]),
        LAYER_PRINT_AFTER: even_odd_fill(r["fa"]["contours"], 0.05)}) for name, r in results.items()])],
        gap_nm=120.0, label_nm=10.0, label_all=True)
    write_klayout_lyp(gds_dir / "layers.lyp")

    M["runtime_s"] = round(time.time() - t_start, 1)
    (out / "metrics.json").write_text(json.dumps(M, indent=1, ensure_ascii=False, default=float))
    print(f"done in {M['runtime_s']} s -> {out}")
    return M


def _mrc_count(cfg: Config, mask) -> int:
    from .correction import mrc_check
    m = mrc_check(mask, cfg.mrc.min_width_nm, cfg.mrc.min_space_nm)
    return int(m["n_width"] + m["n_space"])
