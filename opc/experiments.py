"""Report experiments, stages 1-5 (`python run.py stage 1 2 3 4 5`). Everything lands in outputs/report/:

    summary.md   the report draft: numbers, tables and every figure in reading order
    figures/     Fig01-Fig13 (main) and FigS1-FigS4 (supplementary)
    tables/      T1-T7 + raw_site_epe.csv, raw_convergence.csv
    gds/         test_patterns.gds (input), results.gds (every method x pattern), layers.lyp

    stage 1  simulator check + proximity effects without OPC   Fig01-05, FigS3-S4, T1-T4, test_patterns.gds
    stage 2  rule-based OPC                                    Fig06, FigS2, T5
    stage 3  model-based OPC                                   Fig07-09
    stage 4  candidate contributions + ILT                     Fig10-13, FigS1, T6-T7, raw tables, results.gds
    stage 5  summary.md (reads the tables, so it can be rerun alone)

Stages 2-4 share one Study: every (pattern, method) is computed once.
"""
from __future__ import annotations

import copy
import csv
import time
from pathlib import Path

import numpy as np

from . import viz
from .config import Config
from .correction import KIND_NAMES, RuleBasedOPC
from .layout import (LAYER_MASK, LAYER_PRINT_AFTER, LAYER_PRINT_BEFORE, LAYER_TARGET, Grid, l_shape, rasterize,
                     standard_patterns, write_gds_grid, write_klayout_lyp)
from .litho import (ImageSampler, LithoSimulator, abbe_aerial, gaussian_sigma, measure_epe, measure_gauge,
                    periodic_cd, periodic_profile, sample_edges, source_bitmap, spatial_kernels)
from .pipeline import METHODS, MB_VARIANTS, MethodResult, Runner, printed_polygons, result_row, write_csv

MB = ["mb", "mb_adaptive", "mb_hybrid", "mb_adaptive_hybrid"]
RECOMMENDED = "mb_adaptive"


class Paths:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.fig, self.tab, self.gds = self.root / "figures", self.root / "tables", self.root / "gds"
        for d in (self.fig, self.tab, self.gds):
            d.mkdir(parents=True, exist_ok=True)


def _gallery_gauges(p) -> list[str]:
    """Which gauges to print under each no-OPC gallery panel."""
    fixed = {"iso_line": ["cd_center", "line_end"], "line_end": ["tip_upper", "cd_center"],
             "l_shape": ["outer_corner", "inner_corner", "end_h_arm"],
             "t_shape": ["inner_corner_left", "outer_corner", "end_stem"], "contact_iso": ["cd_x", "corner"]}
    if p.name.startswith("dense"):
        return ["cd_center", "cd_outer", "line_end"]
    if p.name.startswith("contact_array"):
        return ["cd_x_center", "cd_x_corner"]
    return fixed.get(p.name, [])


# =================================================================================================== stage 1
def stage1(cfg: Config, out: Paths) -> None:
    """Simulator validation and proximity effects without correction."""
    print("[stage 1] simulator check + proximity effects without OPC")
    pc = cfg.patterns

    # SOCS truncation vs brute-force Abbe (float64)
    c64 = copy.deepcopy(cfg)
    c64.dtype = "float64"
    lsh = l_shape(pc.cd_nm, pc.arm_length_nm)
    grid = Grid.around(lsh.polygons, cfg.grid.pixel_nm, cfg.grid.margin_nm)
    mask = rasterize(lsh.polygons, grid)
    ref = abbe_aerial(mask, cfg.optics, grid.pixel)
    socs = []
    for K in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 10_000]:
        c64.optics.num_kernels = K
        s = LithoSimulator(c64)
        img = s.aerial(s.to_tensor(mask), grid.pixel).numpy()
        k = s.kernels(grid.shape, grid.pixel).info
        socs.append({"K": len(k.weights), "captured_trace": round(k.captured_energy, 5),
                     "max_abs_err": float(np.abs(img - ref).max()), "rel_max_err": float(np.abs(img - ref).max() / ref.max())})
    write_csv(out.tab / "T2_socs_vs_abbe.csv", socs)
    k_row = next(r for r in socs if r["K"] == cfg.optics.num_kernels)
    print(f"  SOCS full vs Abbe max err {socs[-1]['max_abs_err']:.1e}; K={k_row['K']}: "
          f"{100 * k_row['captured_trace']:.1f}% trace, {100 * k_row['rel_max_err']:.2f}% of peak")

    # Hopkins and Gaussian simulators, dose-to-size calibrated on the same anchor
    hop = LithoSimulator(cfg)
    gcfg = copy.deepcopy(cfg)
    gcfg.optics.model = "gaussian"
    gau = LithoSimulator(gcfg)
    o, r = cfg.optics, cfg.resist
    write_csv(out.tab / "T1_conditions.csv", [{"parameter": a, "value": b} for a, b in [
        ("wavelength", f"{o.wavelength_nm:g} nm (ArF)"), ("NA / immersion index", f"{o.NA} / {o.immersion_index}"),
        ("illumination", f"{o.illumination}, sigma {o.sigma_in}/{o.sigma_out}" if o.illumination != "conventional"
         else f"conventional, sigma {o.sigma}"),
        ("imaging model", f"scalar Hopkins, SOCS K={k_row['K']} ({100 * k_row['captured_trace']:.1f}% of TCC trace)"),
        ("mask", "thin binary (Kirchhoff), dark field: polygon = clear"),
        ("pixel", f"{cfg.grid.pixel_nm:g} nm, exact area-coverage rasterization"),
        ("resist", f"constant threshold {hop.threshold:.4f} (dose-to-size: {r.anchor_cd_nm:g} nm line @ {r.anchor_pitch_nm:g} nm pitch)"),
        ("Gaussian model", f"sigma {gaussian_sigma(gcfg.optics):.1f} nm, threshold {gau.threshold:.4f}"),
        ("process window", f"dose {cfg.process.doses}, defocus {cfg.process.defocus_nm} nm"),
        ("fragmentation", f"corner {cfg.fragment.corner_len_nm:g} nm, segment {cfg.fragment.segment_len_nm:g} nm, "
                          f"line end <= {cfg.fragment.line_end_max_nm:g} nm"),
        ("MB-OPC", f"{cfg.mbopc.iterations} iterations, damping {cfg.mbopc.damping}, "
                   f"move [-{cfg.mbopc.max_in_nm:g}, +{cfg.mbopc.max_out_nm:g}] nm"),
        ("MRC", f"min width {cfg.mrc.min_width_nm:g} nm, min space {cfg.mrc.min_space_nm:g} nm (wafer scale)"),
    ]])

    shape = (512, 512)
    k = hop.kernels(shape, cfg.grid.pixel_nm).info
    viz.fig_optics(source_bitmap(o), k.eigenvalues, len(k.weights), spatial_kernels(k, shape, cfg.grid.pixel_nm, 500),
                   cfg.grid.pixel_nm, title=f"ArF λ={o.wavelength_nm:g} nm, NA={o.NA}, {o.illumination} "
                                            f"σ={o.sigma_in}/{o.sigma_out}", path=out.fig / "Fig01_optics.png")

    # iso/dense bias: through-pitch CD + cutlines
    def sweep(sim):
        p = np.array([x for x in pc.through_pitch_nm if x >= 2 * pc.cd_nm] + [pc.iso_pitch_nm], dtype=float)
        return p, np.array([periodic_cd(sim, pc.cd_nm, x) for x in p])

    p_h, cd_h = sweep(hop)
    p_g, cd_g = sweep(gau)
    write_csv(out.tab / "T4_through_pitch.csv", [
        {"pitch_nm": f"{p:g}" + (" (iso)" if p == pc.iso_pitch_nm else ""), "cd_hopkins_nm": round(a, 3),
         "cd_gaussian_nm": round(b, 3)} for p, a, b in zip(p_h, cd_h, cd_g)])
    profiles = {}
    for label, pitch in ((f"dense p{p_h[0]:g}", p_h[0]), (f"anchor p{r.anchor_pitch_nm:g}", r.anchor_pitch_nm),
                         ("isolated", pc.iso_pitch_nm)):
        spl, xc, _ = periodic_profile(hop, pc.cd_nm, pitch)
        x = np.linspace(-180, 180, 721)
        profiles[label] = (x, spl(xc + x))
    viz.fig_iso_dense({"Hopkins (SOCS)": (p_h, cd_h), "Gaussian": (p_g, cd_g)}, pc.cd_nm, r.anchor_pitch_nm,
                      {"Hopkins (SOCS)": cd_h[-1], "Gaussian": cd_g[-1]}, profiles, hop.threshold,
                      path=out.fig / "Fig02_iso_dense_bias.png")
    print(f"  iso/dense: CD {cd_h[0]:.1f} (pitch {p_h[0]:g}) -> {cd_h[-1]:.1f} nm (iso), threshold {hop.threshold:.4f}")

    # standard patterns, both models
    pats = standard_patterns(pc)
    res, gauge_rows = {}, []
    for model, sim in (("hopkins", hop), ("gaussian", gau)):
        for p in pats:
            s = sim.simulate(p.polygons)
            smp = ImageSampler(s.latent, s.grid)
            samples = sample_edges(p.polygons, 4.0)
            epe, _ = measure_epe(samples, smp, s.level, s.contours)
            gauges = {}
            for g in p.gauges:
                val, ok = measure_gauge(g, smp, s.level)
                err = val - g.target if g.kind == "cd" else val
                gauges[g.name] = (val, err)
                gauge_rows.append({"model": model, "pattern": p.name, "gauge": g.name, "kind": g.kind,
                                   "target_nm": g.target if g.kind == "cd" else 0.0, "measured_nm": round(val, 3),
                                   "error_nm": round(err, 3), "found": int(ok)})
            res[model, p.name] = dict(pattern=p, sim=s, samples=samples, epe=epe, gauges=gauges)
    write_csv(out.tab / "T3_noopc_gauges.csv", gauge_rows)
    H = {n: res["hopkins", n] for n in (p.name for p in pats)}

    viz.fig_gallery([{"name": p.name, "target": p.polygons, "printed": H[p.name]["sim"].contours,
                      "note": "\n".join(f"{k}: {H[p.name]['gauges'][k][1]:+.1f} nm" for k in _gallery_gauges(p)
                                        if k in H[p.name]["gauges"])} for p in pats],
                    path=out.fig / "Fig03_noopc_contours.png",
                    suptitle="Printed contour without OPC (Hopkins SOCS, constant-threshold resist)")

    def ends(name: str, gname: str):           # (target point, printed point, value, error) of a gauge
        gg = next(x for x in H[name]["pattern"].gauges if x.name == gname)
        val, err = H[name]["gauges"][gname]
        p, d = np.asarray(gg.point, float), np.asarray(gg.direction, float) / np.linalg.norm(gg.direction)
        return (p - val / 2 * d, p + val / 2 * d, val, err) if gg.kind == "cd" else (p, p + val * d, val, err)

    dense = f"dense_p{pc.dense_pitches_nm[0]:g}"
    box = (-150.0, -150.0, 150.0, 150.0)
    a0, a1, cd_d, e_d = ends(dense, "cd_center")
    b0, b1, cd_i, e_i = ends("iso_line", "cd_center")
    t0, t1, _, e_t = ends("line_end", "tip_upper")
    u0, u1, _, e_u = ends("line_end", "tip_lower")
    o0, o1, _, e_o = ends("l_shape", "outer_corner")
    i0, i1, _, e_in = ends("l_shape", "inner_corner")
    viz.fig_zoom([
        {"title": f"(a) dense L/S, pitch {pc.dense_pitches_nm[0]:g}", "target": H[dense]["pattern"].polygons,
         "printed": H[dense]["sim"].contours, "window": box, "dims": [(a0, a1, f"CD {cd_d:.1f} nm ({e_d:+.1f})", (0, 55))]},
        {"title": "(b) isolated line", "target": H["iso_line"]["pattern"].polygons, "printed": H["iso_line"]["sim"].contours,
         "window": box, "dims": [(b0, b1, f"CD {cd_i:.1f} nm ({e_i:+.1f})", (0, 55))]},
        {"title": f"(c) line ends, gap {pc.line_end_gap_nm:g}", "target": H["line_end"]["pattern"].polygons,
         "printed": H["line_end"]["sim"].contours, "window": box,
         "dims": [(t0, t1, f"pull-back {-e_t:.1f} nm", (62, 12)), (u0, u1, f"pull-back {-e_u:.1f} nm", (62, -12))]},
        {"title": "(d) L-shape corners", "target": H["l_shape"]["pattern"].polygons, "printed": H["l_shape"]["sim"].contours,
         "window": (o0[0] - 60, o0[1] - 60, o0[0] + 240, o0[1] + 240),
         "dims": [(o0, o1, f"outer {e_o:+.1f} nm", (-5, -38)), (i0, i1, f"inner {e_in:+.1f} nm", (40, 38))]},
    ], path=out.fig / "Fig04_proximity_closeups.png")

    ca = f"contact_array_p{pc.contact_pitch_nm:g}"
    viz.fig_epe_maps([{"name": n, "target": H[n]["pattern"].polygons, "printed": H[n]["sim"].contours,
                       "points": H[n]["samples"].points, "epe": H[n]["epe"]}
                      for n in ["iso_line", dense, "line_end", "l_shape", "t_shape", "contact_iso", ca, "demo_clip"]],
                     path=out.fig / "Fig05_noopc_epe_map.png", vmax=30)
    viz.fig_aerial([{"name": n, "grid": H[n]["sim"].grid, "image": H[n]["sim"].aerial, "level": H[n]["sim"].level,
                     "target": H[n]["pattern"].polygons} for n in [dense, "l_shape", ca]],
                   path=out.fig / "FigS3_aerial_images.png")
    viz.fig_gallery([{"name": p.name, "target": p.polygons, "printed": res["gaussian", p.name]["sim"].contours}
                     for p in pats], path=out.fig / "FigS4_gaussian_model.png",
                    suptitle="Printed contour without OPC (Gaussian sanity-check model)")

    write_gds_grid(out.gds / "test_patterns.gds", [("patterns", [(p.name, {LAYER_TARGET: p.polygons}) for p in pats])])
    write_klayout_lyp(out.gds / "layers.lyp")


# ============================================================================================== stages 2-4
class Study:
    """Standard patterns x correction methods, each computed once and shared by stages 2-4."""

    def __init__(self, cfg: Config, out: Paths):
        self.cfg, self.out = cfg, out
        self.sim = LithoSimulator(cfg)
        self.runner = Runner(self.sim, cfg)
        self.patterns = {p.name: p for p in standard_patterns(cfg.patterns)}
        self.cases = [self.runner.case(n, p.polygons) for n, p in self.patterns.items()]
        self.res: dict[str, dict[str, MethodResult]] = {}

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cases]

    def case(self, name: str):
        return next(c for c in self.cases if c.name == name)

    def ensure(self, methods: list[str]) -> None:
        for m in methods:
            if m in self.res:
                continue
            t0 = time.time()
            self.res[m] = {c.name: self.runner.run(c, m) for c in self.cases}
            e = np.mean([r.evaluation.stats["mean_abs"] for r in self.res[m].values()])
            print(f"  {m:20s} mean|EPE| {e:6.3f} nm   ({time.time() - t0:5.1f} s)", flush=True)

    def save(self) -> None:
        """T6 / T7 / raw tables and gds/results.gds for every method computed so far."""
        methods = [m for m in METHODS if m in self.res]
        rows = [result_row(c, self.res[m][c.name]) for c in self.cases for m in methods]
        write_csv(self.out.tab / "T6_metrics.csv", rows)
        write_csv(self.out.tab / "T7_method_summary.csv", summarize(rows, methods))
        write_csv(self.out.tab / "raw_site_epe.csv", [
            {"pattern": c.name, "segment": i, "kind": KIND_NAMES[c.frag.kind[i]],
             "site_x": round(c.frag.sites[i, 0], 1), "site_y": round(c.frag.sites[i, 1], 1),
             **{f"epe_{m}": round(self.res[m][c.name].evaluation.epe_sites[i], 3) for m in methods}}
            for c in self.cases for i in range(len(c.frag))])
        hist = [{"method": m, "pattern": n, **h} for m in methods if m in MB_VARIANTS
                for n in self.names for h in self.res[m][n].history]
        if hist:
            write_csv(self.out.tab / "raw_convergence.csv", hist)
        before = self.res["no_opc"]
        write_gds_grid(self.out.gds / "results.gds", [
            (m, [(n, {LAYER_TARGET: self.case(n).target, LAYER_MASK: self.res[m][n].mask,
                      LAYER_PRINT_BEFORE: printed_polygons(before[n]),
                      LAYER_PRINT_AFTER: printed_polygons(self.res[m][n])}) for n in self.names])
            for m in methods if m != "no_opc"])
        write_klayout_lyp(self.out.gds / "layers.lyp")


def summarize(rows: list[dict], methods: list[str]) -> list[dict]:
    """Per-method means over patterns (T7)."""
    out = []
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        it = [r["iterations_to_converge"] for r in rs if r["iterations_to_converge"] != ""]
        r3 = [r["rms_at_iter3"] for r in rs if r["rms_at_iter3"] != ""]
        d = {"method": m, "label": viz.method_label(m),
             "epe_mean_abs": np.mean([r["epe_mean_abs"] for r in rs]), "epe_max_abs": np.max([r["epe_max_abs"] for r in rs]),
             "pvb_mean": np.mean([r["pvb_mean"] for r in rs]), "worst_epe_pw": np.mean([r["worst_epe_pw"] for r in rs]),
             "converged": f"{len(it)}/{len(rs)}" if m in MB_VARIANTS else "",
             "iterations_mean": np.mean(it) if it else "", "rms_at_iter3": np.mean(r3) if r3 else "",
             "runtime_s": np.mean([r["runtime_s"] for r in rs]), "mask_vertices": np.mean([r["mask_vertices"] for r in rs]),
             "mrc_violations": int(sum(r["mrc_width"] + r["mrc_space"] for r in rs))}
        out.append({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()})
    return out


def _masks_item(st: Study, name: str, methods: list[str]) -> dict:
    return {"name": name, "target": st.case(name).target,
            **{m: {"mask": st.res[m][name].mask, "printed": st.res[m][name].evaluation.contours,
                   "epe_mean": st.res[m][name].evaluation.stats["mean_abs"],
                   "epe_max": st.res[m][name].evaluation.stats["max_abs"]} for m in methods}}


def stage2(st: Study) -> None:
    print("[stage 2] rule-based OPC")
    st.ensure(["no_opc", "rule"])
    deck: RuleBasedOPC = st.runner.rule
    rows = [{"width_nm": w, "space_nm": s, "bias_nm": round(b, 3)} for w, t in sorted(deck.deck.tables.items())
            for s, b in zip(t.spaces, t.bias)]
    rows += [{"width_nm": w, "space_nm": "line-end extension", "bias_nm": round(e, 3)}
             for w, e in sorted(deck.deck.line_end_ext.items())]
    write_csv(st.out.tab / "T5_rule_deck.csv", rows)
    viz.fig_bias_table(deck.deck.tables, deck.deck.line_end_ext, path=st.out.fig / "Fig06_rule_bias_table.png")
    n0, ru = st.res["no_opc"], st.res["rule"]
    viz.fig_gallery([{"name": n, "target": st.case(n).target, "mask": ru[n].mask, "printed": n0[n].evaluation.contours,
                      "printed_after": ru[n].evaluation.contours,
                      "note": f"mean|EPE| {n0[n].evaluation.stats['mean_abs']:.2f} → {ru[n].evaluation.stats['mean_abs']:.2f} nm"}
                     for n in st.names], path=st.out.fig / "FigS2_rule_all_patterns.png",
                    suptitle="Rule-based OPC (bias table + serif + hammerhead)")


def stage3(st: Study) -> None:
    print("[stage 3] model-based OPC")
    st.ensure(["no_opc", "rule", "mb"])
    cfg, mb, n0 = st.cfg, st.res["mb"], st.res["no_opc"]
    viz.fig_fragmentation(st.case("l_shape").target, st.case("l_shape").frag, mb["l_shape"].offsets,
                          path=st.out.fig / "Fig07_fragmentation.png")
    show = ["iso_line", f"dense_p{cfg.patterns.dense_pitches_nm[0]:g}", "line_end", "t_shape",
            f"contact_array_p{cfg.patterns.contact_pitch_nm:g}"]
    viz.fig_method_masks([_masks_item(st, n, ["no_opc", "rule", "mb"]) for n in show], ["no_opc", "rule", "mb"],
                         path=st.out.fig / "Fig08_opc_comparison.png")
    viz.fig_gallery([{"name": n, "target": st.case(n).target, "mask": mb[n].mask, "printed": n0[n].evaluation.contours,
                      "printed_after": mb[n].evaluation.contours,
                      "note": "mean|EPE| {:.2f} → {:.2f} nm, max {:.1f} → {:.1f} nm".format(
                          n0[n].evaluation.stats["mean_abs"], mb[n].evaluation.stats["mean_abs"],
                          n0[n].evaluation.stats["max_abs"], mb[n].evaluation.stats["max_abs"])} for n in st.names],
                    path=st.out.fig / "Fig09_mbopc_all_patterns.png",
                    suptitle=f"Model-based OPC (constant damping {cfg.mbopc.damping}, {cfg.mbopc.iterations} iterations)")


def stage4(st: Study) -> None:
    print("[stage 4] candidate contributions + ILT")
    st.ensure(METHODS)
    cfg, R, names = st.cfg, st.res, st.names
    it = np.array([[np.nan if R[m][n].converged_at is None else R[m][n].converged_at for m in MB] for n in names], float)
    viz.fig_convergence_summary({m: [R[m][n].history for n in names] for m in MB + ["mb_pw"]}, it, names, MB,
                                tol=cfg.mbopc.tol_rms_nm, path=st.out.fig / "Fig10_convergence.png")
    viz.fig_convergence({n: {m: R[m][n].history for m in MB} for n in names}, "rms_free", cfg.mbopc.tol_rms_nm,
                        path=st.out.fig / "FigS1_convergence_per_pattern.png", ylabel="rms EPE [nm]",
                        title="Convergence per pattern (sites not pinned by move/MRC limits)")

    pats, pc = st.patterns, cfg.patterns

    def zoom(name: str, gauge: str, shift=(0.0, 0.0), half: float = 90.0):
        x, y = next(g.point for g in pats[name].gauges if g.name == gauge)
        return (x + shift[0] - half, y + shift[1] - half, x + shift[0] + half, y + shift[1] + half)

    ca = f"contact_array_p{pc.contact_pitch_nm:g}"
    windows = {"iso_line": zoom("iso_line", "line_end", (0, -30)),
               "line_end": zoom("line_end", "tip_upper", (0, -pc.line_end_gap_nm / 2)),
               "l_shape": zoom("l_shape", "outer_corner", (60, 60)),
               "t_shape": zoom("t_shape", "inner_corner_left", (pc.cd_nm / 2, 0)), ca: zoom(ca, "cd_x_center")}
    pv_methods = ["no_opc", "mb", "mb_pw", "ilt"]
    viz.fig_pvband([{"name": n, "target": st.case(n).target, "window": w,
                     **{m: {"mask": R[m][n].mask, "outer": R[m][n].evaluation.pv_outer, "inner": R[m][n].evaluation.pv_inner,
                            "pvb": R[m][n].evaluation.pvb_mean} for m in pv_methods}} for n, w in windows.items()],
                   pv_methods, path=st.out.fig / "Fig11_pv_band.png",
                   title=f"PV band (dose {min(cfg.process.doses):g}–{max(cfg.process.doses):g}, "
                         f"defocus ±{max(abs(f) for f in cfg.process.defocus_nm):g} nm)")

    def mat(key):
        return np.array([[R[m][n].evaluation.row()[key] for m in METHODS] for n in names], float)

    viz.fig_metric_heatmaps([
        {"matrix": mat("epe_mean_abs"), "title": "(a) mean |EPE| at control sites, nominal", "cbar": "nm", "log": True},
        {"matrix": mat("pvb_mean"), "title": "(b) PV band width at control sites", "cbar": "nm"},
        {"matrix": mat("worst_epe_pw"), "title": "(c) worst |EPE| over the process window", "cbar": "nm"},
    ], names, METHODS, path=st.out.fig / "Fig12_metrics_heatmap.png")
    viz.fig_method_masks([_masks_item(st, n, pv_methods) for n in ["line_end", "l_shape", "t_shape", ca]], pv_methods,
                         path=st.out.fig / "Fig13_masks_mb_vs_ilt.png")


# =================================================================================================== stage 5
def _read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _md_table(rows: list[dict], cols: list[tuple[str, str]], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    out = "| " + " | ".join(h for _, h in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, "")
            if v in ("", None) or str(v).lower() == "nan":
                cells.append("–")
            elif k in fmt:
                try:
                    cells.append(fmt[k].format(float(v)))
                except ValueError:
                    cells.append(str(v))
            else:
                cells.append(str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out


def stage5(cfg: Config, out: Paths) -> None:
    """outputs/summary.md: every number comes from outputs/tables, every figure is embedded."""
    print("[stage 5] summary.md")
    T = {p.stem.split("_")[0]: _read(p) for p in sorted(out.tab.glob("T*.csv"))}
    missing = [t for t in ("T1", "T2", "T3", "T4", "T5", "T6", "T7") if t not in T]
    if missing or {r["method"] for r in T["T7"]} != set(METHODS):
        raise SystemExit(f"stage 5 needs the tables of stages 1-4 ({missing or 'T7 incomplete'}); "
                         "run `python run.py stage 1 2 3 4 5`")
    fig = lambda name, cap: f"![{cap}](figures/{name})\n*{name.split('_')[0]} — {cap}*\n"
    g = {(r["pattern"], r["gauge"]): float(r["error_nm"]) for r in T["T3"] if r["model"] == "hopkins"}
    tp = T["T4"]
    socs = T["T2"]
    k_row = next(r for r in socs if int(r["K"]) == cfg.optics.num_kernels)
    S = {r["method"]: r for r in T["T7"]}
    f = lambda m, k: float(S[m][k])
    m6 = T["T6"]
    per = [{"pattern": n, **{f"{m}_{k}": next(r[f"epe_{k}_abs"] for r in m6 if r["pattern"] == n and r["method"] == m)
                             for m in ("no_opc", "rule", "mb") for k in ("mean", "max")},
            "mb_iter": next(r["iterations_to_converge"] for r in m6 if r["pattern"] == n and r["method"] == "mb")}
           for n in dict.fromkeys(r["pattern"] for r in m6)]
    base_it, ad_it = f("mb", "iterations_mean"), f("mb_adaptive", "iterations_mean")

    md = f"""# OPC 결과 요약

`python run.py stage 1 2 3 4 5`로 다시 돌리면 이 문서와 `figures/`, `tables/`, `gds/`가 전부 갱신돼.
숫자는 모두 `tables/` CSV에서 읽어서 채운 값이야.

## 0. 한눈에 보기

| | no OPC | rule-based | MB-OPC | MB-OPC + adaptive damping |
|---|---|---|---|---|
| mean \\|EPE\\| (nm) | {f('no_opc', 'epe_mean_abs'):.2f} | {f('rule', 'epe_mean_abs'):.2f} | {f('mb', 'epe_mean_abs'):.3f} | {f('mb_adaptive', 'epe_mean_abs'):.3f} |
| PV band (nm) | {f('no_opc', 'pvb_mean'):.2f} | {f('rule', 'pvb_mean'):.2f} | {f('mb', 'pvb_mean'):.2f} | {f('mb_adaptive', 'pvb_mean'):.2f} |
| 수렴 iteration | – | – | {base_it:.1f} | {ad_it:.1f} |

{fig("Fig08_opc_comparison.png", "no OPC vs rule-based vs model-based OPC")}
## 1. 실험 조건 (Table 1)

{_md_table(T["T1"], [("parameter", "parameter"), ("value", "value")])}
## 2. Stage 1 — simulator 검증과 보정 전 proximity effect

- SOCS 전체 분해 vs brute-force Abbe: max error {float(socs[-1]['max_abs_err']):.1e} → TCC/SOCS 구현이 정확함
- K={k_row['K']}: TCC trace {100 * float(k_row['captured_trace']):.1f}%, aerial image 오차 peak 대비 {100 * float(k_row['rel_max_err']):.2f}% (Table 2)
- iso/dense bias: pitch {tp[0]['pitch_nm']} nm에서 CD {float(tp[0]['cd_hopkins_nm']):.1f} nm → iso {float(tp[-1]['cd_hopkins_nm']):.1f} nm (Gaussian model은 {float(tp[0]['cd_gaussian_nm']):.1f} → {float(tp[-1]['cd_gaussian_nm']):.1f} nm: 장거리 효과 없음)
- line-end pull-back (iso line) {-g[('iso_line', 'line_end')]:.1f} nm, tip-to-tip gap {cfg.patterns.line_end_gap_nm:g} nm는 tip마다 {-g[('line_end', 'tip_upper')]:.1f} nm
- corner rounding: L-shape outer {g[('l_shape', 'outer_corner')]:+.1f} nm, inner {g[('l_shape', 'inner_corner')]:+.1f} nm; contact {cfg.patterns.contact_nm:g} nm → CD {g[('contact_iso', 'cd_x')]:+.1f} nm

{fig("Fig01_optics.png", "source, TCC eigenvalues, leading SOCS kernels")}
{fig("Fig02_iso_dense_bias.png", "iso/dense bias: through-pitch CD and intensity cutlines")}
{fig("Fig03_noopc_contours.png", "printed contour without OPC, all test patterns")}
{fig("Fig04_proximity_closeups.png", "iso/dense bias, line-end shortening, corner rounding")}
{fig("Fig05_noopc_epe_map.png", "EPE along target edges without OPC")}
## 3. Stage 2 — rule-based OPC

1D 시뮬레이션으로 만든 space별 bias table + corner serif + calibrated line-end hammerhead (Table 5).
mean |EPE| {f('no_opc', 'epe_mean_abs'):.2f} → {f('rule', 'epe_mean_abs'):.2f} nm. 2D pattern(contact array, demo clip)에서 한계가 커.

{fig("Fig06_rule_bias_table.png", "rule deck: space-dependent edge bias")}
## 4. Stage 3 — model-based OPC

edge fragmentation → control site EPE → `d ← clip(d − λ·EPE)` 반복. 결과는 항상 Manhattan polygon이고,
hammerhead / serif / inner-corner notch가 자동으로 생겨. corner segment의 site는 vertex에서 {cfg.fragment.corner_len_nm:g} nm
떨어진 곳 (그 안쪽은 corner rounding 허용).

{_md_table(per, [("pattern", "pattern"), ("no_opc_mean", "no OPC mean"), ("no_opc_max", "max"), ("rule_mean", "rule mean"),
                 ("rule_max", "max"), ("mb_mean", "MB mean"), ("mb_max", "max"), ("mb_iter", "MB iter.")],
           {k: "{:.2f}" for k in ("no_opc_mean", "no_opc_max", "rule_mean", "rule_max", "mb_mean", "mb_max")})}
{fig("Fig07_fragmentation.png", "fragmentation, control sites and corrected Manhattan mask")}
{fig("Fig09_mbopc_all_patterns.png", "model-based OPC, all patterns")}
## 5. Stage 4 — contribution 후보 비교

{_md_table(T["T7"], [("label", "method"), ("epe_mean_abs", "mean abs EPE"), ("epe_max_abs", "max abs EPE"),
                     ("pvb_mean", "PV band"), ("worst_epe_pw", "worst EPE over PW"), ("converged", "converged"),
                     ("iterations_mean", "iter."), ("rms_at_iter3", "rms@3"), ("runtime_s", "time [s]"),
                     ("mask_vertices", "vertices"), ("mrc_violations", "MRC viol.")],
           {"epe_mean_abs": "{:.3f}", "epe_max_abs": "{:.2f}", "pvb_mean": "{:.2f}", "worst_epe_pw": "{:.2f}",
            "iterations_mean": "{:.1f}", "rms_at_iter3": "{:.3f}", "runtime_s": "{:.1f}", "mask_vertices": "{:.0f}"})}
(패턴 {len(per)}개 평균, max abs EPE만 패턴 중 최대값. PV band = control site별 band 폭의 평균.)

- **A. adaptive damping** (추천): 수렴 iteration {base_it:.1f} → {ad_it:.1f} ({100 * (1 - ad_it / base_it):.0f}% 감소), mean |EPE| {f('mb', 'epe_mean_abs'):.3f} → {f('mb_adaptive', 'epe_mean_abs'):.3f} nm, 추가 비용 없음.
- **B. rule-based init**: 초반 오차는 작지만 (rms@3 {f('mb', 'rms_at_iter3'):.3f} → {f('mb_hybrid', 'rms_at_iter3'):.3f} nm) 수렴 iteration은 {f('mb_hybrid', 'iterations_mean'):.1f}로 거의 같음. 수렴 꼬리는 MEEF가 낮은 corner/line-end site가 결정.
- **C. PW-aware**: worst-case EPE over PW {f('mb', 'worst_epe_pw'):.2f} → {f('mb_pw', 'worst_epe_pw'):.2f} nm, PV band {f('mb', 'pvb_mean'):.2f} → {f('mb_pw', 'pvb_mean'):.2f} nm. 대신 nominal mean |EPE|가 {f('mb_pw', 'epe_mean_abs'):.2f} nm로 증가 (trade-off).
- **ILT** (비교용): PV band 최소 ({f('ilt', 'pvb_mean'):.2f} nm, SRAF 형태의 assist feature 자동 생성). 하지만 EPE {f('ilt', 'epe_mean_abs'):.2f} nm, mask vertex {f('ilt', 'mask_vertices'):.0f}개, MRC 위반 {S['ilt']['mrc_violations']}개.

{fig("Fig10_convergence.png", "convergence of MB-OPC variants and iterations to converge")}
{fig("Fig11_pv_band.png", "PV bands: no OPC, MB-OPC, PW-aware MB-OPC, ILT")}
{fig("Fig12_metrics_heatmap.png", "mean EPE, PV band and worst-case EPE per pattern and method")}
{fig("Fig13_masks_mb_vs_ilt.png", "mask shapes: MB-OPC vs PW-aware vs ILT")}
## 6. 한계 (Discussion)

- thin-mask (Kirchhoff) + scalar imaging: mask 3D effect, polarization (NA 1.35에서 중요) 미반영.
- constant threshold resist: acid diffusion / development 미반영 (`resist.diffusion_nm` 옵션은 있음, 기본 0).
- aberration 없는 scalar pupil이라 ±defocus Bossung이 대칭.
- segment 이동만으로는 image slope(NILS)를 크게 못 바꿔서 PV band 감소 폭이 작음 → SRAF가 필요 (ILT 결과가 근거).

## 7. 부록

{fig("FigS1_convergence_per_pattern.png", "convergence per pattern")}
{fig("FigS2_rule_all_patterns.png", "rule-based OPC, all patterns")}
{fig("FigS3_aerial_images.png", "aerial images with the printed contour")}
{fig("FigS4_gaussian_model.png", "Gaussian sanity-check model, no OPC")}
### 파일

- `tables/T1_conditions` 조건 · `T2_socs_vs_abbe` · `T3_noopc_gauges` · `T4_through_pitch` · `T5_rule_deck`
  · `T6_metrics` (pattern × method 전체 지표) · `T7_method_summary` · `raw_site_epe` · `raw_convergence`
- `gds/results.gds` — 행 = method, 열 = pattern. layer 1 target / 2 mask / 3 보정 전 / 4 보정 후 / 10 label.
  KLayout에서 `gds/layers.lyp`를 같이 열면 색과 이름이 지정돼. 추천 method 행: `{RECOMMENDED}`.
- `gds/test_patterns.gds` — 표준 pattern (layer 1), `run.py correct`의 입력 예시.
"""
    (out.root / "summary.md").write_text(md)


# ===================================================================================================== run
def run_stages(cfg: Config, stages: list[int], out_dir: str | Path = "outputs/report") -> None:
    out = Paths(out_dir)
    t0 = time.time()
    if 1 in stages:
        stage1(cfg, out)
    if any(s in stages for s in (2, 3, 4)):
        st = Study(cfg, out)
        for s, fn in ((2, stage2), (3, stage3), (4, stage4)):
            if s in stages:
                fn(st)
        st.save()
    if 5 in stages:
        stage5(cfg, out)
    print(f"done in {time.time() - t0:.0f} s -> {out.root}/ (summary.md, figures/, tables/, gds/)")
