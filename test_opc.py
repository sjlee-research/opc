"""Tests (python -m pytest -q): imaging model, rasterization, layout I/O, fragmentation, MRC, OPC, gradients."""
import gdstk
import numpy as np
import pytest
import torch

from opc import Config, Grid, LithoSimulator
from opc.correction import CORNER_CONCAVE, LINE_END, RuleBasedOPC, fragment, mrc_check, run_mbopc
from opc.layout import (bitmap_to_polygons, iso_line, l_shape, rasterize, read_gds, read_png, rect, signed_area,
                        standard_patterns, write_gds)
from opc.litho import abbe_aerial, calibrate_threshold, periodic_cd


# ======================================== imaging, raster, I/O
def test_rasterize_exact_area_and_subpixel_edges():
    g = Grid(0.0, 0.0, 2.0, 64, 64)
    poly = np.array([[3.3, 4.1], [50.7, 4.1], [50.7, 20.0], [20.0, 20.0], [20.0, 60.9], [3.3, 60.9]])
    cov = rasterize([poly], g)
    assert cov.sum() * 4.0 == pytest.approx(signed_area(poly), rel=1e-12)
    # moving an edge by 0.3 nm changes the coverage of exactly one pixel column by 0.15
    cov2 = rasterize([rect(10.0, 10.0, 30.3, 30.0)], g) - rasterize([rect(10.0, 10.0, 30.0, 30.0)], g)
    assert np.unique(np.round(cov2[cov2 > 0], 9)).tolist() == [0.15]


def test_overlapping_polygons_are_united():
    g = Grid(0.0, 0.0, 2.0, 32, 32)
    cov = rasterize([rect(0, 0, 40, 20), rect(20, 0, 60, 20)], g)
    assert cov.max() <= 1.0 and cov.sum() * 4 == pytest.approx(60 * 20)


@pytest.mark.parametrize("illum", ["conventional", "annular", "quasar"])
def test_socs_matches_abbe(illum):
    cfg = Config()
    cfg.dtype = "float64"
    cfg.optics.illumination = illum
    cfg.optics.num_kernels = 100_000           # keep every eigenvector -> exact
    cfg.optics.source_grid = 21
    sim = LithoSimulator(cfg)
    g = Grid(-256, -256, 2.0, 256, 256)
    mask = rasterize([rect(-30, -150, 30, 150), rect(60, -100, 140, -20)], g)
    socs = sim.aerial(sim.to_tensor(mask), g.pixel).numpy()
    abbe = abbe_aerial(mask, cfg.optics, g.pixel)
    assert np.abs(socs - abbe).max() < 1e-10


def test_defocus_matches_abbe():
    cfg = Config()
    cfg.dtype = "float64"
    cfg.optics.num_kernels = 100_000
    cfg.optics.source_grid = 21
    sim = LithoSimulator(cfg)
    g = Grid(-256, -256, 2.0, 256, 256)
    mask = rasterize([rect(-30, -150, 30, 150)], g)
    socs = sim.aerial(sim.to_tensor(mask), g.pixel, defocus_nm=60).numpy()
    abbe = abbe_aerial(mask, cfg.optics, g.pixel, defocus_nm=60)
    assert np.abs(socs - abbe).max() < 1e-10


def test_clear_field_is_unity_and_symmetric():
    sim = LithoSimulator(Config())
    import torch
    I = sim.aerial(torch.ones(256, 256), 2.0)
    assert float(I.mean()) == pytest.approx(1.0, abs=2e-3)
    g = Grid(-256, -256, 2.0, 256, 256)
    img = sim.aerial(sim.to_tensor(rasterize([rect(-30, -100, 30, 100)], g)), 2.0).numpy()
    assert np.abs(img - img[:, ::-1]).max() < 1e-5 and np.abs(img - img[::-1, :]).max() < 1e-5


def test_threshold_calibration_prints_anchor_on_target():
    cfg = Config()
    sim = LithoSimulator(cfg)
    th = calibrate_threshold(sim, cfg.resist.anchor_cd_nm, cfg.resist.anchor_pitch_nm)
    assert periodic_cd(sim, cfg.resist.anchor_cd_nm, cfg.resist.anchor_pitch_nm, th) == pytest.approx(
        cfg.resist.anchor_cd_nm, abs=0.01)


def test_gds_roundtrip(tmp_path):
    pats = standard_patterns(Config().patterns)
    polys = [q for p in pats[:4] for q in p.polygons]
    write_gds(tmp_path / "t.gds", {1: polys, 3: polys[:2]})
    back = read_gds(tmp_path / "t.gds", 1)
    assert len(back) == len(polys)
    assert sum(signed_area(p) for p in back) == pytest.approx(sum(signed_area(p) for p in polys), rel=1e-9)


def test_png_input(tmp_path):
    from PIL import Image
    bm = np.zeros((40, 50), bool)
    bm[5:15, 5:40] = True
    bm[15:35, 5:12] = True
    Image.fromarray(np.where(bm, 0, 255).astype(np.uint8)).save(tmp_path / "x.png")
    polys = read_png(tmp_path / "x.png", 2.0)
    assert len(polys) == 1 and signed_area(polys[0]) == pytest.approx(bm.sum() * 4.0)
    assert sum(signed_area(p) for p in bitmap_to_polygons(bm, 2.0)) == pytest.approx(bm.sum() * 4.0)


# ======================================== fragmentation, MRC, OPC, gradients
@pytest.fixture(scope="module")
def setup():
    cfg = Config()
    return cfg, LithoSimulator(cfg)


@pytest.mark.parametrize("pat", standard_patterns(Config().patterns), ids=lambda p: p.name)
def test_mask_reconstruction(pat):
    frag = fragment(pat.polygons, Config().fragment)
    area = sum(signed_area(p) for p in pat.polygons)
    assert sum(signed_area(p) for p in frag.mask_polygons(np.zeros(len(frag)))) == pytest.approx(area)
    grown = gdstk.offset([gdstk.Polygon(p) for p in pat.polygons], 4.0, join="miter", use_union=True)
    assert sum(signed_area(p) for p in frag.mask_polygons(np.full(len(frag), 4.0))) == pytest.approx(
        sum(q.area() for q in grown))


def test_fragment_kinds_and_widths():
    cfg = Config()
    frag = fragment(l_shape(60, 500).polygons, cfg.fragment)
    assert (frag.kind == LINE_END).sum() == 2 and (frag.kind == CORNER_CONCAVE).sum() == 2
    side = frag.kind != LINE_END
    assert np.allclose(np.median(frag.width[side]), 60.0)


def test_mrc_check_detects_narrow_and_close_features():
    ok = [rect(0, 0, 40, 200), rect(70, 0, 110, 200)]
    assert mrc_check(ok, 20, 20)["n_width"] == 0 and mrc_check(ok, 20, 20)["n_space"] == 0
    narrow = [rect(0, 0, 15, 200)]
    close = [rect(0, 0, 40, 200), rect(50, 0, 90, 200)]
    assert mrc_check(narrow, 20, 20)["n_width"] == 1
    assert mrc_check(close, 20, 20)["n_space"] == 1


def test_rule_deck_bias_sign(setup):
    cfg, sim = setup
    rb = RuleBasedOPC(sim, cfg)
    rb.ensure_width(60.0)
    t = rb.deck.tables[60.0]
    assert t(60.0) < 0 < t(1000.0)                  # dense lines print wide, isolated lines narrow
    assert abs(t(cfg.resist.anchor_pitch_nm - 60.0)) < 0.05   # anchor needs no bias
    assert rb.deck.line_end_ext[60.0] > 0


def test_mbopc_converges_on_iso_line(setup):
    cfg, sim = setup
    pat = iso_line(60, 600)
    frag = fragment(pat.polygons, cfg.fragment)
    grid = Grid.around(pat.polygons, cfg.grid.pixel_nm, cfg.grid.margin_nm)
    r = run_mbopc(sim, cfg, frag, grid)
    assert r.history[0]["rms"] > 5.0
    assert r.converged_at is not None and np.abs(r.epe).mean() < 0.3
    assert mrc_check(r.mask, cfg.mrc.min_width_nm, cfg.mrc.min_space_nm)["n_width"] == 0


def test_aerial_gradient_matches_finite_difference(setup):
    cfg, _ = setup
    c = Config()
    c.dtype = "float64"
    sim = LithoSimulator(c)
    g = Grid(-128, -128, 2.0, 128, 128)
    m0 = sim.to_tensor(rasterize([rect(-30, -60, 30, 60)], g))
    w = torch.rand(128, 128, dtype=torch.float64)
    m = m0.clone().requires_grad_(True)
    (sim.aerial(m, 2.0) * w).sum().backward()
    i, j = 64, 79
    e = torch.zeros_like(m0)
    e[i, j] = 1e-6
    fd = ((sim.aerial(m0 + e, 2.0) * w).sum() - (sim.aerial(m0 - e, 2.0) * w).sum()) / 2e-6
    assert float(m.grad[i, j]) == pytest.approx(float(fd), rel=1e-5)
