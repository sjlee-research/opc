"""Report figures (matplotlib).

Color roles are fixed across every figure of the report:
    target            neutral gray fill
    printed (no OPC)  orange   — stage 3 will add "printed (OPC)" in blue and the mask in violet
    EPE               diverging blue (under-print) <-> gray <-> red (over-print)
    intensity         single-hue sequential blue
"""
from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon as MplPolygon

from .layout import Grid, Polygon, bbox

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TARGET_FILL = "#e4e3dd"
TARGET_EDGE = "#6f6e69"
C_PRINT_BEFORE = "#eb6834"      # categorical slot 2
C_PRINT_AFTER = "#2a78d6"       # categorical slot 1
C_MASK = "#4a3aa7"              # categorical slot 7
C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
C_CUTLINE = ["#1baf7a", "#4a3aa7", "#e34948"]   # dense / anchor / isolated (distinct from the model colors)

CMAP_INTENSITY = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
CMAP_EPE = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#a32828"])


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.linewidth": 0.8,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5,
    })


setup_style()


# ------------------------------------------------------------------ primitives
def draw_polygons(ax, polys: Sequence[Polygon], face="none", edge=INK, lw=1.0, alpha=1.0, zorder=1, label=None):
    for i, p in enumerate(polys):
        ax.add_patch(MplPolygon(np.asarray(p), closed=True, facecolor=face, edgecolor=edge, linewidth=lw,
                                alpha=alpha, zorder=zorder, label=label if i == 0 else None))


def draw_contours(ax, loops: Sequence[Polygon], color=C_PRINT_BEFORE, lw=1.6, zorder=3, label=None):
    # closed outline patches: no line caps, so the loop's start/end point leaves no visible blob
    for i, c in enumerate(loops):
        ax.add_patch(MplPolygon(np.asarray(c), closed=True, fill=False, edgecolor=color, linewidth=lw, zorder=zorder,
                                joinstyle="round", label=label if i == 0 else None))


def _nice_length(limit: float) -> float:
    return max([v for v in (5, 10, 20, 50, 100, 200, 500, 1000, 2000) if v <= limit] or [5])


def layout_axes(ax, polys: Sequence[Polygon], pad: float = 60.0, window=None, scalebar: bool | float = True,
                square: bool = True):
    """Frame a layout: square window around the shapes, scale bar in a strip below them."""
    xmin, ymin, xmax, ymax = window if window is not None else bbox(polys)
    xmin, ymin, xmax, ymax = xmin - pad, ymin - pad, xmax + pad, ymax + pad
    if square:
        side = max(xmax - xmin, ymax - ymin)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        xmin, xmax, ymin, ymax = cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2
    strip = 0.12 * (ymax - ymin) if scalebar else 0.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin - strip, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    if scalebar:
        w = xmax - xmin
        L = scalebar if not isinstance(scalebar, bool) else _nice_length(0.3 * w)
        xs, ys = xmax - 0.05 * w - L, ymin - 0.65 * strip
        ax.plot([xs, xs + L], [ys, ys], color=INK, lw=2, solid_capstyle="butt", zorder=5)
        ax.text(xs - 0.02 * w, ys, f"{L:g} nm", ha="right", va="center", fontsize=7, color=INK)


def plot_layout(ax, target: Sequence[Polygon], printed: Sequence[Polygon] = (), printed_after: Sequence[Polygon] = (),
                mask: Sequence[Polygon] = (), title: str = "", pad: float = 60.0, window=None, scalebar=True):
    draw_polygons(ax, target, face=TARGET_FILL, edge=TARGET_EDGE, lw=0.8, zorder=1, label="target")
    if mask:
        draw_polygons(ax, mask, face="none", edge=C_MASK, lw=1.0, zorder=2, label="OPC mask")
    if printed:
        draw_contours(ax, printed, C_PRINT_BEFORE, label="printed (no OPC)")
    if printed_after:
        draw_contours(ax, printed_after, C_PRINT_AFTER, label="printed (OPC)")
    layout_axes(ax, target, pad, window, scalebar)
    if title:
        ax.set_title(title, loc="left")


def legend_handles(*items):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = []
    for kind, label in items:
        if kind == "target":
            h.append(Patch(facecolor=TARGET_FILL, edgecolor=TARGET_EDGE, label=label))
        elif kind == "before":
            h.append(Line2D([], [], color=C_PRINT_BEFORE, lw=2, label=label))
        elif kind == "after":
            h.append(Line2D([], [], color=C_PRINT_AFTER, lw=2, label=label))
        elif kind == "mask":
            h.append(Line2D([], [], color=C_MASK, lw=1.2, label=label))
    return h


# --------------------------------------------------------------------- figures
def fig_gallery(items, ncols: int = 4, path=None, suptitle: str = ""):
    """items: list of dict(name, target, printed, [printed_after], [mask], [note])."""
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    has_notes = any(it.get("note") for it in items)
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(3.0 * ncols, 5.0), (3.75 if has_notes else 3.2) * nrows + 0.3))
    axes = np.atleast_1d(axes).ravel()
    for ax, it in zip(axes, items):
        plot_layout(ax, it["target"], it.get("printed", ()), it.get("printed_after", ()), it.get("mask", ()),
                    title=it["name"])
        if it.get("note"):
            ax.text(0.0, -0.03, it["note"], transform=ax.transAxes, fontsize=7.5, color=INK_2, va="top",
                    linespacing=1.35)
    for ax in axes[n:]:
        ax.axis("off")
    kinds = [("target", "target"), ("before", "printed contour (no OPC)")]
    if any(it.get("printed_after") for it in items):
        kinds.append(("after", "printed contour (OPC)"))
    if any(it.get("mask") for it in items):
        kinds.append(("mask", "OPC mask"))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5 if has_notes else 1.0)
    fig.legend(handles=legend_handles(*kinds), loc="upper center", ncol=len(kinds), bbox_to_anchor=(0.5, 0.985))
    if suptitle:
        fig.suptitle(suptitle, y=1.0, fontsize=11, fontweight="bold", color=INK)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def fig_epe_maps(items, path=None, vmax: float | None = None, ncols: int = 4):
    """items: list of dict(name, target, printed, points, epe)."""
    vmax = vmax or max(np.percentile(np.abs(it["epe"]), 99) for it in items)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    sc = None
    for ax, it in zip(axes, items):
        draw_polygons(ax, it["target"], face=TARGET_FILL, edge="none", zorder=1)
        draw_contours(ax, it["printed"], color=MUTED, lw=0.8, zorder=2)
        sc = ax.scatter(it["points"][:, 0], it["points"][:, 1], c=it["epe"], cmap=CMAP_EPE, norm=norm, s=7,
                        linewidths=0, zorder=3)
        layout_axes(ax, it["target"], 50)
        ax.set_title(it["name"], loc="left")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cax = fig.add_axes((0.935, 0.2, 0.012, 0.6))
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("EPE [nm]   (− under-print · + over-print)", color=INK_2)
    cb.outline.set_edgecolor(AXIS)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def _ax_through_pitch(ax, curves: dict[str, tuple[np.ndarray, np.ndarray]], target_cd: float, anchor_pitch: float,
                      iso_cd: dict[str, float] | None = None):
    """curves: label -> (pitch, printed CD)."""
    ax.grid(True, axis="y")
    ax.axhline(target_cd, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(0.01, target_cd - 0.3, f"target {target_cd:g} nm", color=INK_2, fontsize=8,
            ha="left", va="top", transform=ax.get_yaxis_transform())
    for i, (label, (p, cd)) in enumerate(curves.items()):
        ax.plot(p, cd, color=C_SERIES[i], lw=2, marker="o", ms=4.5, markeredgecolor="white", markeredgewidth=0.8,
                label=label, zorder=3)
        if iso_cd and label in iso_cd:
            ax.annotate(f"iso {iso_cd[label]:.1f}", (p[-1], cd[-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=INK_2, va="center")
    ax.axvline(anchor_pitch, color=AXIS, lw=0.8, zorder=0)
    ax.text(anchor_pitch, 0.02, " anchor", transform=ax.get_xaxis_transform(), fontsize=8, color=MUTED)
    ax.set_xscale("log")
    pmax = max(p.max() for p, _ in curves.values())
    ticks = [t for t in (120, 150, 200, 300, 400, 600, 1000, 2000) if t < 0.7 * pmax] + [pmax]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks[:-1]] + [f"{pmax:g}\n(iso)"])
    ax.minorticks_off()
    ax.set_xlabel("pitch [nm] (log scale)")
    ax.set_ylabel("printed CD [nm]")
    ax.set_title("(a) through-pitch printed CD, no OPC", loc="left")
    ax.legend(loc="upper right")
    _clean_axes(ax)


def _ax_cutlines(ax, profiles: dict[str, tuple[np.ndarray, np.ndarray]], threshold: float, cd: float):
    """profiles: label -> (x relative to line center [nm], intensity)."""
    ax.grid(True, axis="y")
    ax.axvspan(-cd / 2, cd / 2, color=TARGET_FILL, zorder=0, lw=0)
    ax.text(0, 0.98, "target line", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8,
            color=INK_2)
    for i, (label, (x, I)) in enumerate(profiles.items()):
        ax.plot(x, I, color=C_CUTLINE[i % 3], lw=2, label=label, zorder=3)
    ax.axhline(threshold, color=INK, lw=1.0, ls="--", zorder=2)
    ax.text(0.99, threshold, f"threshold {threshold:.3f}", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=8, color=INK)
    ax.set_xlabel("x [nm] (from line center)")
    ax.set_ylabel("latent image intensity")
    ax.set_title("(b) intensity cutline across the center line", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _clean_axes(ax)


def fig_iso_dense(curves, target_cd, anchor_pitch, iso_cd, profiles, threshold, path=None):
    """Iso/dense bias: (a) through-pitch CD for each model, (b) why — peak intensity drops toward isolated."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1, 1.15]})
    _ax_through_pitch(axes[0], curves, target_cd, anchor_pitch, iso_cd)
    _ax_cutlines(axes[1], profiles, threshold, target_cd)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def fig_optics(source: np.ndarray, eigenvalues: np.ndarray, K: int, kernels: np.ndarray, pixel: float,
               title: str = "", path=None):
    """Source map, TCC eigenvalue spectrum, and |phi_k| of the leading SOCS kernels."""
    fig = plt.figure(figsize=(12, 3.8))
    gs = fig.add_gridspec(2, 6, width_ratios=[1.2, 1.6, 1, 1, 1, 1], wspace=0.75, hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    h_px = 1.0 / (source.shape[0] - 1)
    ax.imshow(source, origin="lower", extent=(-1 - h_px, 1 + h_px, -1 - h_px, 1 + h_px), cmap=CMAP_INTENSITY,
              vmin=0, vmax=1.3)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=INK_2, lw=0.8, ls="--"))
    ax.set_title("(a) source", loc="left")
    ax.set_xlabel(r"$\sigma_x$")
    ax.set_ylabel(r"$\sigma_y$")
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[:, 1])
    ev = eigenvalues / eigenvalues[0]
    n = min(len(ev), 80)
    ax.semilogy(np.arange(1, n + 1), ev[:n], color=C_SERIES[0], marker="o", ms=3, lw=1.2)
    ax.axvline(K + 0.5, color=MUTED, lw=1, ls="--")
    cap = eigenvalues[:K].sum() / eigenvalues.sum()
    ax.text(K + 1.5, ev[0] * 0.5, f"K = {K}\n{100 * cap:.1f}% of trace", fontsize=8, color=INK_2, va="top")
    ax.set_xlabel("kernel index k")
    ax.set_ylabel(r"$\lambda_k / \lambda_1$")
    ax.set_title("(b) TCC eigenvalues", loc="left")
    ax.grid(True, axis="y", which="major")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    h = kernels.shape[-1] * pixel / 2
    for i in range(min(8, len(kernels))):
        ax = fig.add_subplot(gs[i // 4, 2 + i % 4])
        ax.imshow(np.abs(kernels[i]), origin="lower", extent=(-h, h, -h, h), cmap=CMAP_INTENSITY)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$|\phi_{{{i + 1}}}|$  {eigenvalues[i] / eigenvalues[0]:.3f}", fontsize=8,
                     fontweight="normal")
        if i == 0:
            ax.text(-h, h * 1.45, "(c) leading SOCS kernels (window %.0f nm)" % (2 * h), fontsize=10,
                    fontweight="bold", color=INK)
    if title:
        fig.suptitle(title, y=1.04, fontsize=11, fontweight="bold", color=INK)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def fig_zoom(panels, path=None):
    """Close-ups with dimension marks.

    panels: list of dict(title, target, printed, window, dims) where each dim is
    (p_from, p_to, text, (dx, dy)): a double arrow between two layout points
    (e.g. target edge -> printed edge) and a label offset in points from its midpoint.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.9))
    for ax, pn in zip(np.atleast_1d(axes), panels):
        plot_layout(ax, pn["target"], pn["printed"], title=pn["title"], pad=0, window=pn["window"])
        for p0, p1, text, off in pn.get("dims", []):
            p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
            ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0, shrinkA=0, shrinkB=0,
                                                    mutation_scale=8), zorder=6)
            ax.annotate(text, (p0 + p1) / 2, xytext=off, textcoords="offset points", fontsize=8, color=INK,
                        ha="center", va="center", zorder=7,
                        bbox=dict(facecolor="white", edgecolor=GRID, pad=2, lw=0.6),
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(handles=legend_handles(("target", "target"), ("before", "printed contour (no OPC)")),
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def fig_aerial(items, path=None, ncols: int = 3):
    """items: list of dict(name, grid, image, level, target, window)."""
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, it in zip(axes, items):
        g: Grid = it["grid"]
        im = ax.imshow(it["image"], origin="lower", extent=g.extent, cmap=CMAP_INTENSITY, vmin=0,
                       vmax=max(np.max(x["image"]) for x in items), interpolation="bilinear")
        draw_polygons(ax, it["target"], face="none", edge=INK, lw=0.8, zorder=2)
        ax.contour(g.x_centers(), g.y_centers(), it["image"], levels=[it["level"]], colors=[C_PRINT_BEFORE],
                   linewidths=1.6, zorder=3)
        layout_axes(ax, it["target"], 100, it.get("window"))
        ax.set_title(it["name"], loc="left")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cax = fig.add_axes((0.935, 0.2, 0.012, 0.6))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("aerial intensity (clear field = 1)", color=INK_2)
    cb.outline.set_edgecolor(AXIS)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


# ============================================================ OPC-stage figures
METHOD_STYLE = {   # fixed method -> (label, color); color follows the method in every chart
    "no_opc": ("no OPC", "#eb6834"),
    "rule": ("rule-based", "#1baf7a"),
    "mb": ("MB-OPC (baseline)", "#2a78d6"),
    "mb_adaptive": ("MB-OPC + adaptive damping", "#4a3aa7"),
    "mb_hybrid": ("MB-OPC + rule init", "#eda100"),
    "mb_adaptive_hybrid": ("MB-OPC + adaptive + rule init", "#e34948"),
    "mb_pw": ("MB-OPC + PW-aware", "#e87ba4"),
    "ilt": ("pixel ILT", "#008300"),
}


def method_label(m: str) -> str:
    return METHOD_STYLE.get(m, (m, INK))[0]


def method_color(m: str) -> str:
    return METHOD_STYLE.get(m, (m, INK))[1]


def _clean_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_bias_table(tables: dict, line_end_ext: dict, path=None):
    """tables: width -> BiasTable."""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.grid(True, axis="y")
    ax.axhline(0, color=AXIS, lw=0.8)
    for i, (w, t) in enumerate(sorted(tables.items())):
        ax.plot(t.spaces, t.bias, color=C_SERIES[i % 4], marker="o", ms=4, markeredgecolor="white",
                markeredgewidth=0.8, label=f"line width {w:g} nm  (line-end ext {line_end_ext.get(w, 0):.1f} nm)")
    ax.set_xscale("log")
    ticks = [60, 100, 200, 400, 1000]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("space to the neighbouring edge [nm] (log scale)")
    ax.set_ylabel("edge bias per side [nm]")
    ax.set_title("Rule deck: space-dependent edge bias (from 1D simulation)", loc="left")
    ax.legend(loc="lower right")
    _clean_axes(ax)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def fig_fragmentation(target, frag, offsets=None, window=None, path=None, title=""):
    """Segments coloured by kind, control sites, and (optionally) the corrected mask."""
    from .correction import CORNER_CONCAVE, CORNER_CONVEX, KIND_NAMES, LINE_END, NORMAL
    colors = {NORMAL: "#2a78d6", CORNER_CONVEX: "#eb6834", CORNER_CONCAVE: "#1baf7a", LINE_END: "#4a3aa7"}
    ncol = 2 if offsets is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 4.6))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    draw_polygons(ax, target, face=TARGET_FILL, edge="none")
    for i in range(len(frag)):
        ax.plot(*np.array([frag.a[i], frag.b[i]]).T, color=colors[frag.kind[i]], lw=2.4, solid_capstyle="butt")
        ax.plot(*np.array([frag.a[i]]).T, marker="|", color=INK, ms=6, mew=0.8)
    ax.scatter(frag.sites[:, 0], frag.sites[:, 1], s=10, color=INK, zorder=4, linewidths=0)
    layout_axes(ax, target, 40, window)
    ax.set_title("(a) fragmentation and control sites", loc="left")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=colors[k], lw=2.4, label=KIND_NAMES[k]) for k in (NORMAL, CORNER_CONVEX, CORNER_CONCAVE, LINE_END)]
    h.append(Line2D([], [], color=INK, marker="o", ls="none", ms=3, label="control site"))
    ax.legend(handles=h, loc="upper right", fontsize=7.5, frameon=True, facecolor="white", edgecolor=GRID)
    if offsets is not None:
        ax = axes[1]
        draw_polygons(ax, target, face=TARGET_FILL, edge=TARGET_EDGE, lw=0.8)
        draw_polygons(ax, frag.mask_polygons(offsets), face="none", edge=C_MASK, lw=1.3)
        layout_axes(ax, target, 40, window)
        ax.set_title("(b) corrected Manhattan mask", loc="left")
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def fig_convergence(curves: dict, key: str = "rms", tol: float | None = None, ncols: int = 5, path=None,
                    ylabel: str = "rms EPE [nm]", title: str = ""):
    """curves: pattern -> {method: history list}. Small multiples, one panel per pattern."""
    names = list(curves)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.3 * nrows + 0.6), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    methods = []
    for ax, name in zip(axes, names):
        ax.grid(True, axis="y", which="major")
        for m, hist in curves[name].items():
            it = [h["iteration"] for h in hist]
            v = np.maximum([h[key] for h in hist], 1e-3)
            ax.semilogy(it, v, color=method_color(m), lw=1.6)
            if m not in methods:
                methods.append(m)
        if tol:
            ax.axhline(tol, color=MUTED, lw=0.8, ls="--")
        ax.set_title(name, loc="left", fontsize=9)
        _clean_axes(ax)
    for ax in axes[len(names):]:
        ax.axis("off")
    for ax in axes[::ncols]:
        ax.set_ylabel(ylabel)
    for ax in axes[-ncols:]:
        ax.set_xlabel("iteration")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=method_color(m), lw=2, label=method_label(m)) for m in methods]
    if tol:
        h.append(Line2D([], [], color=MUTED, lw=0.8, ls="--", label=f"tolerance {tol:g} nm"))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(handles=h, loc="upper center", ncol=min(len(h), 4), bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, y=1.06, fontsize=11, fontweight="bold", color=INK)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def fig_convergence_summary(curves: dict, iterations: np.ndarray, rows: list[str], cols: list[str],
                            key: str = "rms_free", tol: float | None = None, path=None):
    """(a) geometric-mean rms EPE per iteration for each method, (b) iterations to converge per pattern.

    curves: method -> list of histories (one per pattern); iterations: rows (patterns) x cols (methods).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.grid(True, axis="y", which="major")
    for m, hists in curves.items():
        n = min(len(h) for h in hists)
        v = np.exp(np.mean([np.log(np.maximum([h[i][key] for i in range(n)], 1e-3)) for h in hists], axis=0))
        ax.semilogy(range(n), v, color=method_color(m), lw=2, label=method_label(m))
    if tol:
        ax.axhline(tol, color=MUTED, lw=0.8, ls="--")
        ax.text(0.99, tol, f"tolerance {tol:g} nm", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=8, color=INK_2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("rms EPE [nm], geometric mean over patterns")
    ax.set_title("(a) convergence of model-based OPC variants", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    _clean_axes(ax)
    _ax_heatmap(fig, axes[1], iterations, rows, cols, "(b) iterations to converge", "iterations", "{:.0f}")
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def fig_pvband(rows, methods, path=None, title=""):
    """rows: list of dict(name, target, per-method dict(mask, outer, inner, nominal, pvb))."""
    import gdstk
    from .layout import even_odd_fill
    nr, nc = len(rows), len(methods)
    fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc, 2.9 * nr + 0.5))
    axes = np.array(axes).reshape(nr, nc)
    for r, row in enumerate(rows):
        for c, m in enumerate(methods):
            ax = axes[r, c]
            d = row[m]
            draw_polygons(ax, row["target"], face=TARGET_FILL, edge="none", zorder=1)
            if d.get("mask") is not None and m != "no_opc":
                draw_polygons(ax, d["mask"], face="none", edge=C_MASK, lw=0.6, zorder=2, alpha=0.7)
            outer, inner = even_odd_fill(d["outer"]), even_odd_fill(d["inner"])
            band = gdstk.boolean([gdstk.Polygon(p) for p in outer], [gdstk.Polygon(p) for p in inner], "not")
            draw_polygons(ax, [b.points for b in band], face=method_color(m), edge="none", alpha=0.55, zorder=3)
            draw_polygons(ax, row["target"], face="none", edge=TARGET_EDGE, lw=0.7, zorder=4)
            layout_axes(ax, row["target"], 50, row.get("window"))
            if r == 0:
                ax.set_title(method_label(m), fontsize=9, color=INK)
            ax.text(0.03, 0.025, f"PV band {d['pvb']:.2f} nm", transform=ax.transAxes, fontsize=7.5, va="bottom",
                    color=INK_2)
        axes[r, 0].set_ylabel(row["name"], fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if title:
        fig.suptitle(title, y=0.995, fontsize=11, fontweight="bold", color=INK)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def _ax_heatmap(fig, ax, matrix: np.ndarray, rows: list[str], cols: list[str], title: str, cbar: str,
                fmt: str = "{:.2f}", log: bool = False):
    """Pattern x method table as a single-hue heatmap with the value printed in every cell."""
    from matplotlib.colors import LogNorm, Normalize
    vals = np.where(np.isfinite(matrix), matrix, np.nan)
    pos = vals[vals > 0]
    log = log and len(pos) > 0
    norm = LogNorm(max(pos.min(), 1e-2), pos.max()) if log else Normalize(np.nanmin(vals), np.nanmax(vals))
    im = ax.imshow(np.maximum(vals, 1e-2) if log else vals, cmap=CMAP_INTENSITY, norm=norm, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = vals[i, j]
            if np.isfinite(v):
                dark = norm(max(v, 1e-2)) > 0.55
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color="white" if dark else INK)
    ax.set_xticks(range(len(cols)), [method_label(c) for c in cols], rotation=25, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label(cbar, color=INK_2)
    cb.outline.set_edgecolor(AXIS)


def fig_metric_heatmaps(panels: list[dict], rows: list[str], cols: list[str], path=None):
    """Stacked pattern x method heatmaps. panels: dict(matrix, title, cbar, fmt, log)."""
    h = 0.42 * len(rows) + 1.0
    fig, axes = plt.subplots(len(panels), 1, figsize=(1.25 * len(cols) + 2.6, h * len(panels) + 1.2))
    for ax, pn in zip(np.atleast_1d(axes), panels):
        _ax_heatmap(fig, ax, pn["matrix"], rows, cols, pn["title"], pn["cbar"], pn.get("fmt", "{:.2f}"),
                    pn.get("log", False))
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def fig_method_masks(items, methods, path=None, title=""):
    """rows = patterns, cols = methods: target, mask, printed contour (before = orange thin, after = blue)."""
    nr, nc = len(items), len(methods)
    fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc, 2.9 * nr + 0.6))
    axes = np.array(axes).reshape(nr, nc)
    for r, it in enumerate(items):
        for c, m in enumerate(methods):
            ax = axes[r, c]
            d = it[m]
            draw_polygons(ax, it["target"], face=TARGET_FILL, edge=TARGET_EDGE, lw=0.6, zorder=1)
            if m != "no_opc":
                draw_polygons(ax, d["mask"], face=C_MASK, edge=C_MASK, lw=0.5, alpha=0.18, zorder=2)
                draw_contours(ax, it["no_opc"]["printed"], C_PRINT_BEFORE, lw=0.9, zorder=3)
            draw_contours(ax, d["printed"], C_PRINT_AFTER if m != "no_opc" else C_PRINT_BEFORE, lw=1.5, zorder=4)
            layout_axes(ax, it["target"], 50, it.get("window"))
            if r == 0:
                ax.set_title(method_label(m), fontsize=9, color=INK)
            ax.text(0.03, 0.025, f"mean / max |EPE|  {d['epe_mean']:.2f} / {d['epe_max']:.1f} nm", transform=ax.transAxes,
                    fontsize=7, va="bottom", color=INK_2)
        axes[r, 0].set_ylabel(it["name"], fontsize=9, color=INK)
    from matplotlib.patches import Patch
    h = legend_handles(("target", "target"), ("before", "printed (no OPC)"), ("after", "printed (corrected)"))
    h.append(Patch(facecolor=C_MASK, alpha=0.3, edgecolor=C_MASK, label="mask"))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, y=1.03, fontsize=11, fontweight="bold", color=INK)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


C_IMPROVED = "#006300"     # text color for "reduced by x %" (always written out, never color alone)


def _metrics_block(ax, m: dict, ref: dict | None = None, footer: str = "") -> None:
    """Aligned metric rows drawn into an empty axis; with ``ref`` (original-mask metrics) each row shows the change."""
    ax.axis("off")
    rows = [("EPE on edges, mean", m["epe_mean"], "nm", "epe_mean"),
            ("EPE on edges, max", m["epe_max"], "nm", "epe_max"),
            ("corner error", m["corner"], "nm", "corner"),
            ("area error", m["area_pct"], "%", "area_pct")]
    for i, (label, v, unit, key) in enumerate(rows):
        y = 0.95 - 0.19 * i
        ax.text(0.0, y, label, transform=ax.transAxes, va="top", fontsize=10, color=INK_2)
        ax.text(0.47, y, f"{v:.2f} {unit}", transform=ax.transAxes, va="top", fontsize=10, color=INK, fontweight="bold")
        if ref is not None and ref[key] > 0:
            red = 100 * (1 - v / ref[key])
            ax.text(0.72, y, f"{red:.0f}% lower" if red >= 0 else f"{-red:.0f}% higher", transform=ax.transAxes,
                    va="top", fontsize=10, color=C_IMPROVED if red >= 0 else C_PRINT_BEFORE)
    if footer:
        ax.text(0.0, 0.95 - 0.19 * len(rows), footer, transform=ax.transAxes, va="top", fontsize=9, color=MUTED)


def fig_before_after(items, path=None, title: str = ""):
    """One row per clip. Left: original mask (= target) and what it prints. Right: OPC mask and what it prints.

    The target is drawn the same way in both panels (gray fill underneath); both panels share the same window and
    scale, and each panel carries its own legend box (MATLAB-style). Under each panel the print fidelity is
    quantified (``correction.fidelity_metrics``); the OPC panel also shows the reduction against the original mask.
    items: list of dict(name, target, printed_before, mask_after, printed_after, metrics_before, metrics_after,
    footer_after); mask_after=None draws only the left panel (simulation without OPC).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    with_opc = any(it.get("mask_after") is not None for it in items)
    ncols, n = (2 if with_opc else 1), len(items)
    fig = plt.figure(figsize=(5.8 * ncols, 7.4 * n + 0.6))
    gs = fig.add_gridspec(2 * n, ncols, height_ratios=[1.0, 0.3] * n, hspace=0.08, wspace=0.1,
                          left=0.02, right=0.98, top=1 - 0.55 / (7.4 * n + 0.6), bottom=0.35 / (7.4 * n + 0.6))
    for r, it in enumerate(items):
        window = bbox(list(it["target"]) + list(it.get("mask_after") or []))
        panels = [("original mask", it["target"], it["printed_before"], C_PRINT_BEFORE)]
        if with_opc:
            panels.append(("OPC mask", it["mask_after"], it["printed_after"], C_PRINT_AFTER))
        for k, (label, mask, printed, color) in enumerate(panels):
            ax = fig.add_subplot(gs[2 * r, k])
            draw_polygons(ax, it["target"], face=TARGET_FILL, edge="none", zorder=1)
            draw_polygons(ax, mask, face="none", edge=C_MASK, lw=1.5, zorder=2)
            draw_contours(ax, printed, color, lw=2.2, zorder=3)
            layout_axes(ax, it["target"], 110, window)
            ax.set_title(f"({'ab'[k]}) {label}" + (" (= target)" if k == 0 else "")
                         + (f" — {it['name']}" if n > 1 else ""), loc="left")
            leg = ax.legend(handles=[Patch(facecolor=TARGET_FILL, edgecolor="none", label="target (desired)"),
                                     Patch(facecolor="none", edgecolor=C_MASK, linewidth=1.5, label=label),
                                     Line2D([], [], color=color, lw=2.2, label=f"printed with {label}")],
                            loc="upper right", frameon=True, fancybox=False, facecolor="white", framealpha=1.0,
                            edgecolor=INK_2, fontsize=8.5, borderpad=0.6, labelspacing=0.5, handlelength=1.8)
            leg.get_frame().set_linewidth(0.8)
            leg.set_zorder(10)
            mx = fig.add_subplot(gs[2 * r + 1, k])
            ax.apply_aspect()                                     # align the metric block with the square plot
            p, q = ax.get_position(), mx.get_position()
            mx.set_position([p.x0, q.y0, p.width, q.height])
            if k == 0 and it.get("metrics_before"):
                _metrics_block(mx, it["metrics_before"])
            elif k == 1 and it.get("metrics_after"):
                _metrics_block(mx, it["metrics_after"], it.get("metrics_before"), it.get("footer_after", ""))
            else:
                mx.axis("off")
    fig.text(0.02, 0.004, "EPE on edges: distance of the printed edge from the target edge, sampled every 1 nm outside "
             "the corner zones  ·  corner error: along each corner bisector  ·  area error: area where printed ≠ target",
             fontsize=8, color=MUTED, va="bottom")
    if title:
        fig.suptitle(title, y=0.998, fontsize=12, fontweight="bold", color=INK)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def fig_thumbnails(items, ncols: int = 10, path=None, title: str = ""):
    """Contact sheet of many OPC results: target (gray), OPC mask (violet), printed after OPC (blue).

    items: list of dict(name, target, mask, printed, caption).
    """
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 2.15 * nrows + 0.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, it in zip(axes, items):
        draw_polygons(ax, it["target"], face=TARGET_FILL, edge="none", zorder=1)
        draw_polygons(ax, it["mask"], face="none", edge=C_MASK, lw=0.6, zorder=2)
        draw_contours(ax, it["printed"], C_PRINT_AFTER, lw=0.9, zorder=3)
        layout_axes(ax, list(it["target"]) + list(it["mask"]), 40, scalebar=False)
        ax.set_title(it["name"], fontsize=6.5, fontweight="normal", loc="left", pad=2)
        ax.text(0.03, 0.03, it.get("caption", ""), transform=ax.transAxes, fontsize=6, color=INK_2, va="bottom")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.6, w_pad=0.4)
    if title:
        fig.suptitle(title, y=0.995, fontsize=13, fontweight="bold", color=INK)
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def fig_metric_distribution(before: list[dict], after: list[dict], path=None, title: str = ""):
    """Before/after distribution of each fidelity metric over many examples (box + every example as a dot)."""
    keys = [("epe_mean", "EPE on edges, mean [nm]", True), ("epe_max", "EPE on edges, max [nm]", True),
            ("corner", "corner error [nm]", False), ("area_pct", "area error [%]", False)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))
    rng = np.random.default_rng(0)
    for ax, (k, label, log) in zip(axes.ravel(), keys):
        data = [np.array([m[k] for m in before]), np.array([m[k] for m in after])]
        ax.grid(True, axis="x", which="major")
        for i, (d, c) in enumerate(zip(data, (C_PRINT_BEFORE, C_PRINT_AFTER))):
            y = 1 - i
            ax.boxplot(d, positions=[y], vert=False, widths=0.5, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="white", edgecolor=c, lw=1.4), medianprops=dict(color=c, lw=2),
                       whiskerprops=dict(color=c), capprops=dict(color=c))
            ax.scatter(np.maximum(d, 1e-2), y + rng.uniform(-0.17, 0.17, len(d)), s=10, color=c, alpha=0.55,
                       linewidths=0, zorder=3)
            ax.text(1.01, y, f"median {np.median(d):.2f}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=8.5, color=INK_2)
        red = 100 * (1 - np.median(data[1]) / np.median(data[0]))
        ax.set_yticks([1, 0], ["original mask", "OPC mask"])
        ax.set_ylim(-0.6, 1.6)
        if log:
            ax.set_xscale("log")
        ax.set_title(f"{label}   (median {red:.0f}% lower)", loc="left", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0, 0.94, 0.93))
    if title:
        fig.suptitle(title, y=0.99, fontsize=12, fontweight="bold", color=INK)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


# ======================================================================= assignment
C_MISMATCH = "#e34948"
C_CROSS = "#1baf7a"            # intensity-model OPC mask evaluated with the Hopkins model


def _dashed_target(ax, target, lw=1.0, label=None):
    for i, p in enumerate(target):
        q = np.vstack([p, p[:1]])
        ax.plot(q[:, 0], q[:, 1], color=INK, lw=lw, ls=(0, (3, 2)), zorder=6, label=label if i == 0 else None)


def _panel_legend(ax, handles, loc="upper right", fontsize=7.5):
    leg = ax.legend(handles=handles, loc=loc, frameon=True, fancybox=False, facecolor="white", framealpha=1.0,
                    edgecolor=INK_2, fontsize=fontsize, borderpad=0.5, labelspacing=0.35, handlelength=1.6)
    leg.get_frame().set_linewidth(0.7)
    leg.set_zorder(20)


def _cell_window(ax, target, pad=12.0, legend_room=0.2):
    """Same window for every panel of a cell; legend_room (fraction of the height) is left free above the
    pattern so the in-panel legend never covers it."""
    xmin, ymin, xmax, ymax = bbox(target)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad + legend_room * (ymax - ymin))
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x [nm]", fontsize=8)
    ax.set_ylabel("y [nm]", fontsize=8)


COL_W_IN = 3.35                # one column of the two-column paper (8.5 cm)


def _col_axes(ax, fontsize=6):
    ax.tick_params(labelsize=fontsize, length=2, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def fig_light_distribution(dist2d, pixel, sigma, radius, table, path=None):
    """Column figure: (a) 2D Gaussian light distribution with the cut-off circle, (b) light amount per lambda cell."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_W_IN, 1.78), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    h = dist2d.shape[0] // 2 * pixel
    ax.imshow(dist2d / dist2d.max(), extent=(-h, h, -h, h), origin="lower", cmap=CMAP_INTENSITY, vmin=0, vmax=1)
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, color=INK_2, lw=0.6, ls=":"))
    ax.text(0, -radius * 0.93, f"4σ = {radius:.2f} nm", ha="center", va="bottom", fontsize=5.5, color=INK_2)
    ax.set_xticks([-20, 0, 20])
    ax.set_yticks([-20, 0, 20])
    ax.set_xlabel("x [nm]", fontsize=6.5, labelpad=1)
    ax.set_ylabel("y [nm]", fontsize=6.5, labelpad=0)
    ax.set_title(f"(a) Gaussian, σ = {sigma:.2f} nm", loc="left", fontsize=7)
    _col_axes(ax)
    ax = axes[1]
    n = table.shape[0]
    ax.imshow(table, cmap=CMAP_INTENSITY, vmin=0, vmax=table.max() * 1.3)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{table[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if table[i, j] > 0.55 * table.max() else INK)
    ax.set_xticks(range(n), [f"{(j - n // 2) * 3.5:g}" for j in range(n)])
    ax.set_yticks(range(n), [f"{(i - n // 2) * 3.5:g}" for i in range(n)])
    ax.set_xlabel("offset [nm] (1 cell = λ)", fontsize=6.5, labelpad=1)
    ax.set_title("(b) per λ cell, center = 10", loc="left", fontsize=7)
    _col_axes(ax)
    if path:
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return fig


def fig_targets(cells, path=None):
    """Column figure. cells: list of dict(name, target, notes=[(xy_from, xy_to, text, side)]) on the 3.5 nm lambda
    grid; side "above" centers the text over the arrow, "right" puts it to the right."""
    widths = [bbox(c["target"])[2] - bbox(c["target"])[0] + 16 for c in cells]
    fig, axes = plt.subplots(1, len(cells), figsize=(COL_W_IN, 2.45),
                             gridspec_kw={"width_ratios": widths, "wspace": 0.18})
    for ax, c in zip(np.atleast_1d(axes), cells):
        xmin, ymin, xmax, ymax = bbox(c["target"])
        for x in np.arange(xmin, xmax + 1e-6, 3.5):
            ax.axvline(x, color=GRID, lw=0.25, zorder=0)
        for y in np.arange(ymin, ymax + 1e-6, 3.5):
            ax.axhline(y, color=GRID, lw=0.25, zorder=0)
        draw_polygons(ax, c["target"], face="#9cc3ea", edge=C_SERIES[0], lw=0.6, zorder=2)
        for p0, p1, text, side in c.get("notes", []):
            ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="<->", color=INK, lw=0.6, shrinkA=0, shrinkB=0,
                                                    mutation_scale=5), zorder=5)
            (x0, y0), (x1, y1) = p0, p1
            if side == "above":
                xy, ha, va = ((x0 + x1) / 2, max(y0, y1) + 1.2), "center", "bottom"
            else:
                xy, ha, va = (max(x0, x1) + 1.5, (y0 + y1) / 2), "left", "center"
            ax.text(*xy, text, fontsize=5, color=INK, ha=ha, va=va, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.3, alpha=0.9))
        ax.set_xlim(xmin - 8, xmax + 8)
        ax.set_ylim(ymin - 8, ymax + 8)
        ax.set_aspect("equal")
        ax.set_xlabel("x [nm]", fontsize=6.5, labelpad=1)
        ax.set_title(c["name"], loc="left", fontsize=7)
        _col_axes(ax)
    np.atleast_1d(axes)[0].set_ylabel("y [nm]", fontsize=6.5, labelpad=1)
    if path:
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return fig


def fig_cell_result(name, target, rows, path=None, width_in=COL_W_IN, window=None):
    """Column (or page-width) figure. rows (no OPC, OPC): dict(label, mask, grid, exposure, threshold, contours, color, metrics).
    Columns: mask with the target, exposure distribution with the threshold contour, developed pattern on the
    target with the mismatch (XOR) area. One legend for the figure, one colorbar for the exposure column.
    window (w, h) in nm: the same view for every cell, so the figures of different cells share one scale."""
    import gdstk
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from .layout import even_odd_fill
    pad = 8.0
    xmin, ymin, xmax, ymax = bbox(target)
    wx, wy = window or (xmax - xmin + 2 * pad, ymax - ymin + 2 * pad)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    W = width_in
    f = min(W / COL_W_IN, 1.35)                    # text and margins grow with the figure, but less than the panels
    left, right, gap, hgap = 0.52 * f, 0.04, 0.05 * f, 0.05 * f
    one_row = f >= 1.2                            # a page-width legend fits on one line
    top, title_h, bottom = (0.24 if one_row else 0.36) * f, 0.13 * f, 0.50 * f
    pw = (W - left - right - 2 * gap) / 3
    ph = pw * wy / wx
    H = top + len(rows) * (title_h + ph) + (len(rows) - 1) * hgap + bottom
    fig = plt.figure(figsize=(W, H))
    letters = "abcdef"
    im = None
    for r, row in enumerate(rows):
        g, m = row["grid"], row["metrics"]
        y0 = H - top - (r + 1) * (title_h + ph) - r * hgap
        axes = [fig.add_axes([(left + k * (pw + gap)) / W, y0 / H, pw / W, ph / H]) for k in range(3)]
        printed = even_odd_fill(row["contours"])
        xor = gdstk.boolean([gdstk.Polygon(p) for p in printed], [gdstk.Polygon(p) for p in target], "xor")
        draw_polygons(axes[0], row["mask"], face=C_MASK, edge="none", alpha=0.25, zorder=2)
        draw_polygons(axes[0], row["mask"], face="none", edge=C_MASK, lw=0.6 * f, zorder=3)
        im = axes[1].imshow(row["exposure"], origin="lower", extent=g.extent, cmap=CMAP_INTENSITY, vmin=0, vmax=1.0,
                            interpolation="bilinear")
        axes[1].contour(g.x_centers(), g.y_centers(), row["exposure"], levels=[row["threshold"]],
                        colors=[row["color"]], linewidths=0.8 * f)
        draw_polygons(axes[2], printed, face=row["color"], edge="none", alpha=0.35, zorder=2)
        draw_contours(axes[2], row["contours"], row["color"], lw=0.7 * f, zorder=3)
        draw_polygons(axes[2], [q.points for q in xor], face=C_MISMATCH, edge="none", alpha=0.75, zorder=4)
        for k, (ax, t) in enumerate(zip(axes, ("mask", "exposure", "developed"))):
            _dashed_target(ax, target, lw=0.55 * f)
            ax.set_xlim(cx - wx / 2, cx + wx / 2)
            ax.set_ylim(cy - wy / 2, cy + wy / 2)
            ax.set_aspect("equal")
            _col_axes(ax, 5.5 * f)
            ax.set_title(f"({letters[3 * r + k]}) {t}", loc="left", fontsize=6.5 * f, pad=2)
            if k:
                ax.tick_params(labelleft=False)
            if r < len(rows) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("x [nm]", fontsize=6 * f, labelpad=1)
        fig.text(0.02 / W, (y0 + ph / 2) / H, f"{row['label']}\nIoU {m['iou']:.2f}, {m['area_pct']:.2f} %",
                 rotation=90, ha="left", va="center", fontsize=6 * f, color=row["color"], fontweight="bold")
    cax = fig.add_axes([(left + pw + gap) / W, 0.10 * f / H, pw / W, 0.05 * f / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("exposure (open area = 1)", fontsize=5.5 * f, labelpad=1)
    cb.ax.tick_params(labelsize=5 * f, length=1.5, pad=1)
    cb.outline.set_linewidth(0.5)
    handles = [Line2D([], [], color=INK, ls=(0, (3, 2)), lw=0.8, label="target"),
               Patch(facecolor=(0.29, 0.23, 0.65, 0.25), edgecolor=C_MASK, lw=0.6, label="mask")]
    for row in rows:
        handles.append(Line2D([], [], color=row["color"], lw=1.2, label=f"printed, {row['label']}"))
    handles.append(Patch(facecolor=C_MISMATCH, alpha=0.75, label="mismatch"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.02 / H), ncol=5 if one_row else 3,
               fontsize=5.5 * f, frameon=False, handlelength=1.6, columnspacing=1.0, labelspacing=0.3)
    if path:
        fig.savefig(path, dpi=300, bbox_inches=None)
        plt.close(fig)
    return fig


def fig_dose(dose, path=None):
    """Column figure. dose: dict(x, lines: {label: (values, color, ls, bridge_flags)}, nominal)."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(COL_W_IN, 2.1))
    ax.grid(True, axis="y", lw=0.4)
    for label, (vals, col, ls, flags) in dose["lines"].items():
        ax.plot(dose["x"], vals, color=col, ls=ls, lw=1.1, marker="o", ms=2.5, label=label)
        fx = [x for x, f in zip(dose["x"], flags) if f]
        if fx:
            ax.scatter(fx, [v for v, f in zip(vals, flags) if f], marker="x", s=20, color=C_MISMATCH, zorder=5)
    ax.axvline(dose["nominal"], color=MUTED, lw=0.6, ls="--")
    ax.set_xlabel("relative dose D", fontsize=6.5, labelpad=1)
    ax.set_ylabel("mismatch [% of target]", fontsize=6.5, labelpad=1)
    _clean_axes(ax)
    _col_axes(ax)
    ax.legend(fontsize=5.5, frameon=False, loc="upper center", ncol=2, handlelength=2.2, columnspacing=1.0)
    ax.set_ylim(0, max(max(v[0]) for v in dose["lines"].values()) * 1.25)
    fig.tight_layout(pad=0.3)
    if path:
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return fig


def fig_opc_progress(curves, snaps, target, path=None):
    """curves: name -> dict(it, epe_rms, epe_max, mism, iou, conv); snaps: list of (iteration, mask, contours)."""
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, len(snaps), height_ratios=[1, 1.35], hspace=0.38, wspace=0.25)
    top = [fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:])] if len(snaps) >= 5 \
        else [fig.add_subplot(gs[0, i]) for i in range(3)]
    for i, (name, c) in enumerate(curves.items()):
        col = C_SERIES[i]
        top[0].semilogy(c["it"], np.maximum(c["epe_rms"], 1e-3), color=col, lw=1.8, label=f"{name} rms")
        top[0].semilogy(c["it"], np.maximum(c["epe_max"], 1e-3), color=col, lw=1.2, ls="--", label=f"{name} max")
        top[1].plot(c["it"], c["mism"], color=col, lw=1.8, label=name)
        top[2].plot(c["it"], c["iou"], color=col, lw=1.8, label=name)
        for ax in top:
            if c["conv"] is not None:
                ax.axvline(c["conv"], color=col, lw=0.7, ls=":")
    for ax, t, yl in zip(top, ["(a) EPE at the control sites", "(b) mismatch area", "(c) IoU"],
                         ["EPE [nm]", "mismatch [% of target]", "IoU"]):
        ax.set_title(t, loc="left", fontsize=10)
        ax.set_xlabel("iteration")
        ax.set_ylabel(yl)
        ax.grid(True, axis="y")
        _clean_axes(ax)
        ax.legend(fontsize=7.5, loc="upper right" if ax is not top[2] else "lower right", frameon=True,
                  edgecolor=INK_2, fancybox=False)
    for j, (it, mask, contours) in enumerate(snaps):
        ax = fig.add_subplot(gs[1, j])
        draw_polygons(ax, mask, face=C_MASK, edge=C_MASK, alpha=0.2, lw=0.6, zorder=2)
        draw_contours(ax, contours, C_PRINT_AFTER if it > 0 else C_PRINT_BEFORE, lw=1.1, zorder=3)
        _dashed_target(ax, target, lw=0.7)
        _cell_window(ax, target, pad=8)
        ax.set_title(f"({'defghij'[j]}) iteration {it}", loc="left", fontsize=9)
        if j == 0:
            _panel_legend(ax, [Line2D([], [], color=C_MASK, lw=1, label="mask"),
                               Line2D([], [], color=C_PRINT_BEFORE, lw=1.2, label="developed (it. 0)"),
                               Line2D([], [], color=C_PRINT_AFTER, lw=1.2, label="developed (it. > 0)"),
                               Line2D([], [], color=INK, ls=(0, (3, 2)), lw=0.8, label="target")], fontsize=6.5)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def fig_conditions(dose, sigma, path=None):
    """dose / sigma: dict(x, lines: {label: (values, color, ls, bridge_flags)}, nominal) for the two sweeps."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, d, title, xl in ((axes[0], dose, "(a) exposure-dose variation", "relative dose (threshold / dose)"),
                             (axes[1], sigma, "(b) Gaussian width variation (masks fixed)", "Gaussian width σ [nm]")):
        ax.grid(True, axis="y")
        for label, (vals, col, ls, flags) in d["lines"].items():
            ax.plot(d["x"], vals, color=col, ls=ls, lw=1.8, marker="o", ms=3.5, label=label)
            fx = [x for x, f in zip(d["x"], flags) if f]
            fy = [v for v, f in zip(vals, flags) if f]
            if fx:
                ax.scatter(fx, fy, marker="x", s=46, color=C_MISMATCH, zorder=5, linewidths=1.6)
        ax.axvline(d["nominal"], color=MUTED, lw=0.8, ls="--")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel("mismatch [% of target]")
        _clean_axes(ax)
        h, l = ax.get_legend_handles_labels()
        from matplotlib.lines import Line2D
        h.append(Line2D([], [], marker="x", color=C_MISMATCH, ls="none", label="bridge / open"))
        ax.legend(handles=h, fontsize=7.5, frameon=True, edgecolor=INK_2, fancybox=False, loc="upper center")
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def fig_model_compare(rows, path=None):
    """rows (one per cell): dict(name, target, grid, exp_int, thr_int, exp_hop, thr_hop, cont: {label: (contours, color)})."""
    from matplotlib.lines import Line2D
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 5.2 * n), squeeze=False)
    for r, row in enumerate(rows):
        g, t = row["grid"], row["target"]
        for k, (key, thr, title) in enumerate((("exp_int", row["thr_int"], "intensity superposition"),
                                               ("exp_hop", row["thr_hop"], "Hopkins (interference)"))):
            ax = axes[r][k]
            im = ax.imshow(row[key], origin="lower", extent=g.extent, cmap=CMAP_INTENSITY, vmin=0,
                           vmax=max(1.0, float(row[key].max())), interpolation="bilinear")
            ax.contour(g.x_centers(), g.y_centers(), row[key], levels=[thr], colors=[C_PRINT_BEFORE], linewidths=1.3)
            _dashed_target(ax, t)
            _cell_window(ax, t)
            ax.set_title(f"({'abcdef'[3 * r + k]}) {row['name']}: {title}, original mask", loc="left", fontsize=9)
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
            cb.ax.tick_params(labelsize=6)
            cb.outline.set_edgecolor(AXIS)
            _panel_legend(ax, [Line2D([], [], color=C_PRINT_BEFORE, lw=1.3, label=f"threshold {thr:g}"),
                               Line2D([], [], color=INK, ls=(0, (3, 2)), lw=1, label="target")])
        ax = axes[r][2]
        hs = []
        for label, (cont, col) in row["cont"].items():
            draw_contours(ax, cont, col, lw=1.4)
            hs.append(Line2D([], [], color=col, lw=1.4, label=label))
        _dashed_target(ax, t)
        _cell_window(ax, t)
        ax.set_title(f"({'abcdef'[3 * r + 2]}) printed with the Hopkins model", loc="left", fontsize=9)
        hs.append(Line2D([], [], color=INK, ls=(0, (3, 2)), lw=1, label="target"))
        _panel_legend(ax, hs, fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def fig_sraf_focus(curves, mask_view, path=None):
    """(a) mismatch vs defocus with / without assist features (Hopkins); (b) the OPC mask with assist features."""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    ax.grid(True, axis="y")
    for label, (x, y, col, ls) in curves.items():
        ax.plot(x, y, color=col, ls=ls, lw=1.8, marker="o", ms=3.5, label=label)
    ax.set_xlabel("defocus [nm]")
    ax.set_ylabel("mismatch [% of target]")
    ax.set_title("(a) through-focus print fidelity (Hopkins model)", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=True, edgecolor=INK_2, fancybox=False, loc="upper center")
    _clean_axes(ax)
    ax = axes[1]
    t = mask_view["target"]
    draw_polygons(ax, mask_view["mask"], face=C_MASK, edge=C_MASK, alpha=0.25, lw=0.8)
    draw_polygons(ax, mask_view["srafs"], face=C_CROSS, edge=C_CROSS, alpha=0.6, lw=0.8)
    _dashed_target(ax, t)
    _cell_window(ax, t, pad=30)
    ax.set_title(f"(b) {mask_view['name']}: OPC mask with assist features", loc="left", fontsize=10)
    _panel_legend(ax, [Line2D([], [], color=C_MASK, lw=4, alpha=0.5, label="OPC mask (main)"),
                       Line2D([], [], color=C_CROSS, lw=4, alpha=0.7, label="assist features"),
                       Line2D([], [], color=INK, ls=(0, (3, 2)), lw=1, label="target")])
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig
