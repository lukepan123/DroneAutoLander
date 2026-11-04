#!/usr/bin/env python3
"""
Figures for the static YOLO box-fit sweep (journal, two-column).

Inputs : sweep_out/metrics_geom.csv, sweep_out/metrics_case.csv (from analyse_sweep.py)
Outputs: sweep_out/plots/
    fan_rmse.pdf/.png      double column: lateral | longitudinal RMSE of correct
                           detections over the (horizontal distance, height) plane,
                           pooled over aspect, one shared colour scale. Hatched =
                           correct detections < 90 % of frames, grey = < 50 %,
                           blank = not flown (below MIN_ALT).
    fan_detection.pdf/.png double column: % of hold-period frames with a correct
                           detection | with YOLO locked onto the wrong object.
    aspect_bias.pdf/.png   single column: signed lateral / longitudinal bias vs
                           aspect, one line per slant range (mean over elevations).
    aspect_bias_by_elev.pdf/.png
                           small multiples, one panel per elevation, for the appendix
                           or for checking that pooling over elevation is fair.
Output: sweep_out/plots_compare/
    error_vs_geometry.pdf/.png  single column (CMP_LAYOUT), styled like batch_analysis.py:
                           (a) error vs elevation | (b) error vs slant range, one line
                           per condition (mesh / ray-cast x true / wrong height), mean
                           over cases with a ±1σ band. Config: CMP_* below.
    error_vs_geometry_legend.pdf  the shared legend on its own (LEGEND_SEPARATE)
    error_vs_geometry.csv  the plotted numbers, for quoting in the text

Conditions (SWEEP-TEMP): with SWEEP_OUT set, plots that one folder (as run_sweep.sh does).
Without it, plots every condition folder under sweep_out/ that has a manifest.csv
(mesh_true, raycast_true, mesh_wrong, raycast_wrong, ...), re-running analyse_sweep.py
first wherever the metrics are missing or older than the manifest. Folders sharing a
suffix (_true, _wrong) share one RMSE colour scale, so mesh and ray-cast are comparable
within a height condition. Plots land in sweep_out/<tag>/plots/; ray-cast folders also
get plots_naive/ (raw box-centre estimate, no pad offset). --reanalyse forces
analyse_sweep.py to run again (do this after changing analyse_sweep.py).

Usage: python3 plot_sweep.py [--annotate]    (--annotate prints values in each cell: RMSE in cm, rates in %)
"""
# Written for matplotlib >= 3.5 (Ubuntu 22.04 apt version)
import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Wedge

HERE = Path(__file__).resolve().parent
SWEEP_ROOT = HERE / "sweep_out"
OUT_DIR = Path(os.environ.get("SWEEP_OUT", SWEEP_ROOT))   # SWEEP-TEMP: per condition
PLOT_DIR = OUT_DIR / "plots"                             # reset per condition in main()

DET_OK = 0.90        # correct-detection rate below this: cell hatched
DET_MIN = 0.50       # correct-detection rate below this (or no fits): cell greyed out

# ---- style: viridis everywhere, for consistency with the rest of the thesis ---------
# Perceptually uniform and readable in greyscale and for colour-blind readers.
SEQ = plt.get_cmap("viridis")
VIRIDIS_LINE_MAX = 0.85   # stop short of viridis' pale yellow end, which vanishes on white
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
NO_DATA = "#e4e3df"
COL1, COL2 = 3.5, 7.16   # inches, single / double column

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 400, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def edges(values, half_default):
    """Cell edges halfway between sorted grid values (half_default if only one value)."""
    v = np.sort(np.unique(values)).astype(float)
    if len(v) == 1:                                                    # SWEEP-TEMP
        return {v[0]: (v[0] - half_default, v[0] + half_default)}
    mid = (v[:-1] + v[1:]) / 2
    return dict(zip(v, zip(np.r_[v[0] - (mid[0] - v[0]), mid],
                           np.r_[mid, v[-1] + (v[-1] - mid[-1])])))


def save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(PLOT_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {PLOT_DIR / name}.pdf/.png")


# ---- fan maps -----------------------------------------------------------------------
def draw_fan(ax, geom, key, cmap, norm, label_fmt, annotate, flag_detection):
    """One fan panel: a wedge per (R, elev) cell, coloured by geom[key]."""
    r_edge, e_edge = edges(geom.R, 2.5), edges(geom.elev, 7.5)
    r_max = max(hi for _, hi in r_edge.values())
    for _, g in geom.iterrows():
        r0, r1 = r_edge[g.R]
        e0, e1 = e_edge[g.elev]
        e0, e1 = max(e0, 0.0), min(e1, 90.0)
        bad = not np.isfinite(g[key])
        if flag_detection:
            bad = bad or g.n == 0 or g.valid_rate < DET_MIN
        face = NO_DATA if bad else cmap(norm(g[key]))
        ax.add_patch(Wedge((0, 0), r1, e0, e1, width=r1 - r0, facecolor=face,
                           edgecolor="white", linewidth=1.0))
        if flag_detection and not bad and g.valid_rate < DET_OK:
            ax.add_patch(Wedge((0, 0), r1, e0, e1, width=r1 - r0, facecolor="none",
                               edgecolor=INK2, linewidth=0, hatch="////"))
        arc = np.radians(e1 - e0) * (r0 + r1) / 2      # cell width at its middle (m)
        if annotate and not bad and arc > 2.5 and r1 - r0 > 2.5:
            rc, ec = np.radians((e0 + e1) / 2), (r0 + r1) / 2
            r_, g_, b_, _ = cmap(norm(g[key]))
            light = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_ > 0.5   # ink vs white label
            ax.text(ec * np.cos(rc), ec * np.sin(rc), label_fmt(g[key]),
                    ha="center", va="center", fontsize=5.5, color=INK if light else "white")
    # Elevation rays as a light polar guide
    for e in sorted(e_edge):
        a = np.radians(e)
        ax.plot([0, r_max * np.cos(a)], [0, r_max * np.sin(a)], color=MUTED,
                lw=0.4, ls=":", zorder=0)
        ax.text(1.03 * r_max * np.cos(a) + (1.5 if e >= 80 else 0), 1.03 * r_max * np.sin(a),
                f"{e:.0f}°", fontsize=6, color=INK2, ha="left", va="center")
    ax.plot(0, 0, marker="s", ms=4, color=INK, zorder=5)
    ax.set_xlim(-1, r_max * 1.12)
    ax.set_ylim(-1, r_max * 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal distance to pad (m)")


def colorbar(fig, ax, cmap, norm, label, fmt=None):
    with plt.rc_context({"axes.grid": False}):
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          shrink=0.85, pad=0.02, label=label, format=fmt)
    cb.outline.set_visible(False)


def fan_rmse(geom, annotate, vmax=None):
    """Lateral | longitudinal RMSE of correct detections, one shared scale."""
    if vmax is None:
        vmax = np.nanmax(geom[["rmse_lat", "rmse_lon"]].to_numpy())
    norm = Normalize(0, vmax if np.isfinite(vmax) and vmax > 0 else 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0), sharey=True,
                             gridspec_kw=dict(wspace=0.08), constrained_layout=True)
    for ax, key, title in zip(axes, ("rmse_lat", "rmse_lon"),
                              ("Lateral RMSE (car orientation averaged)", "Longitudinal RMSE (car orientation averaged)")):
        draw_fan(ax, geom, key, SEQ, norm, lambda v: f"{100 * v:.0f}", annotate, True)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Height above pad (m)")
    colorbar(fig, list(axes), SEQ, norm, "RMSE (m)")
    handles = [Patch(facecolor="white", edgecolor=INK2, hatch="////",
                     label=f"Car detected in < {DET_OK:.0%} of frames"),
               Patch(facecolor=NO_DATA, edgecolor="none",
                     label=f"Car detected in < {DET_MIN:.0%} of frames")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               frameon=False)
    save(fig, "fan_rmse")


def fan_detection(geom, annotate):
    """Share of hold-period frames with a correct detection | a wrong-object detection."""
    norm = Normalize(0, 1)
    pct = lambda v: f"{100 * v:.0f}"
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0), sharey=True,
                             gridspec_kw=dict(wspace=0.08), constrained_layout=True)
    draw_fan(axes[0], geom, "valid_rate", SEQ, norm, pct, annotate, False)
    axes[0].set_title("Correct detections (% of frames)", loc="left")
    colorbar(fig, axes[0], SEQ, norm, None, fmt=lambda v, _: f"{100 * v:.0f}%")
    draw_fan(axes[1], geom, "false_rate", SEQ, norm, pct, annotate, False)
    axes[1].set_title("Wrong-object detections (% of frames)", loc="left")
    colorbar(fig, axes[1], SEQ, norm, None, fmt=lambda v, _: f"{100 * v:.0f}%")
    axes[0].set_ylabel("Height above pad (m)")
    save(fig, "fan_detection")


# ---- figure 2: bias vs aspect -------------------------------------------------------
def range_colours(ranges):
    """Viridis steps, dark purple (near) -> green-yellow (far)."""
    pos = np.linspace(0.0, VIRIDIS_LINE_MAX, len(ranges))
    return {R: SEQ(p) for R, p in zip(sorted(ranges), pos)}


def aspect_bias(case):
    ranges = sorted(case.R.unique())
    colours = range_colours(ranges)
    m = case.groupby(["R", "aspect"])[["bias_lat", "bias_lon"]].mean().reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(COL1, 3.6), sharex=True, constrained_layout=True)
    for ax, key, label in zip(axes, ("bias_lat", "bias_lon"),
                              ("Lateral bias (m)", "Longitudinal bias (m)")):
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        for R in ranges:
            d = m[m.R == R].sort_values("aspect")
            ax.plot(d.aspect, d[key], color=colours[R], lw=1.5, marker="o", ms=3,
                    markeredgecolor="white", markeredgewidth=0.6, label=f"{R:g} m", zorder=3)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Aspect (deg, 0 = viewing the nose)")
    axes[-1].set_xticks(sorted(case.aspect.unique()))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Slant range", ncol=len(ranges), loc="lower center", bbox_to_anchor=(0.5, 1.0),
               frameon=False, title_fontsize=7, handlelength=1.2, columnspacing=0.8)
    save(fig, "aspect_bias")


def aspect_bias_by_elev(case):
    elevs = sorted(case.elev.unique(), reverse=True)
    ranges = sorted(case.R.unique())
    colours = range_colours(ranges)
    fig, axes = plt.subplots(2, len(elevs), figsize=(COL2, 3.4), sharex=True,
                             sharey="row", constrained_layout=True, squeeze=False)
    for j, e in enumerate(elevs):
        for i, (key, label) in enumerate((("bias_lat", "Lateral bias (m)"),
                                          ("bias_lon", "Longitudinal bias (m)"))):
            ax = axes[i, j]
            ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
            for R in ranges:
                d = case[(case.elev == e) & (case.R == R)].sort_values("aspect")
                if len(d):
                    ax.plot(d.aspect, d[key], color=colours[R], lw=1.2, label=f"{R:g} m")
            if i == 0:
                ax.set_title(f"{e:g}° elevation", loc="left")
            if j == 0:
                ax.set_ylabel(label)
            ax.set_xticks([0, 90, 180])
    fig.supxlabel("Aspect (deg, 0 = viewing the nose)", fontsize=8)
    handles = [plt.Line2D([], [], color=colours[R], lw=1.2) for R in ranges]
    fig.legend(handles, [f"{R:g} m" for R in ranges], title="Slant range",
               loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, title_fontsize=7)
    save(fig, "aspect_bias_by_elev")


# ---- figure 3: error vs elevation | error vs range, every condition ------------------
# Styled to match batch_analysis.py (PAPER_RC, viridis stops, mean line with a ±1σ band,
# legend written separately). Colour + marker = measurement method, dash = assumed height.
CMP_CONDITIONS = {          # folder -> (legend label, viridis stop, dash, marker)
    "mesh_true":     ("Proj. Mesh, True Height",     0.15, "-",  "o"),
    "raycast_true":  ("Ray-Cast, True Height",            0.55, "-",  "s"),
    "mesh_wrong":    ("Proj. Mesh, Wrong Height",    0.15, "--", "o"),
    "raycast_wrong": ("Ray-Cast, Wrong Height",           0.55, "--", "s"),
}
CMP_RAYCAST_SUFFIX = "_naive"   # ray-cast metrics: "_naive" = box-centre ray, "" = + pad offset
CMP_METRIC   = "xy"             # per-case error: "xy" = hypot(lat, lon) RMSE, "lat", "lon"
CMP_ASPECTS  = None             # None = every aspect; e.g. [150, 180] = chase geometry only
CMP_R_MAX    = 15.0             # m, drop longer slant ranges (detection collapses at 20 m)
CMP_MIN_N    = 10               # drop cases with fewer correct-detection frames
CMP_BALANCED = True             # each panel pools only over the values of the other variable
                                # flown at EVERY x point, so the trend is not a change in mix
                                # (low elevations are not flown at short range: MIN_ALT)
CMP_NAME     = "error_vs_geometry"
CMP_LAYOUT   = "stacked"        # "stacked" = 2 rows x 1 column, one IEEE column (3.5 in)
                               # "row"     = 1 row x 2 columns, full page width (7.16 in)
CMP_SHARE_Y  = False             # one RMSE scale for both panels
CMP_PANEL_H  = 1.45             # in, height per panel when stacked
CMP_YLABEL   = "shared"         # "shared" = one y label for the figure, "each" = per panel
PAPER_RC = {                    # copied from batch_analysis.py
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "lines.linewidth": 0.8, "lines.markersize": 2.6,
    "axes.linewidth": 0.5, "grid.linewidth": 0.3, "grid.alpha": 0.3,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.minor.size": 1.2, "xtick.minor.width": 0.4,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.0,
    "axes.titlepad": 3.0,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.2,
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
}
BAND_ALPHA = 0.18
CMP_BANDS  = False               # False: mean lines only, no ±1σ bands
LEGEND_SEPARATE = True          # as batch_analysis.py: <name>_legend.pdf placed once
LEGEND_BOX      = True          # thin frame around the legend
LEGEND_EDGE     = "0.8"   # frame colour (grey level 0 black to 1 white, or any colour)
LEGEND_ROUND    = True    # rounded corners
LEGEND_FONTSIZE = 7.5           # pt, legend text
# The separate legend file is saved at exactly the figure's width (height fitted), so
# \includegraphics[width=\columnwidth] keeps the text at LEGEND_FONTSIZE. Entries that
# do not fit in one row wrap onto more rows automatically.


def place_legend(fig, handles, labels, ncol, **kw):
    """fig.legend with the shared look, dropping columns until it fits the figure width."""
    while True:
        leg = fig.legend(handles=handles, labels=labels, ncol=ncol, fontsize=LEGEND_FONTSIZE,
                         frameon=LEGEND_BOX, fancybox=LEGEND_ROUND, edgecolor=LEGEND_EDGE,
                         framealpha=1.0, borderpad=0.4, handletextpad=0.5, **kw)
        leg.get_frame().set_linewidth(plt.rcParams["axes.linewidth"])
        fig.canvas.draw()
        if ncol == 1 or leg.get_window_extent().width <= fig.bbox.width - 2:
            return leg
        leg.remove()
        ncol -= 1


def case_error(case):
    if CMP_METRIC == "xy":
        return np.hypot(case.rmse_lat, case.rmse_lon)
    return case[f"rmse_{CMP_METRIC}"]


def cmp_cases(loaded):
    """{folder name: per-case table with an 'err' column}, filtered as configured."""
    out = {}
    for (d, suffix), (_, case) in loaded.items():
        if d.name not in CMP_CONDITIONS:
            continue
        want = CMP_RAYCAST_SUFFIX if d.name.startswith("raycast") else ""
        if (d, want) not in loaded:      # true-heading runs have no _naive variant
            want = ""
        if suffix != want:
            continue
        c = case[case.R <= CMP_R_MAX].copy()
        if CMP_ASPECTS is not None:
            c = c[c.aspect.isin(CMP_ASPECTS)]
        c["err"] = case_error(c)
        out[d.name] = c
    return out


def common_values(cases, x, other):
    """Values of `other` flown at every value of `x` (before the n filter)."""
    allc = pd.concat(cases.values())
    sets = [set(g[other]) for _, g in allc.groupby(x)]
    return sorted(set.intersection(*sets)) if sets else []


def line_band(ax, x, y, s, colour, ls, marker):
    """Mean line with a translucent ±1σ band (clipped at 0), as batch_analysis.py."""
    x, y, s = np.asarray(x, float), np.asarray(y, float), np.nan_to_num(np.asarray(s, float))
    ax.plot(x, y, color=colour, ls=ls, marker=marker, zorder=3)
    if not CMP_BANDS:
        return
    ax.fill_between(x, np.clip(y - s, 0, None), y + s, color=colour, alpha=BAND_ALPHA,
                    lw=0, zorder=2)


def error_vs_geometry(loaded, out_dir):
    cases = cmp_cases(loaded)
    if not cases:
        print("  error_vs_geometry: no matching condition folders, skipped")
        return
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    # (x, other, x label, (labelled tick step, minor gridline step), x-axis start or None)
    panels = (("elev", "R", "Elevation view angle (deg)", (15, 7.5), None),
              ("R", "elev", "Distance to pad (m)", (5, 2.5), 0.0))
    keep = {x: (common_values(cases, x, other) if CMP_BALANCED else None)
            for x, other, *_ in panels}

    summary = []
    with plt.style.context("default"), plt.rc_context(PAPER_RC):
        stacked = CMP_LAYOUT == "stacked"
        width = COL1 if stacked else COL2
        fig, axes = plt.subplots(*((2, 1) if stacked else (1, 2)),
                                 figsize=(width, 2 * CMP_PANEL_H + 0.45 if stacked else 1.85),
                                 sharey=CMP_SHARE_Y, constrained_layout=True)
        pads = dict(w_pad=0.03, h_pad=0.03, wspace=0.04, hspace=0.06)
        if hasattr(fig, "get_layout_engine"):      # matplotlib 3.6+
            fig.get_layout_engine().set(**pads)
        else:                                      # 3.5 (Ubuntu 22.04)
            fig.set_constrained_layout_pads(**pads)
        handles = []
        for name, (lab, stop, ls, mk) in CMP_CONDITIONS.items():
            if name not in cases:
                continue
            colour = SEQ(stop)
            for ax, (x, other, *_) in zip(axes, panels):
                c = cases[name][cases[name].n >= CMP_MIN_N]
                if keep[x] is not None:
                    c = c[c[other].isin(keep[x])]
                g = c.groupby(x)["err"].agg(["mean", "std", "count"]).reset_index()
                line_band(ax, g[x], g["mean"], g["std"], colour, ls, mk)
                for _, r in g.iterrows():
                    summary.append(dict(condition=name, x=x, value=r[x], mean=r["mean"],
                                        std=r["std"], n_cases=int(r["count"])))
            line = Line2D([], [], color=colour, ls=ls, marker=mk, label=lab)
            handles.append((Patch(facecolor=colour, alpha=BAND_ALPHA, lw=0), line)
                           if CMP_BANDS else line)

        for letter, ax, (x, other, xlab, (major, minor), x0) in zip("ab", axes, panels):
            vals = sorted(pd.concat(cases.values())[x].unique())
            ax.xaxis.set_major_locator(MultipleLocator(major))   # evenly spaced labels
            ax.xaxis.set_minor_locator(MultipleLocator(minor))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.set_xlim(min(vals) - 0.5 * minor if x0 is None else x0,
                        max(vals) + 0.5 * minor)
            ax.grid(True, which="major")
            ax.grid(True, which="minor", axis="x", alpha=0.6 * plt.rcParams["grid.alpha"])
            ax.set_xlabel(xlab)
            title = "Avg. Error vs Elevation" if x == "elev" else "Avg. Error vs Distance"
            ax.set_title(f"({letter}) {title}", loc="center", fontweight="bold")
        metric = {"xy": "Horizontal", "lat": "Lateral", "lon": "Longitudinal"}[CMP_METRIC]
        for ax in axes:
            ax.set_ylim(bottom=0)
        # Same number of decimals on every panel's y ticks, so the tick labels are the
        # same width and the y labels line up
        dp = 0
        for ax in axes:
            lo, hi = ax.get_ylim()
            for t in ax.get_yticks():
                if lo - 1e-9 <= t <= hi + 1e-9:
                    dp = max(dp, len(f"{t:.6g}".partition(".")[2]))
        for ax in axes:
            ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{dp}f"))
        ylab = f"{metric} RMSE (m)"
        if CMP_YLABEL == "shared":
            fig.supylabel(ylab, fontsize=plt.rcParams["axes.labelsize"])
        else:
            for ax in axes:
                if stacked or ax is axes[0] or not CMP_SHARE_Y:
                    ax.set_ylabel(ylab)
            fig.align_ylabels(list(axes))

        out_dir.mkdir(parents=True, exist_ok=True)
        labels = [(h[-1] if isinstance(h, tuple) else h).get_label() for h in handles]
        ncol = 2 if stacked else len(handles)     # 4 entries do not fit one row at 3.5 in
        if not LEGEND_SEPARATE:
            place_legend(fig, handles, labels, ncol, loc="lower center",
                         bbox_to_anchor=(0.5, 1.0))
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{CMP_NAME}.{ext}", dpi=600, bbox_inches="tight")
        plt.close(fig)
        if LEGEND_SEPARATE:
            fl = plt.figure(figsize=(width, 2.0))
            leg = place_legend(fl, handles, labels, ncol, loc="center", borderaxespad=0.0)
            h = leg.get_window_extent().height / fl.dpi     # fit the height to the legend
            fl.set_size_inches(width, h + 2 * plt.rcParams["savefig.pad_inches"] + 0.02)
            for ext in ("pdf", "png"):                      # exact width: no tight crop
                fl.savefig(out_dir / f"{CMP_NAME}_legend.{ext}", dpi=600, bbox_inches=None)
            plt.close(fl)

    pd.DataFrame(summary).to_csv(out_dir / f"{CMP_NAME}.csv", index=False)
    print(f"  {out_dir / CMP_NAME}.pdf/.png (+ _legend, .csv)")
    if CMP_BALANCED:
        print(f"    elevation panel pools R = {keep['elev']}, range panel pools elev = {keep['R']}")


def conditions():
    """SWEEP-TEMP: folders to plot, honouring SWEEP_OUT when it is set."""
    if "SWEEP_OUT" in os.environ:
        return [OUT_DIR]
    dirs = sorted(d for d in SWEEP_ROOT.iterdir() if (d / "manifest.csv").exists())
    if not dirs and (SWEEP_ROOT / "manifest.csv").exists():
        dirs = [SWEEP_ROOT]                       # old single-condition layout
    return dirs


def ensure_metrics(d, force=False):
    """Run analyse_sweep.py for folder d if its metrics are missing, stale or forced."""
    m = d / "metrics_geom.csv"
    if not force and m.exists() and m.stat().st_mtime >= (d / "manifest.csv").stat().st_mtime:
        return True
    print(f"  analysing {d.name} ...")
    r = subprocess.run([sys.executable, str(HERE / "analyse_sweep.py")],
                       env={**os.environ, "SWEEP_OUT": str(d)})
    return r.returncode == 0 and m.exists()


def main():
    global PLOT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotate", action="store_true", help="print RMSE (cm) in each fan cell")
    ap.add_argument("--reanalyse", action="store_true", help="re-run analyse_sweep.py first")
    args = ap.parse_args()

    loaded = {}      # (folder, metrics suffix) -> (geom, case)
    for d in conditions():
        if not ensure_metrics(d, args.reanalyse):
            print(f"  skipping {d.name}: no usable metrics")
            continue
        for suffix in ("", "_naive"):
            g, c = d / f"metrics_geom{suffix}.csv", d / f"metrics_case{suffix}.csv"
            if g.exists() and c.exists():
                loaded[(d, suffix)] = (pd.read_csv(g), pd.read_csv(c))
    if not loaded:
        raise SystemExit(f"No condition folders with data under {SWEEP_ROOT}")

    # Shared RMSE scale per height condition (suffix after the last "_")
    group = lambda d: d.name.rsplit("_", 1)[-1] if "_" in d.name else d.name
    vmax = {}
    for (d, _), (geom, _) in loaded.items():
        v = np.nanmax(geom[["rmse_lat", "rmse_lon"]].to_numpy()) if len(geom) else np.nan
        if np.isfinite(v):
            vmax[group(d)] = max(vmax.get(group(d), 0.0), v)

    for (d, suffix), (geom, case) in loaded.items():
        PLOT_DIR = d / f"plots{suffix}"
        print(f"{d.name}{suffix}: writing figures")
        fan_rmse(geom, args.annotate, vmax.get(group(d)))
        fan_detection(geom, args.annotate)
        aspect_bias(case)
        aspect_bias_by_elev(case)

    # Every condition on one figure (needs the per-condition folders under sweep_out/)
    print("comparison: writing figures")
    error_vs_geometry(loaded, SWEEP_ROOT / "plots_compare")


if __name__ == "__main__":
    main()