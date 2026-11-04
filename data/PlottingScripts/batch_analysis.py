"""
batch_analysis.py  –  Trial-set statistics for the vision landing runs
=====================================================================
Wraps plot_errors.py (keep both in the same folder) and runs every controller CSV
written by run_batch.sh, laid out as

    <root>/<study>/<trajectory>/<NN>kmh/repNN/controller_*.csv

Each study (the name passed to run_batch.sh) is one mode, e.g.

    <root>/main/...       UKF with OOSM latency compensation
    <root>/no_oosm/...    UKF with OOSM disabled (same sweep, launch param changed)

Only finished runs (a DONE marker in the rep folder) are read, so it is safe to run
while a batch is still going. Every trajectory is analysed separately.

Usage:
    Set ROOT / MODES (and optionally TRAJECTORIES / OUT_DIR / PER_RUN_PLOTS) below, then
        python batch_analysis.py
    Command-line arguments override the config:
        python batch_analysis.py ~/sitl_results --trajectories straight sin --per-run-plots

Per run it computes
    Estimation error (estimate - truth, car frame), scored over the window
        [start,  end of the first attempt: first 2xxx/5xxx/6xxx/7xxx after lock]
    i.e. once the filter has settled, up to the end of the FIRST landing attempt. Data
    from any re-attempt is ignored. The start is set by WINDOW_START:
        'converged'  the start of the first stretch after 3000 in which the UKF reports
                     itself converged (box update active, sigma_xy, sigma_yaw and sigma_v
                     under thresholds) continuously for at least CONV_DWELL_S. The dwell is
                     scored. Truth-free, so it can't cherry-pick good stretches.
                     Falls back to 'fixed' if it never converges (CONV_FALLBACK_TO_FIXED).
        'fixed'      first 3000 + SETTLE_S
        'descent'    the first descent only: first 4000 to the first abort / touchdown /
                     RTL / loss of the pad after it (replaces the end above)
    Only rows where the pad is in view are scored (pad_found, latest YOLO frame at most
    MAX_MEAS_AGE_S old, raw fit finite), and the filter and the raw fit are scored on the
    same rows; tracked_frac in runs.csv is the share of the window kept. Otherwise the same
    scoring as plot_errors.py:
        filt   filter estimate at the UKF tick it is valid at (est_time)
        held   YOLO car fit held until the next one, at the row time (the live competitor)
        stamp  YOLO car fit at its own image stamp
        post   filter state at the YOLO image stamp (NaN when OOSM is off)
    Landing outcome, from the drone's final resting pose (median of the last REST_AVG_S)
        success  = upright (FCU roll/pitch within UPRIGHT_MAX_TILT_DEG) AND its centre is
                   over the pad AND its height above the pad top is in REST_HEIGHT_RANGE
                   (rules out the hood, the roof behind the pad and the ground)
        landing error (long / lat / radial) from the pad centre, successful runs only
        Estimation-error statistics in summary.csv use successful runs only (n_scored);
        failed runs still count in the success rate and landing statistics.
        time to acquisition = first 4000 - first REF_STATE
        time to landing     = first 6000 - first REF_STATE  (includes any re-attempts)
        time detect to land = first 6000 - first 2100 (first YOLO sighting), in the table
        attempts            = number of entries into 4000

Outputs (in <root>/batch_results unless --out is given):
    runs_all.csv          one row per run, every trajectory, every metric below; shared
                          0 km/h copies (SHARE_STATIONARY) are flagged in shared_from
    summary_all.csv       every trajectory's summary.csv stacked, with a trajectory column
    landing_table.tex     compact LaTeX (booktabs) table, one row per trajectory and speed:
                          landing error, success rate, time to land (mean ± std); STANDALONE
                          trajectories get their own in their folder
    landing_table.csv     the same numbers, unrounded
    all_trajectories/
        (STANDALONE trajectories excluded; each trajectory named in bold above its row)
        rmse_vs_speed         3-D RMSE, a panel per trajectory stacked (single column;
                              RMSE_GRID)
        rmse_axes_vs_speed    a row per trajectory, columns lateral / longitudinal /
                              height (full width; RMSE_AXES_FIGURE)
        landing_vs_speed      a row per trajectory, columns success rate and landing
                              error (single column)
        landing_clusters      every speed pooled, a row per trajectory and a column per
                              mode (CLUSTER_POOL_SPEEDS, CLUSTER_GRID)
        trajectories_topdown  top-down car and drone paths, a panel per trajectory
    <trajectory>/
        runs.csv              one row per CSV, every metric above
        summary.csv           per speed and mode: mean / std over runs, paired filter-vs-raw
                              test, success rate with 95% Wilson interval
        mode_comparison.csv   per speed: first two modes' filter RMSE (Mann-Whitney U);
                              only written when at least two modes have data
        raw_pooled.csv        per speed: raw YOLO fit RMSE over the RMSE_MODES runs (with
                              several modes, a Kruskal-Wallis check that they agree)
        rmse_vs_speed         RMSE (RMSE_METRICS, default 3-D) vs speed: the filter
                              modes in RMSE_MODES plus the raw fit from the same runs
        rmse_axes_vs_speed    the same split into lateral / longitudinal / height
                              (RMSE_AXES_FIGURE)
        landing_vs_speed      success rate and landing error (successful landings) vs
                              speed, every mode; same size and style as rmse_vs_speed
        landing_clusters      successful landings in the pad frame, one panel per speed
                              in rows of CLUSTER_GROUP_SIZE, one shared legend
        landing_clusters_pooled  every speed pooled into one panel (CLUSTER_POOL_SPEEDS)
        paths_3d/             3-D flight paths, one figure per speed and frame (PATHS_3D):
                              every repeat overlaid, one panel per mode
        per_run/...           plot_errors.py figures for each run (only with --per-run-plots)
    Figures are sized for a two-column journal (FIG_WIDTH_*), styled by PAPER_RC and
    saved in SAVE_FORMATS.

Dependencies:
    pip install pandas numpy matplotlib scipy
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # batch job: save figures, never open windows
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
from matplotlib.patches import Ellipse, Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import stats
from scipy.interpolate import interp1d

import plot_errors as pe


# ── Config ─────────────────────────────────────────────────────────────────

# run_batch.sh results root (holds one folder per study)
ROOT = Path.home() / 'sitl_results'
OUT_DIR = None          # None -> ROOT/batch_results
PER_RUN_PLOTS = False   # True: also save plot_errors.py figures for every run (slow)

# Study folder (run_batch.sh study name) -> label. Studies not listed are skipped;
# listed studies that don't exist yet are noted and skipped. The first two are compared.
MODES = {
    'main':    'OOSM (latency compensated)',
    'no_oosm': 'No latency compensation',
}
TRAJECTORIES = None     # None -> every trajectory folder found, e.g. ['straight', 'sin']
ONLY_DONE = True        # skip rep folders without a DONE marker (in progress / interrupted)

# Colour, line style and marker per mode, assigned in MODES order
# Colours are stops on one perceptually uniform colormap, so they also stay distinct in
# greyscale (lightness rises along it). The yellow end (> ~0.85) is too pale on white.
COLOURMAP  = 'viridis'
CMAP_STOPS = {'main': 0.15, 'no_oosm': 0.55, 'raw': 0.75, 'car': 0.75}   # car: 3-D world paths


def cmap_colour(key, default=0.40):
    return mcolors.to_hex(matplotlib.colormaps[COLOURMAP](CMAP_STOPS.get(key, default)))


_MODE_LOOK = [('-', 'o'), ('--', 's'), ('-.', '^'), (':', 'D')]   # dash, marker per mode
MODE_COLOURS = {m: cmap_colour(m, 0.15 + 0.4 * i) for i, m in enumerate(MODES)}
MODE_STYLES  = {m: _MODE_LOOK[i % len(_MODE_LOOK)][0] for i, m in enumerate(MODES)}
MODE_MARKERS = {m: _MODE_LOOK[i % len(_MODE_LOOK)][1] for i, m in enumerate(MODES)}

# Start of the error window (end is always the end of the first attempt)
WINDOW_START = 'converged'   # 'converged' (filter's own covariance), 'fixed' (3000 +
                             # SETTLE_S) or 'descent' (first 4000 to touchdown / abort)
SETTLE_S     = 0.0           # 'fixed': s after the first 3000

# 'converged': all of these must hold continuously for CONV_DWELL_S
CONV_REQUIRE_BOX_MODE  = False   # UKF using the pixel-space box update (yaw known)
CONV_POS_SIGMA         = 0.70   # m, hypot(sigma_px, sigma_py) below this
CONV_YAW_SIGMA         = 1.00   # rad, sigma_yaw below this (box mode itself needs < 0.25)
CONV_VEL_SIGMA         = 0.80   # m/s, sigma_v (UKF speed state) below this; None: off.
                                # 0.90 is the controller's landing_confident_vel_sigma, which
                                # gates the full pad-velocity sigma (sigma_v plus heading and
                                # yaw-rate terms), so on sigma_v alone it is slightly looser
CONV_DWELL_S           = 3.0    # s the criteria must hold; the window starts where they began
CONV_MIN_DELAY_S       = 0.0    # s, never start earlier than first 3000 + this
CONV_FALLBACK_TO_FIXED = False   # never converged -> use 'fixed'; False -> run scores NaN
# Scored rows: the pad must actually be in view. A row counts only if the latest YOLO
# frame (by image stamp) is at most this old; pad_found alone stays 1 for the controller's
# 0.5 s found-timeout, so it would score coasting and a stale raw fit. Measured on a real
# log: YOLO ~12 Hz, age 120 ms median / 180 ms p95 (frame interval + pipeline latency),
# so 0.25 s keeps continuous tracking and drops any gap of two or more missed frames.
MAX_MEAS_AGE_S = 0.25
REF_STATE = 3000     # reference for time to acquisition / landing (2100 = first sighting)

# Car geometry, car base_link frame (x fwd, y left, z up). Keep equal to the launch file.
PAD_POSITION  = (-0.5, 0.0, 1.525)          # pad top-centre (yolo_car_pad_position)
PAD_SIZE      = (1.80, 1.48)                # length x width (landing_pad_size)
CAR_FOOTPRINT = (-2.31, 2.24, -0.89, 0.89)  # x_min, x_max, y_min, y_max (body box)

# Successful landing: final resting pose
REST_AVG_S           = 0.5            # s, median over the end of the log
UPRIGHT_MAX_TILT_DEG = 30.0           # max tilt from FCU roll/pitch
REST_HEIGHT_RANGE    = (-0.15, 0.60)  # m, drone base_link above the pad top
PAD_EDGE_MARGIN      = 0.0            # m, shrink the pad when checking "centre over pad"

# ── Figures and table ──
SPEED_UNIT  = 'km/h'   # label only: speeds are read from the NNkmh folder names as-is
SPEED_TICKS = (10, 5)  # speed axes: labelled major tick and faint minor gridline spacing
LANDING_SPEED_TICKS = (20, 10)   # the same for the landing-performance figures

# RMSE vs speed: these modes plus the raw YOLO fit from the same runs (pooled over them
# if there are several, with a consistency check in the printout). The LaTeX table always
# reports every mode.
RMSE_MODES   = ['main', 'no_oosm']
# Error metric(s), one panel each: '3d' (|long, lat, height|), 'xy' (horizontal only), or
# per axis 'lat' / 'long' / 'z'. NOTE: the raw fit's height is NOT independent of the
# filter (its plane height tracks the filter's pad height in box mode), so '3d' and 'z'
# flatter the raw baseline; 'xy' is the clean filter-vs-raw comparison until a rerun.
RMSE_METRICS = ['3d']
RMSE_GRID    = 'stacked'   # all_trajectories RMSE with one metric: 'stacked' = a row per
                           # trajectory (single column), 'row' = side by side (full width)
RMSE_AXES_FIGURE = True    # also rmse_axes_vs_speed: lateral / longitudinal / height
                           # (1x3 per trajectory, a row per trajectory in all_trajectories)

# Landing clusters (successful landings): view = pad + margin, rows of up to N speeds
CLUSTER_MARGIN     = 0.0    # m beyond the pad edge; 0 = axes exactly at the pad boundary
CLUSTER_GROUP_SIZE = 5      # 4 gives bigger panels
CLUSTER_ELLIPSES   = False   # 1σ ellipse per mode on every cluster figure
CLUSTER_POOL_SPEEDS = True  # also every speed pooled into one panel per trajectory:
                            # <trajectory>/landing_clusters_pooled, and the trajectories
                            # side by side in all_trajectories/landing_clusters
CLUSTER_POOL_COLOUR = 'speed'   # pooled figures: 'speed' = colour by speed (COLOURMAP), a
                                # panel per mode and trajectory; 'mode' = modes overlaid
CLUSTER_GRID = 'traj_rows'      # 'speed' grid: 'traj_rows' = a row per trajectory, a column
                                # per mode (single column); 'mode_rows' = the transpose

# LaTeX table (<out>/landing_table.tex): one row per trajectory and speed with landing
# error, success rate and time to land
TABLE_MODES     = ['main', 'no_oosm']   # modes compared, side by side under each metric
TABLE_MODE_SHORT = {'main': 'OOSM', 'no_oosm': 'No OOSM'}   # sub-column headers
TABLE_MODE_NOTE  = ('OOSM / No OOSM: UKF with / without out-of-sequence measurement '
                    '(latency) compensation.')               # caption, when comparing modes
STATIONARY_ROW  = True       # 0 km/h reported once as 'Stationary', not under every trajectory
TABLE_ENV       = None       # None: 'table*' (full width) for a mode comparison, 'table'
                             # (one column) for a single mode; or force either
TABLE_LABEL     = 'tab:landing_performance'
TABLE_DECIMALS  = dict(error=2, time=1, pct=0)
TABLE_BOLD_BEST = True       # with several modes: the better value per row in bold
TABLE_FONT      = '\\small'   # '' for normalsize (IEEE / Elsevier page widths take either)
TABLE_COLSEP_PT = 4          # LaTeX default is 6 pt

# Display names, used in the table caption; other trajectories show their folder name
TRAJ_LABELS = {'straight': '(a) Straight', 'sin': '(b) Sinusoidal', 'circle': '(c) Circular',
               'mbzirc': 'MBZIRC'}

# A 0 km/h run is the same experiment whichever trajectory folder it was run under (the
# car never moves), so its results are copied into every trajectory without its own.
# Copies are flagged (shared_from) and left out wherever trajectories are pooled.
SHARE_STATIONARY = True

# Standalone trajectories: analysed entirely on their own. They keep their own figures
# and get their own LaTeX table (in their folder), but are left out of the pooled table
# and the all_trajectories grid, and neither give nor receive shared 0 km/h runs.
# Matched case-insensitively against the folder name.
STANDALONE = ['mbzirc']

# Top-down trajectory figure: car (pad centre) and drone paths over the whole run, start
# and end marked; one representative run per trajectory (<trajectory>/trajectory_topdown,
# and the trajectories side by side in all_trajectories/trajectories_topdown)
PLOT_TOPDOWN  = True
TOPDOWN_MODE  = 'main'
TOPDOWN_SPEED = None    # km/h; None = the fastest speed with a successful landing

# 3-D flight paths: one figure per trajectory and speed, one panel per mode, every repeat
# overlaid (written to <trajectory>/paths_3d/<frame>_<NN>kmh)
PATHS_3D       = False
PATH3D_FRAMES  = ('world', 'pad')
#   'world'  drone and car (pad centre) in the simulator frame from first detection
#            (PATH3D_WORLD_START) to the end of the log; origin at the car's start, which
#            is the same every repeat, so repeats line up
#   'pad'    drone relative to the pad (car axes, height above the pad top) from first
#            pad lock: the approach and descent, every run converging on the origin
PATH3D_WORLD_START   = 2100           # first YOLO sighting
PATH3D_VIEW          = {'world': (25, -60), 'pad': (22, -128)}   # (elevation, azimuth) deg
PATH3D_MIN_ASPECT    = 0.3            # box sides relative to the longest horizontal one:
PATH3D_MAX_Z_ASPECT  = 0.6            #   no side under MIN, height capped at MAX_Z
                                      #   (axes are labelled, so the stretch is honest)
PATH3D_MAX_POINTS    = 1500           # per line, keeps the PDFs light
PATH3D_CAR_GLITCH_SPEED = 30.0        # m/s, despike the car's world track (> 2x top speed)

# Journal sizing. Widths are the usual two-column template sizes
# (IEEE: 3.5 in column / 7.16 in page; Elsevier: 90 mm / 190 mm = 3.54 / 7.48 in).
FIG_WIDTH_SINGLE = 3.5
FIG_WIDTH_DOUBLE = 7.16
SAVE_FORMATS     = ('pdf',)          # vector PDF for LaTeX; add 'png' for a quick look
# Legends shared by a figure's panels: True writes each figure without its legend plus a
# <name>_legend.pdf to place once (e.g. above a row of subfigures); False draws it on top
LEGEND_SEPARATE  = True
LEGEND_BOX       = True    # thin frame around every legend
LEGEND_EDGE      = '0.8'   # frame colour (grey level 0 black to 1 white, or any colour)
LEGEND_ROUND     = True    # rounded corners
LEGEND_FONTSIZE  = 7.5     # pt, legend text (independent of PAPER_RC)
# A separate legend file is saved at exactly its figure's width (height fitted), so
# \includegraphics[width=<same width as the figure>] keeps the text at LEGEND_FONTSIZE.
# A tight crop would make it narrower, and scaling it up to the column would enlarge the
# font. If the entries do not fit in one row, they wrap onto more rows automatically.
# Legend labels for the figures (MODES labels are still used in the printout)
LABELS    = {'main': 'UKF, with Latency Compensation', 'no_oosm': 'UKF, no Latency Compensation'}
RAW_LABEL = 'Raw Measurement'
RAW_C     = cmap_colour('raw')

# Figure style: the one place to change how the figures look. Every line width, marker
# and font size in the figures is a multiple of these values.
BAND_ALPHA = 0.18       # opacity of the ±1σ bands that replace error bars
PAPER_RC = {
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 9.5,
    'lines.linewidth': 0.8, 'lines.markersize': 2.6, 'errorbar.capsize': 1.5,
    'axes.linewidth': 0.5, 'grid.linewidth': 0.3, 'grid.alpha': 0.3,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'xtick.minor.size': 1.2, 'xtick.minor.width': 0.4,
    'xtick.major.pad': 2.0, 'ytick.major.pad': 2.0, 'axes.labelpad': 2.0,
    'axes.titlepad': 3.0,
    'legend.frameon': False, 'legend.handlelength': 1.8, 'legend.columnspacing': 1.2,
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed TrueType fonts (journal requirement)
    'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
}

PER_RUN_DPI = 200    # plot_errors.py figures in --per-run-plots (its own default is 500)
DPI = 600            # PNG only; journals ask for >= 600 dpi on line art


# ── Helpers ────────────────────────────────────────────────────────────────

def rms(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else np.nan


def mean(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else np.nan


def parse_speed(name):
    """ '35kmh' -> 35.0, '05kmh' -> 5.0, 'spd_7p5' / 'spd_7.5' -> 7.5, else NaN. """
    m = re.search(r'(\d+(?:[p.]\d+)?)', name)
    return float(m.group(1).replace('p', '.')) if m else np.nan


def first_time(t, mask):
    idx = np.flatnonzero(mask)
    return float(t[idx[0]]) if len(idx) else np.nan


def wilson(k, n, z=1.96):
    """ 95% Wilson score interval for a success proportion (better than k/n +- at n=10). """
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def paired_stats(a, b):
    """ Paired difference a - b over runs: mean, 95% t-interval half width, Wilcoxon p.
        Wilcoxon needs n >= 6 to be able to reach p < 0.05 at all. """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    n = len(d)
    if n < 2:
        return np.nan, np.nan, np.nan, n
    ci = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    p = np.nan
    if n >= 6 and np.any(d != 0):
        p = float(stats.wilcoxon(d).pvalue)
    return float(d.mean()), float(ci), p, n


def improvement_stats(filt, raw):
    """ Relative improvement of the filter over the raw baseline, from per-run ratios
        r = RMSE_filter / RMSE_raw. Averaged on the log scale so a halving and a
        doubling cancel (a plain mean of ratios is biased upwards):
            improvement = 1 - exp(mean ln r)       (+ = filter better)
        Returns (improvement %, 95% CI low %, 95% CI high %, Wilcoxon p on ln r, n).
    """
    filt, raw = np.asarray(filt, float), np.asarray(raw, float)
    ok = np.isfinite(filt) & np.isfinite(raw) & (filt > 0) & (raw > 0)
    L = np.log(filt[ok] / raw[ok])
    n = len(L)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, n
    m = L.mean()
    ci = stats.t.ppf(0.975, n - 1) * L.std(ddof=1) / np.sqrt(n)
    p = float(stats.wilcoxon(L).pvalue) if n >= 6 and np.any(L != 0) else np.nan
    # Larger ln r = worse filter, so the CI ends swap when converted to improvement
    return (100 * (1 - np.exp(m)), 100 * (1 - np.exp(m + ci)), 100 * (1 - np.exp(m - ci)), p, n)


# ── Single run ─────────────────────────────────────────────────────────────

def converged_time(df, t, t_3000, t_end):
    """ Start of the first stretch after pad lock in which the UKF's own convergence
        criteria hold continuously for at least CONV_DWELL_S. The window starts there, so
        the dwell is scored too. Uses only what the filter knew, never the truth. NaN if
        no stretch that long completes before t_end.
    """
    if not np.isfinite(t_3000):
        return np.nan
    ok = np.ones(len(t), dtype=bool)
    if CONV_REQUIRE_BOX_MODE and 'yolo_box_mode' in df.columns:
        ok &= df['yolo_box_mode'].to_numpy() == 1
    if {'sigma_px', 'sigma_py'} <= set(df.columns):
        ok &= np.hypot(df['sigma_px'].to_numpy(float), df['sigma_py'].to_numpy(float)) < CONV_POS_SIGMA
    if 'sigma_yaw' in df.columns:
        ok &= df['sigma_yaw'].to_numpy(float) < CONV_YAW_SIGMA
    if CONV_VEL_SIGMA is not None and 'sigma_v' in df.columns:
        ok &= df['sigma_v'].to_numpy(float) < CONV_VEL_SIGMA

    t_min = t_3000 + CONV_MIN_DELAY_S
    run_start = None
    for k in np.flatnonzero((t >= t_3000) & (t <= t_end)):
        if not ok[k]:
            run_start = None
            continue
        if run_start is None:
            run_start = t[k]
        if t[k] - run_start >= CONV_DWELL_S and t[k] >= t_min:
            return float(max(run_start, t_min))
    return np.nan


def decimate(*arrays, n_max=None):
    """ Evenly thin equal-length arrays to at most n_max points, keeping both ends. """
    n_max = n_max or PATH3D_MAX_POINTS
    n = len(arrays[0])
    if n <= n_max:
        return arrays
    idx = np.unique(np.linspace(0, n - 1, n_max).round().astype(int))
    return tuple(a[idx] for a in arrays)


def flight_paths(df, t, state, sl, rel_x, rel_y, cos_yaw, sin_yaw):
    """ Drone (and car) paths for the 3-D figures.
        pad:   drone relative to the pad centre in car axes, from pad lock (rows sl),
               built from the same cleaned truth the error scoring uses
        world: drone and pad-centre tracks in the simulator frame from first detection
        topdown: the same tracks (x, y) over the whole log, for the top-down figure
    """
    out = {}
    if sl.any() and not sl.all():   # sl is every row when there was no pad lock
        h = -df['landing_pad_rel_true_z'].to_numpy(float)[sl] - PAD_POSITION[2]
        long = -(cos_yaw * rel_x + sin_yaw * rel_y)    # drone - pad, rotated into car axes
        lat = sin_yaw * rel_x - cos_yaw * rel_y
        out['pad'] = decimate(long, lat, h)

    need = ['quad_true_x', 'quad_true_y', 'quad_true_z', 'landing_pad_glob_true_x',
            'landing_pad_glob_true_y', 'landing_pad_glob_true_z', 'landing_pad_glob_true_yaw']
    t_start = first_time(t, state == PATH3D_WORLD_START)
    if set(need) <= set(df.columns):
        cx, cy, _ = pe.clean_true_position(df['landing_pad_glob_true_x'].to_numpy(),
                                           df['landing_pad_glob_true_y'].to_numpy(), t,
                                           thresh=PATH3D_CAR_GLITCH_SPEED)
        cyaw, _ = pe.clean_true_yaw(df['landing_pad_glob_true_yaw'].to_numpy(), t)
        c, s = np.cos(cyaw), np.sin(cyaw)
        px = cx + PAD_POSITION[0] * c - PAD_POSITION[1] * s
        py = cy + PAD_POSITION[0] * s + PAD_POSITION[1] * c
        pz = df['landing_pad_glob_true_z'].to_numpy(float) + PAD_POSITION[2]
        x0, y0 = px[0], py[0]
        qx, qy = df['quad_true_x'].to_numpy(float), df['quad_true_y'].to_numpy(float)
        out['topdown'] = dict(car=decimate(px - x0, py - y0), drone=decimate(qx - x0, qy - y0))
        if not np.isfinite(t_start):
            return out
        w = t >= t_start
        out['world'] = dict(
            drone=decimate(df['quad_true_x'].to_numpy(float)[w] - x0,
                           df['quad_true_y'].to_numpy(float)[w] - y0,
                           df['quad_true_z'].to_numpy(float)[w]),
            car=decimate(px[w] - x0, py[w] - y0, pz[w]))
    return out


def analyse_run(csv_path):
    """ All metrics for one controller CSV. Returns (metrics dict, scoring window,
        flight paths for the 3-D figures). """
    df = pd.read_csv(csv_path)
    t0 = float(df['timestamp'].iloc[0])
    t = df['timestamp'].to_numpy(dtype=float) - t0
    state = pd.to_numeric(df['controller_state'], errors='coerce').to_numpy()
    group = state // 1000

    out = {}

    # ── State timeline ──
    t_detect = first_time(t, state == 2100)   # first YOLO sighting of the pad (2000 -> 2100)
    t_ref = first_time(t, state == REF_STATE) if REF_STATE % 1000 \
        else first_time(t, group == REF_STATE // 1000)
    t_3000 = first_time(t, group == 3)
    t_4000 = first_time(t, group == 4)
    t_6000 = first_time(t, group == 6)
    after_lock = np.isfinite(t_3000) & (t >= t_3000)
    # First attempt ends at the first abort (5xxx), landing (6xxx), RTL (7xxx) or loss of the
    # pad (back to 2xxx) after lock, so a run that loses the car isn't scored to the end of the log
    t_first_end = first_time(t, after_lock & np.isin(group, (2, 5, 6, 7)))

    entries_4000 = (group == 4) & np.r_[True, group[1:] != group[:-1]]
    out['attempts'] = int(entries_4000.sum())
    out['reached_6000'] = bool(np.isfinite(t_6000))
    out['t_acquisition'] = t_4000 - t_ref
    out['t_landing'] = t_6000 - t_ref
    out['t_detect_land'] = t_6000 - t_detect   # first detection to touchdown (incl. re-attempts)

    # ── Scoring window: settled tracking up to the end of the first attempt ──
    w_end = t_first_end if np.isfinite(t_first_end) else t[-1]
    t_conv = converged_time(df, t, t_3000, w_end)   # always computed: also a result in its own right
    out['t_converged'] = t_conv - t_3000   # time from pad lock to filter convergence
    out['t_converged_ref'] = t_conv - t_ref   # same origin as t_acquisition / t_landing
    if WINDOW_START == 'descent':
        # The first descent only: from the first entry into 4xxx to the first abort,
        # landing, RTL or loss of the pad after it (scored even if lock was lost earlier)
        w_start = t_4000
        if np.isfinite(t_4000):
            t_d_end = first_time(t, (t > t_4000) & np.isin(group, (2, 5, 6, 7)))
            w_end = t_d_end if np.isfinite(t_d_end) else t[-1]
        out['window_rule'] = 'descent' if np.isfinite(t_4000) else 'no descent'
    elif WINDOW_START == 'converged' and np.isfinite(t_conv):
        w_start, out['window_rule'] = t_conv, 'converged'
    elif WINDOW_START == 'fixed' or CONV_FALLBACK_TO_FIXED:
        w_start = t_3000 + SETTLE_S
        out['window_rule'] = 'fixed' if WINDOW_START == 'fixed' else 'fixed (not converged)'
    else:
        w_start, out['window_rule'] = np.nan, 'not converged'
    out['window_start'], out['window_end'] = w_start, w_end
    out['window_s'] = w_end - w_start if np.isfinite(w_start) else np.nan
    window = (w_start, w_end)

    # ── Truth (same cleaning and pad offset as plot_errors.py) ──
    # Cleaned from pad lock onwards only: the glitch filter flags relative speeds above
    # OUTLIER_SPEED_THRESH, which the approach legitimately exceeds at high car speed.
    sl = (t >= t_3000) if np.isfinite(t_3000) else np.ones(len(t), bool)
    ts = t[sl]
    true_x, true_y, _ = pe.clean_true_position(
        df['landing_pad_rel_true_x'].to_numpy()[sl], df['landing_pad_rel_true_y'].to_numpy()[sl], ts)
    true_yaw, _ = pe.clean_true_yaw(df['landing_pad_glob_true_yaw'].to_numpy()[sl], ts)
    true_z = df['landing_pad_rel_true_z'].to_numpy(dtype=float)[sl] + pe.TRUE_OFFSET_Z_BODY
    c, s = np.cos(true_yaw), np.sin(true_yaw)
    true_x = true_x + pe.TRUE_OFFSET_X_BODY * c - pe.TRUE_OFFSET_Y_BODY * s
    true_y = true_y + pe.TRUE_OFFSET_X_BODY * s + pe.TRUE_OFFSET_Y_BODY * c
    truth = interp1d(ts, np.vstack([true_x, true_y, true_z, c, s]), kind='linear', axis=1,
                     bounds_error=False, fill_value='extrapolate')
    path = flight_paths(df, t, state, sl, true_x, true_y, c, s) \
        if (PATHS_3D or PLOT_TOPDOWN) else {}

    def car_frame(tq, ex, ey):
        tx, ty, tz, cq, sq = truth(tq)
        dx, dy = ex - tx, ey - ty
        return cq * dx + sq * dy, -sq * dx + cq * dy, tz

    def score(prefix, long, lat, z=None):
        out[f'{prefix}_long'] = rms(long)
        out[f'{prefix}_lat'] = rms(lat)
        out[f'{prefix}_xy'] = rms(np.hypot(long, lat))
        out[f'{prefix}_lat_bias'] = mean(lat)
        out[f'{prefix}_long_bias'] = mean(long)
        out[f'{prefix}_z'] = rms(z) if z is not None else np.nan
        out[f'{prefix}_3d'] = rms(np.sqrt(long ** 2 + lat ** 2 + z ** 2)) if z is not None else np.nan
        out[f'{prefix}_n'] = int(np.isfinite(lat).sum())

    in_win = (t >= w_start) & (t <= w_end) if np.isfinite(w_start) else np.zeros(len(t), bool)

    # Score only while the pad is in view, and score the filter and the raw fit on the
    # SAME rows, so the comparison is like for like:
    #   pad_found         the controller has a detection (within its found-timeout)
    #   fresh measurement the latest YOLO frame is at most MAX_MEAS_AGE_S old, so a
    #                     dropout is excluded rather than scoring coasting vs a stale fit
    #   raw fit exists    the YOLO car fit is finite (NaN when a box-mode shadow fit fails)
    n_win = int(in_win.sum())
    if 'pad_found' in df.columns:
        in_win &= pd.to_numeric(df['pad_found'], errors='coerce').to_numpy() == 1
    pre = 'landing_pad_yolo_car'
    has_raw = f'{pre}_raw_stamp' in df.columns
    if has_raw:
        stamp = df[f'{pre}_raw_stamp'].to_numpy(dtype=float)
        rx = df[f'{pre}_rel_x_raw'].to_numpy(dtype=float)
        ry = df[f'{pre}_rel_y_raw'].to_numpy(dtype=float)
        rz = df[f'{pre}_rel_z_raw'].to_numpy(dtype=float)
        t_meas = np.where(stamp > 0, stamp - t0, np.nan)
        age = t - t_meas                                   # NaN before the first frame
        in_win &= (age <= MAX_MEAS_AGE_S) & np.isfinite(rx) & np.isfinite(ry)
    out['tracked_frac'] = in_win.sum() / n_win if n_win else np.nan   # share of window scored

    # Filter estimate, valid at the UKF tick (est_time), not the log row
    est_raw = pd.to_numeric(df.get('est_time', pd.Series(np.zeros(len(t)))),
                            errors='coerce').to_numpy(dtype=float)
    est_t = np.where(est_raw > 0, est_raw - t0, t)
    m = in_win
    long, lat, tz = car_frame(est_t[m], df['landing_pad_rel_x'].to_numpy(float)[m],
                              df['landing_pad_rel_y'].to_numpy(float)[m])
    score('filt', long, lat, df['landing_pad_rel_z'].to_numpy(float)[m] - tz)

    # YOLO car fit (fallback fit or shadow fit): held to the row time on the same rows as
    # the filter (the live competitor), and at its own image stamp
    if has_raw:
        long, lat, tz = car_frame(t[m], rx[m], ry[m])
        score('held', long, lat, rz[m] - tz)

        new = pe.new_measurement_mask(stamp) & np.isfinite(rx) & np.isfinite(ry) \
            & (t_meas >= w_start) & (t_meas <= w_end)
        long, lat, _ = car_frame(t_meas[new], rx[new], ry[new])
        score('stamp', long, lat)

        if 'yolo_post_rel_x' in df.columns:
            px = df['yolo_post_rel_x'].to_numpy(dtype=float)
            py = df['yolo_post_rel_y'].to_numpy(dtype=float)
            ok = pe.new_measurement_mask(stamp) & np.isfinite(px) & np.isfinite(py) \
                & (t_meas >= w_start) & (t_meas <= w_end)
            long, lat, _ = car_frame(t_meas[ok], px[ok], py[ok])
            score('post', long, lat)

    # ── Final resting pose in the pad frame ──
    # Logged relative truth is car base_link minus drone, in the world-aligned frame
    rest = t >= t[-1] - REST_AVG_S
    yaw_c = df['landing_pad_glob_true_yaw'].to_numpy(dtype=float)[rest]
    dx = -df['landing_pad_rel_true_x'].to_numpy(dtype=float)[rest]
    dy = -df['landing_pad_rel_true_y'].to_numpy(dtype=float)[rest]
    dz = -df['landing_pad_rel_true_z'].to_numpy(dtype=float)[rest]
    cr, sr = np.cos(yaw_c), np.sin(yaw_c)
    out['rest_long'] = float(np.median(cr * dx + sr * dy)) - PAD_POSITION[0]
    out['rest_lat'] = float(np.median(-sr * dx + cr * dy)) - PAD_POSITION[1]
    out['rest_height'] = float(np.median(dz)) - PAD_POSITION[2]
    out['rest_error'] = float(np.hypot(out['rest_long'], out['rest_lat']))

    if {'quad_roll', 'quad_pitch'} <= set(df.columns):
        r = df['quad_roll'].to_numpy(dtype=float)[rest]
        p = df['quad_pitch'].to_numpy(dtype=float)[rest]
        out['rest_tilt_deg'] = float(np.median(np.degrees(np.arccos(
            np.clip(np.cos(r) * np.cos(p), -1.0, 1.0)))))
    else:
        out['rest_tilt_deg'] = np.nan

    half_l = 0.5 * PAD_SIZE[0] - PAD_EDGE_MARGIN
    half_w = 0.5 * PAD_SIZE[1] - PAD_EDGE_MARGIN
    out['upright'] = bool(not (out['rest_tilt_deg'] > UPRIGHT_MAX_TILT_DEG))  # NaN -> unknown, pass
    out['over_pad'] = bool(abs(out['rest_long']) <= half_l and abs(out['rest_lat']) <= half_w)
    out['height_ok'] = bool(REST_HEIGHT_RANGE[0] <= out['rest_height'] <= REST_HEIGHT_RANGE[1])
    out['success'] = out['upright'] and out['over_pad'] and out['height_ok']
    out['first_attempt_success'] = out['success'] and out['attempts'] == 1
    return out, window, path


# ── Aggregation ────────────────────────────────────────────────────────────

ERROR_KEYS = [f'{src}_{f}' for src in ('filt', 'held', 'stamp', 'post')
              for f in ('lat', 'long', 'xy', 'z', '3d', 'lat_bias')]


def summarise(runs):
    rows = []
    for (speed, mode), g in runs.groupby(['speed', 'mode'], sort=True):
        # Estimation error over successful landings only; failed runs still count in the
        # success rate and landing statistics below
        gs = g[g['success']]
        r = dict(speed=speed, mode=mode, n_runs=len(g), n_scored=len(gs))
        for k in ERROR_KEYS:
            if k in gs:
                r[f'{k}_mean'], r[f'{k}_std'] = gs[k].mean(), gs[k].std(ddof=1)

        # Paired, per run: raw held minus filter (positive = filter better)
        for f in ('lat', 'long', 'xy', 'z', '3d'):
            if f'held_{f}' in gs:
                d, ci, p, n = paired_stats(gs[f'held_{f}'], gs[f'filt_{f}'])
                r[f'gain_{f}_mean'], r[f'gain_{f}_ci95'], r[f'gain_{f}_wilcoxon_p'] = d, ci, p
                pct, lo, hi, p_log, _ = improvement_stats(gs[f'filt_{f}'], gs[f'held_{f}'])
                r[f'improve_{f}_pct'], r[f'improve_{f}_ci_lo'], r[f'improve_{f}_ci_hi'] = pct, lo, hi
                r[f'improve_{f}_wilcoxon_p'] = p_log

        k, n = int(g['success'].sum()), len(g)
        lo, hi = wilson(k, n)
        r.update(success_n=k, success_pct=100 * k / n, success_ci_lo=100 * lo, success_ci_hi=100 * hi,
                 first_attempt_success_pct=100 * g['first_attempt_success'].mean())

        ok = g[g['success']]
        for col in ('rest_error', 'rest_long', 'rest_lat'):
            r[f'{col}_mean'], r[f'{col}_std'] = ok[col].mean(), ok[col].std(ddof=1)
        for col in ('t_acquisition', 't_landing', 't_detect_land', 'attempts'):
            r[f'{col}_mean'], r[f'{col}_std'] = g[col].mean(), g[col].std(ddof=1)
        r['attempts_max'] = g['attempts'].max()
        for col in ('t_converged', 't_converged_ref'):
            r[f'{col}_mean'], r[f'{col}_std'] = g[col].mean(), g[col].std(ddof=1)
        r['n_not_converged'] = int(g['t_converged'].isna().sum())   # excluded from t_converged stats
        r['window_s_mean'] = gs['window_s'].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def compare_modes(runs):
    """ Per speed: filter RMSE with vs without latency compensation (different runs, so
        unpaired: Mann-Whitney U). """
    present = [m for m in MODES if m in set(runs['mode'])]
    if len(present) < 2:
        return pd.DataFrame()
    rows = []
    a_mode, b_mode = present[:2]
    for speed, g in runs.groupby('speed'):
        r = dict(speed=speed)
        for f in ('lat', 'long', 'xy'):
            a = g.loc[(g['mode'] == a_mode) & g['success'], f'filt_{f}'].dropna()
            b = g.loc[(g['mode'] == b_mode) & g['success'], f'filt_{f}'].dropna()
            r[f'{a_mode}_{f}'], r[f'{b_mode}_{f}'] = a.mean(), b.mean()
            r[f'{f}_mannwhitney_p'] = (stats.mannwhitneyu(a, b).pvalue
                                       if len(a) >= 3 and len(b) >= 3 else np.nan)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.attrs['modes'] = (a_mode, b_mode)
    return out


def pool_raw(runs):
    """ Raw YOLO fit RMSE per speed, pooled over every mode's successful runs.

        The raw fit never sees the filter, but it is not strictly mode-independent: the
        drone flies on the filter estimate (so range and viewing angle differ) and the
        scoring window starts at the filter's own convergence. Pooling is only fair if
        the modes' raw errors agree, so each speed and axis gets a Kruskal-Wallis test
        across modes (equivalent to Mann-Whitney U for two modes) and the largest
        relative gap between mode means.
    """
    present = [m for m in MODES if m in set(runs['mode'])]
    rows = []
    for speed, g in runs[runs['success']].groupby('speed'):
        r = dict(speed=speed, n=len(g))
        for f in ('lat', 'long', 'xy', 'z', '3d'):
            col = f'held_{f}'
            if col not in g:
                continue
            v = g[col].dropna()
            r[f'{col}_mean'], r[f'{col}_std'] = v.mean(), v.std(ddof=1)
            groups = [g.loc[g['mode'] == m, col].dropna() for m in present]
            means = [x.mean() for x in groups if len(x)]
            for m, x in zip(present, groups):
                r[f'{col}_{m}'] = x.mean() if len(x) else np.nan
            r[f'{col}_gap_pct'] = (100 * (max(means) - min(means)) / np.mean(means)
                                   if len(means) >= 2 else np.nan)
            testable = [x for x in groups if len(x) >= 3]
            try:
                r[f'{col}_kw_p'] = (float(stats.kruskal(*testable).pvalue)
                                    if len(testable) >= 2 else np.nan)
            except ValueError:   # every value identical
                r[f'{col}_kw_p'] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def print_raw_check(raw):
    """ Only meaningful when the raw line pools several modes (RMSE_MODES). """
    if raw.empty or not any(c.endswith('_kw_p') and raw[c].notna().any() for c in raw.columns):
        return
    print('\n' + '=' * 100)
    print('RAW FIT CONSISTENCY ACROSS MODES  (Kruskal-Wallis p; gap = spread of mode means / pooled mean)')
    print('=' * 100)
    flagged, n_tests = [], 0
    for _, r in raw.iterrows():
        parts = []
        for f in ('lat', 'long', 'z', '3d'):
            p = r.get(f'held_{f}_kw_p', np.nan)
            n_tests += np.isfinite(p)
            if np.isfinite(p) and p < 0.05:
                flagged.append(f"{r['speed']:g} {SPEED_UNIT} {f}")
            parts.append(f"{f} p = {'-' if not np.isfinite(p) else f'{p:.3f}'}"
                         f" (gap {r.get(f'held_{f}_gap_pct', np.nan):.0f}%)")
        print(f"{r['speed']:>6g} {SPEED_UNIT}   " + '   '.join(parts))
    if flagged:
        print(f"  NOTE: p < 0.05 for {', '.join(flagged)} ({n_tests} tests: expect about "
              f"{0.05 * n_tests:.1f} false flags by chance). If a gap is large or the flags "
              f"cluster, drop a mode from RMSE_MODES.")
    else:
        print("  No mode-dependence detected: pooling the raw series is justified.")


# ── Figures ────────────────────────────────────────────────────────────────
# Every size below is a multiple of a PAPER_RC value, so PAPER_RC alone sets the look.

def lw(k=1.0):
    """ Line width relative to PAPER_RC['lines.linewidth']. """
    return k * plt.rcParams['lines.linewidth']


def axlw(k=1.0):
    """ Line width relative to PAPER_RC['axes.linewidth'] (panel furniture, outlines). """
    return k * plt.rcParams['axes.linewidth']


def ms(k=1.0):
    """ Marker size relative to PAPER_RC['lines.markersize']. """
    return k * plt.rcParams['lines.markersize']


def fs(k=1.0):
    """ Font size relative to PAPER_RC['font.size']. """
    return k * plt.rcParams['font.size']


def label(mode):
    return LABELS.get(mode, MODES[mode])


def traj_order(values):
    """ Trajectories in TRAJ_LABELS order, then any others alphabetically. """
    present = set(values)
    return [t for t in TRAJ_LABELS if t in present] + sorted(present - set(TRAJ_LABELS))


def traj_label(traj):
    if traj in TRAJ_LABELS:
        return TRAJ_LABELS[traj]
    for k, v in TRAJ_LABELS.items():          # folder names may differ in case (MBZIRC)
        if k.lower() == traj.lower():
            return v
    return traj.replace('_', ' ').capitalize()


def is_standalone(traj):
    return traj.lower() in {x.lower() for x in STANDALONE}


def panel_title(ax, letter, text):
    ax.set_title(f'({letter}) {text}', loc='left')


def new_figure(nrows, ncols, figsize, pads, **kw):
    """ Constrained-layout subplots that work on matplotlib 3.4+ (ROS 2 Humble ships
        3.5, which predates the layout-engine API). """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, **kw)
    if hasattr(fig, 'get_layout_engine'):      # 3.6+
        fig.get_layout_engine().set(**pads)
    else:
        fig.set_constrained_layout_pads(**pads)
    return fig, axes


def handle_label(h):
    return (h[-1] if isinstance(h, tuple) else h).get_label()


def legend_kw():
    """ Shared legend look: LEGEND_FONTSIZE text, a thin frame (LEGEND_BOX, LEGEND_EDGE, LEGEND_ROUND). """
    return dict(fontsize=LEGEND_FONTSIZE, frameon=LEGEND_BOX, fancybox=LEGEND_ROUND,
                edgecolor=LEGEND_EDGE, framealpha=1.0, borderpad=0.4, handletextpad=0.5)


def place_legend(fig, handles, labels, ncol, **kw):
    """ fig.legend with the shared look, dropping columns until it fits the figure
        width (so a long-labelled legend wraps rather than spilling past the column). """
    ncol = ncol or len(handles)
    while True:
        leg = fig.legend(handles=handles, labels=labels, ncol=ncol, **legend_kw(), **kw)
        leg.get_frame().set_linewidth(axlw())
        fig.canvas.draw()
        if ncol == 1 or leg.get_window_extent().width <= fig.bbox.width - 2:
            return leg
        leg.remove()
        ncol -= 1


def top_legend(fig, handles, ncol=None):
    """ One shared legend for the figure. LEGEND_SEPARATE: kept aside and written by
        finish() as its own <name>_legend file. Otherwise centred just above all panels,
        anchored above the axes rather than inside the layout ('outside' locations need
        3.7), so it behaves the same on every version. A handle may be a tuple of artists
        drawn on top of each other (a band patch under its line). """
    labels = [handle_label(h) for h in handles]
    if LEGEND_SEPARATE:
        fig._shared_legend = (handles, labels, ncol)
        return
    fig.canvas.draw()   # settle constrained layout so the axes' top edge is final
    top = max(ax.get_tightbbox(fig.canvas.get_renderer()).y1 for ax in fig.axes
              if ax.get_visible()) / fig.bbox.height     # includes the panel titles
    place_legend(fig, handles, labels, ncol, loc='lower center',
                 bbox_to_anchor=(0.5, top + 0.01), borderaxespad=0.0)


def save_figure(fig, out_dir, name, tight=True):
    for ext in SAVE_FORMATS:
        path = out_dir / f'{name}.{ext}'
        fig.savefig(path, dpi=DPI, bbox_inches='tight' if tight else None)
        print(f"Saved: {path}")


def finish(fig, out_dir, name):
    spec, width = getattr(fig, '_shared_legend', None), fig.get_figwidth()
    save_figure(fig, out_dir, name)
    plt.close(fig)
    if spec:   # the legend on its own, as wide as the figure it belongs to
        handles, labels, ncol = spec
        fl = plt.figure(figsize=(width, 2.0))
        leg = place_legend(fl, handles, labels, ncol, loc='center', borderaxespad=0.0)
        h = leg.get_window_extent().height / fl.dpi        # fit the height to the legend
        fl.set_size_inches(width, h + 2 * plt.rcParams['savefig.pad_inches'] + 0.02)
        save_figure(fl, out_dir, f'{name}_legend', tight=False)   # exact figure width
        plt.close(fl)


def line_band(ax, x, y, s, colour, ls, marker):
    """ Mean line with a translucent ±1σ band (clipped at 0) in place of error bars.
        s None: line only. A missing σ (one run) gives no band at that point. """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y)
    ax.plot(x[ok], y[ok], color=colour, ls=ls, marker=marker, zorder=3)
    if s is not None and ok.any():
        s = np.nan_to_num(np.asarray(s, float))
        ax.fill_between(x[ok], np.clip(y - s, 0, None)[ok], (y + s)[ok], color=colour,
                        alpha=BAND_ALPHA, lw=0, zorder=2)


def band_handle(colour, ls, marker, text):
    """ Legend entry: the line drawn over a patch of its band. """
    return (Patch(facecolor=colour, alpha=BAND_ALPHA, lw=0),
            Line2D([], [], color=colour, ls=ls, marker=marker, label=text))


HEADING_H = 0.2   # in, height of a bold heading row above a row of panels


def add_heading(ax, text):
    ax.axis('off')
    ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes,
            fontsize=plt.rcParams['axes.titlesize'], fontweight='bold')


def headed_grid(width, ncols, panel_h, headings, share='all', tail=(), extra_h=0.55):
    """ Figure with a bold heading above each row of panels (the trajectory names).
        share: 'all' = one x and one y scale; 'col' = one x scale, a y scale per column.
        tail: heights (in) of extra full-width rows at the bottom (x label, colourbar),
        returned as axes. Returns (fig, axes[nrows][ncols], tail_axes). """
    heights = []
    for h in headings:
        if h:
            heights.append(HEADING_H)
        heights.append(panel_h)
    heights += list(tail)
    fig = plt.figure(figsize=(width, sum(heights) + extra_h), constrained_layout=True)
    pads = dict(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.02)
    if hasattr(fig, 'get_layout_engine'):
        fig.get_layout_engine().set(**pads)
    else:
        fig.set_constrained_layout_pads(**pads)
    gs = fig.add_gridspec(len(heights), ncols, height_ratios=heights)
    n = len(headings)
    axes = np.empty((n, ncols), dtype=object)
    r = 0
    for i, h in enumerate(headings):
        if h:
            add_heading(fig.add_subplot(gs[r, :]), h)
            r += 1
        for j in range(ncols):
            kw = {}
            if i or j:
                kw['sharex'] = axes[0][0]
            if share == 'all' and (i or j):
                kw['sharey'] = axes[0][0]
            elif share == 'col' and i:
                kw['sharey'] = axes[0][j]
            ax = fig.add_subplot(gs[r, j], **kw)
            if i < n - 1:
                ax.tick_params(labelbottom=False)
            if share == 'all' and j:
                ax.tick_params(labelleft=False)
            axes[i][j] = ax
        r += 1
    return fig, axes, [fig.add_subplot(gs[r + k, :]) for k in range(len(tail))]


def speed_axis(ax, speeds, ticks=None):
    """ Labelled ticks every ticks[0] (default SPEED_TICKS), minor ticks and a fainter
        gridline every ticks[1], so a dense sweep (every 5 km/h) stays readable. """
    major, minor = ticks or SPEED_TICKS
    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    pad = 0.5 * minor
    ax.set_xlim(min(speeds) - pad, max(speeds) + pad)
    ax.grid(True, which='major')
    ax.grid(True, which='minor', axis='x', alpha=0.6 * plt.rcParams['grid.alpha'])


def rescale_shared_columns(axes, bottoms=None, tops=None):
    """ Grids with sharey='col': fit each column's y range to the data of every row
        (setting ylim while drawing row by row would freeze it at the first rows'
        data and clip the later ones), then pin the bottom (default 0) and any top. """
    for j in range(axes.shape[1]):
        ax = axes[0][j]
        ax.set_autoscaley_on(True)
        ax.autoscale_view(scalex=False)
        ax.set_ylim(bottom=(bottoms or {}).get(j, 0), top=(tops or {}).get(j))


def cov_ellipse(ax, pts, colour, ls='-'):
    """ 1-sigma covariance ellipse of a set of successful landings. """
    if len(pts) < 3:
        return
    vals, vecs = np.linalg.eigh(np.cov(pts.T))
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(pts.mean(axis=0), 2 * np.sqrt(vals[1]), 2 * np.sqrt(vals[0]),
                         angle=ang, fill=False, ec=colour, ls=ls, lw=lw(0.9), zorder=3))


TARGET_C = '#d62728'   # the target point (pad centre) on the cluster plots


def fwd_arrow(ax):
    """ Direction of travel (up the page), drawn once in the first panel's corner. """
    ax.annotate('', xy=(0.10, 0.94), xytext=(0.10, 0.76), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='0.4', lw=axlw()), zorder=10)
    ax.text(0.14, 0.85, 'fwd', transform=ax.transAxes, ha='left', va='center',
            fontsize=fs(0.8), color='0.4', zorder=10)


def cluster_xy(df):
    """ Landing positions in the plotting frame: forward (longitudinal) up, the car's
        right to the right, i.e. a top-down view with the car driving up the page. """
    return -df['rest_lat'].to_numpy(float), df['rest_long'].to_numpy(float)


def draw_pad(ax, xlim, ylim):
    """ Pad-frame panel furniture (forward up, white background): a red cross at the target
        point (0, 0), equal axes and sparse ticks kept off the panel edges so
        neighbouring labels don't collide. """
    ax.plot(0, 0, 'x', ms=ms(1.6), color=TARGET_C, mew=lw(1.6), zorder=9)   # target
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    for axis, lim in ((ax.xaxis, xlim), (ax.yaxis, ylim)):
        ticks = np.arange(np.ceil(lim[0] / 0.5) * 0.5, lim[1] + 1e-9, 0.5)
        axis.set_major_locator(plt.FixedLocator(
            ticks[(ticks > lim[0] + 0.05) & (ticks < lim[1] - 0.05)]))


AXIS_METRICS = ['lat', 'long', 'z']
METRIC_NAMES = {'3d': '3-D', 'xy': 'Horizontal', 'lat': 'Lateral',
                'long': 'Longitudinal', 'z': 'Height'}


def rmse_series(summary, raw):
    """ (colour, dash, marker, legend label, rows, column prefix) for each line on an RMSE
        panel: every mode in RMSE_MODES, then the raw YOLO fit from the same runs. """
    modes = [m for m in RMSE_MODES if m in set(summary['mode'])]
    series = [(MODE_COLOURS[m], MODE_STYLES[m], MODE_MARKERS[m], label(m),
               summary[summary['mode'] == m], 'filt') for m in modes]
    if not raw.empty:
        series.append((RAW_C, ':', 'D', RAW_LABEL, raw, 'held'))
    return series


def draw_rmse_row(axes, series, speeds, metrics):
    """ One RMSE panel per metric: mean line with a ±1σ band over runs. """
    for ax, f in zip(axes, metrics):
        for c, ls, mk, _, g, src in series:
            g = g.sort_values('speed')
            if f'{src}_{f}_mean' not in g or not g[f'{src}_{f}_mean'].notna().any():
                continue
            line_band(ax, g['speed'], g[f'{src}_{f}_mean'], g[f'{src}_{f}_std'], c, ls, mk)
        ax.set_ylim(bottom=0)
        speed_axis(ax, speeds)


def rmse_legend(fig, series):
    top_legend(fig, [band_handle(c, ls, mk, lab) for c, ls, mk, lab, _, _ in series])


def plot_rmse(summary, raw, out_dir, metrics=None, name='rmse_vs_speed'):
    """ RMSE vs speed for one trajectory, one panel per metric (default RMSE_METRICS):
        each mode in RMSE_MODES plus the raw YOLO fit from the same runs, mean ± 1σ
        (band) over successful runs. One metric: a single-column figure. """
    metrics, speeds = metrics or RMSE_METRICS, sorted(summary['speed'].unique())
    k = len(metrics)
    fig, axes = new_figure(1, k, (FIG_WIDTH_SINGLE if k == 1 else FIG_WIDTH_DOUBLE,
                                  2.0 if k == 1 else 1.85),
                           dict(w_pad=0.03, h_pad=0.03, wspace=0.04), sharex=True,
                           squeeze=False)
    axes = axes[0]
    series = rmse_series(summary, raw)
    draw_rmse_row(axes, series, speeds, metrics)
    if k == 1:
        axes[0].set_ylabel(f'{METRIC_NAMES[metrics[0]]} RMSE (m)')
        axes[0].set_xlabel(f'Car speed ({SPEED_UNIT})')
    else:
        for ax, f in zip(axes, metrics):
            ax.set_title(f'{METRIC_NAMES[f]} RMSE (m)')
        fig.supxlabel(f'Car speed ({SPEED_UNIT})', fontsize=plt.rcParams['axes.labelsize'])
    rmse_legend(fig, series)
    finish(fig, out_dir, name)


def plot_rmse_grid(per_traj, out_dir, metrics=None, name='rmse_vs_speed'):
    """ Every trajectory in one figure, each named in bold above its panels.
        One metric: RMSE_GRID 'stacked' (a row per trajectory, single column, one y
        scale) or 'row' (side by side, full width). Several metrics: a row per
        trajectory, a column per metric, full width, a y scale per column.
        per_traj: {trajectory: (summary, raw)} """
    metrics, trajs = metrics or RMSE_METRICS, traj_order(per_traj)
    speeds = sorted(set().union(*(set(sm['speed']) for sm, _ in per_traj.values())))
    xlab = f'Car speed ({SPEED_UNIT})'
    if len(metrics) == 1 and RMSE_GRID == 'row':
        fig, axes = new_figure(1, len(trajs), (FIG_WIDTH_DOUBLE, 2.0),
                               dict(w_pad=0.03, h_pad=0.03, wspace=0.03), sharex=True,
                               sharey=True, squeeze=False)
        for ax, traj in zip(axes[0], trajs):
            draw_rmse_row([ax], rmse_series(*per_traj[traj]), speeds, metrics)
            ax.set_title(traj_label(traj), fontweight='bold')
        rescale_shared_columns(axes)
        axes[0][0].set_ylabel(f'{METRIC_NAMES[metrics[0]]} RMSE (m)')
    elif len(metrics) == 1:
        fig, axes, _ = headed_grid(FIG_WIDTH_SINGLE, 1, 1.15, [traj_label(t) for t in trajs])
        for i, traj in enumerate(trajs):
            draw_rmse_row([axes[i][0]], rmse_series(*per_traj[traj]), speeds, metrics)
        rescale_shared_columns(axes)
        fig.supylabel(f'{METRIC_NAMES[metrics[0]]} RMSE (m)',
                      fontsize=plt.rcParams['axes.labelsize'])
    else:
        fig, axes, _ = headed_grid(FIG_WIDTH_DOUBLE, len(metrics), 1.3,
                                   [traj_label(t) for t in trajs], share='col')
        for i, traj in enumerate(trajs):
            draw_rmse_row(axes[i], rmse_series(*per_traj[traj]), speeds, metrics)
        rescale_shared_columns(axes)
        for ax, f in zip(axes[0], metrics):
            ax.set_title(f'{METRIC_NAMES[f]} RMSE (m)')
    fig.supxlabel(xlab, fontsize=plt.rcParams['axes.labelsize'])
    rmse_legend(fig, rmse_series(*per_traj[trajs[0]]))
    finish(fig, out_dir, name)


def draw_landing_row(axes, runs, speeds):
    """ Success rate (all runs) and landing error (mean, ±1σ band over successful
        landings) vs speed, every mode, on a pair of axes. """
    for mode in [m for m in MODES if m in set(runs['mode'])]:
        g = runs[runs['mode'] == mode].groupby('speed')['success']
        pct = 100 * g.mean().reindex(speeds)
        err = runs[(runs['mode'] == mode) & runs['success']].groupby('speed')['rest_error']
        e = pd.DataFrame(dict(mean=err.mean(), std=err.std(ddof=1))).reindex(speeds)
        look = (MODE_COLOURS[mode], MODE_STYLES[mode], MODE_MARKERS[mode])
        line_band(axes[0], speeds, pct, None, *look)            # success: no band
        line_band(axes[1], speeds, e['mean'], e['std'], *look)
    for ax in axes:
        speed_axis(ax, speeds, LANDING_SPEED_TICKS)
    axes[0].set_ylim(0, 105)
    axes[1].set_ylim(bottom=0)


def mode_handles(modes):
    return [Line2D([], [], color=MODE_COLOURS[m], ls=MODE_STYLES[m], marker=MODE_MARKERS[m],
                   label=label(m)) for m in modes]


def plot_landing(runs, out_dir):
    """ (a) success rate over all runs (no error bar; the 95% Wilson interval is in
        summary.csv) and (b) landing error from the pad centre, mean ± 1σ over
        successful landings. Every mode, vs speed. Same canvas as the RMSE figure. """
    modes = [m for m in MODES if m in set(runs['mode'])]
    speeds = sorted(runs['speed'].unique())
    fig, axes = new_figure(1, 2, (FIG_WIDTH_DOUBLE, 1.85),     # same canvas as RMSE
                           dict(w_pad=0.03, h_pad=0.03, wspace=0.04), sharex=True)
    draw_landing_row(axes, runs, speeds)
    for letter, ax, name, ylab in zip('ab', axes, ('Success rate', 'Landing error'),
                                      ('Success (%)', 'Error (m)')):
        panel_title(ax, letter, name)
        ax.set_ylabel(ylab)
    fig.supxlabel(f'Car speed ({SPEED_UNIT})', fontsize=plt.rcParams['axes.labelsize'])
    top_legend(fig, mode_handles(modes))
    finish(fig, out_dir, 'landing_vs_speed')


def plot_landing_grid(per_traj, out_dir):
    """ Every trajectory's landing figure in one single-column figure: a row per
        trajectory (TRAJ_LABELS order, named in bold above the row), columns success
        rate and landing error. Each column shares its y axis, speed axes are shared
        across the grid. per_traj: {trajectory: runs} """
    trajs = traj_order(per_traj)
    speeds = sorted(set().union(*(set(r['speed']) for r in per_traj.values())))
    modes = [m for m in MODES if any(m in set(r['mode']) for r in per_traj.values())]
    fig, axes, _ = headed_grid(FIG_WIDTH_SINGLE, 2, 1.1, [traj_label(t) for t in trajs],
                               share='col')
    for i, traj in enumerate(trajs):
        draw_landing_row(axes[i], per_traj[traj], speeds)
    rescale_shared_columns(axes, tops={0: 105})
    axes[0][0].set_title('Success (%)')
    axes[0][1].set_title('Landing error (m)')
    fig.supxlabel(f'Car speed ({SPEED_UNIT})', fontsize=plt.rcParams['axes.labelsize'])
    top_legend(fig, mode_handles(modes))
    finish(fig, out_dir, 'landing_vs_speed')


def cluster_view():
    """ (xlim, ylim) of a cluster panel: lateral across, longitudinal up. """
    hlong, hlat = PAD_SIZE[0] / 2 + CLUSTER_MARGIN, PAD_SIZE[1] / 2 + CLUSTER_MARGIN
    return (-hlat, hlat), (-hlong, hlong)


def draw_cluster_panel(ax, runs, modes, xlim, ylim):
    """ Pad-frame panel: every mode's successful landings, 1σ ellipse and mean, plus its
        k/n successes stacked in the bottom-right corner in the mode colour. """
    draw_pad(ax, xlim, ylim)
    tally = []
    for mode in modes:
        g = runs[runs['mode'] == mode]
        ok = g[g['success']]
        c = MODE_COLOURS[mode]
        ax.scatter(*cluster_xy(ok), s=ms(0.86) ** 2, color=c, clip_on=False,
                   marker=MODE_MARKERS[mode], ec='white', lw=axlw(0.4), alpha=0.8, zorder=4)
        if len(ok):
            ax.plot(*(v.mean() for v in cluster_xy(ok)), marker=MODE_MARKERS[mode],
                    ms=ms(1.25), mfc=c, mec='black', mew=axlw(), ls='none', zorder=6)
        if CLUSTER_ELLIPSES:
            cov_ellipse(ax, np.column_stack(cluster_xy(ok)), c, ls=MODE_STYLES[mode])
        if len(g):
            tally.append((f'{len(ok)}/{len(g)}', c))
    for j, (txt, c) in enumerate(reversed(tally)):
        ax.text(0.93, 0.10 + 0.11 * j, txt, transform=ax.transAxes, ha='right',
                va='bottom', fontsize=fs(0.8), color=c, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75), zorder=10)


def cluster_handles(modes):
    if CLUSTER_ELLIPSES:   # line style in the legend = that mode's ellipse
        return mode_handles(modes) + [
            Line2D([], [], color='0.35', lw=lw(0.9), label='1σ ellipse'),
            Line2D([], [], ls='none', marker='o', mfc='0.6', mec='black', mew=axlw(),
                   ms=ms(1.25), label='Mean')]
    # Nothing is drawn as a line, so the mode entries are markers only
    return [Line2D([], [], ls='none', color=MODE_COLOURS[m], marker=MODE_MARKERS[m],
                   label=label(m)) for m in modes] + [
        Line2D([], [], ls='none', marker='o', mfc='0.6', mec='black', mew=axlw(),
               ms=ms(1.25), label='Mean')]


def plot_clusters(runs, out_dir):
    """ Successful landings in the pad frame: one panel per speed, every mode overlaid
        with its landings, 1σ ellipse, mean and k/n successes. One figure with a single
        legend; panels wrap into balanced rows of at most CLUSTER_GROUP_SIZE
        (7 speeds -> 4 + 3, not 5 + 2). A success always ends over the pad, so the view
        is the pad plus CLUSTER_MARGIN and nothing is clipped. """
    modes = [m for m in MODES if m in set(runs['mode'])]
    speeds = sorted(runs['speed'].unique())
    xlim, ylim = cluster_view()
    nrows = int(np.ceil(len(speeds) / CLUSTER_GROUP_SIZE))
    ncols = int(np.ceil(len(speeds) / nrows))
    width = FIG_WIDTH_DOUBLE if ncols > 2 else FIG_WIDTH_SINGLE
    panel_w = (width - 0.35) / ncols                 # 0.35 in for the y label and ticks
    aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig, axes = new_figure(nrows, ncols, (width, nrows * (panel_w * aspect + 0.14) + 0.22),
                           dict(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.03),
                           sharex=True, sharey=True, squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= len(speeds):
            ax.set_visible(False)
            axes.flat[k - ncols].tick_params(labelbottom=True)   # the panel above keeps ticks
            continue
        draw_cluster_panel(ax, runs[runs['speed'] == speeds[k]], modes, xlim, ylim)
        if k == 0:
            fwd_arrow(ax)
        ax.set_title(f'{speeds[k]:g} {SPEED_UNIT}', pad=1.5)
    axes[0][0].set_xlim(*xlim)    # shared axes: sets every panel
    axes[0][0].set_ylim(*ylim)
    fig.supxlabel('Lateral (m)', fontsize=plt.rcParams['axes.labelsize'])
    fig.supylabel('Longitudinal (m)', fontsize=plt.rcParams['axes.labelsize'])
    top_legend(fig, cluster_handles(modes))
    finish(fig, out_dir, 'landing_clusters')


def plot_clusters_pooled_by_mode(per_traj, out_dir, name):
    """ Successful landings with every speed pooled into one panel per trajectory (modes
        overlaid). One trajectory: a single-column figure. Several: one full-width row,
        a panel per trajectory (TRAJ_LABELS order). per_traj: {trajectory: runs} """
    trajs = traj_order(per_traj)
    modes = [m for m in MODES if any(m in set(r['mode']) for r in per_traj.values())]
    xlim, ylim = cluster_view()
    n = len(trajs)
    width = FIG_WIDTH_DOUBLE if n > 1 else FIG_WIDTH_SINGLE
    panel_w = (width - 0.35) / n
    aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig, axes = new_figure(1, n, (width, panel_w * aspect + 0.36),
                           dict(w_pad=0.02, h_pad=0.02, wspace=0.02), sharex=True,
                           sharey=True, squeeze=False)
    for k, (ax, traj) in enumerate(zip(axes[0], trajs)):
        draw_cluster_panel(ax, per_traj[traj], modes, xlim, ylim)
        if k == 0:
            fwd_arrow(ax)
        if n > 1:
            ax.set_title(traj_label(traj), fontweight='bold')
    axes[0][0].set_xlim(*xlim)
    axes[0][0].set_ylim(*ylim)
    fig.supxlabel('Lateral (m)', fontsize=plt.rcParams['axes.labelsize'])
    axes[0][0].set_ylabel('Longitudinal (m)')
    top_legend(fig, cluster_handles(modes),
               ncol=None if n > 1 else int(np.ceil(len(cluster_handles(modes)) / 2)))
    finish(fig, out_dir, name)


def speed_colours(speeds):
    """ One colour per speed from COLOURMAP, slow to fast, avoiding its palest end.
        Returns ({speed: colour}, ScalarMappable for a discrete colourbar). """
    speeds = sorted(speeds)
    cmap = matplotlib.colormaps[COLOURMAP]
    pos = np.linspace(0.0, 0.85, len(speeds)) if len(speeds) > 1 else [0.4]
    cols = {sp: mcolors.to_hex(cmap(x)) for sp, x in zip(speeds, pos)}
    edges = np.r_[speeds[0] - 0.5, (np.asarray(speeds[:-1]) + np.asarray(speeds[1:])) / 2,
                  speeds[-1] + 0.5] if len(speeds) > 1 else [speeds[0] - 0.5, speeds[0] + 0.5]
    sm = plt.cm.ScalarMappable(cmap=mcolors.ListedColormap([cols[sp] for sp in speeds]),
                               norm=mcolors.BoundaryNorm(edges, len(speeds)))
    sm.set_array([])
    return cols, sm


def plot_clusters_pooled_by_speed(per_traj, out_dir, name):
    """ Every speed pooled, coloured by speed (one shared scale and colourbar), one panel
        per mode and trajectory so the colour scale can be reused. CLUSTER_GRID:
            'traj_rows'  a row per trajectory (its name as a heading above the row), a
                         column per mode, single column, colourbar at the bottom
            'mode_rows'  a row per mode, a column per trajectory, full width, colourbar
                         on the right
        Each panel shows its successful landings, a 1σ ellipse per speed
        (CLUSTER_ELLIPSES) and k/n. """
    trajs = traj_order(per_traj)
    modes = [m for m in MODES if any(m in set(r['mode']) for r in per_traj.values())]
    speeds = sorted(set().union(*(set(r['speed']) for r in per_traj.values())))
    cols, sm = speed_colours(speeds)
    xlim, ylim = cluster_view()
    aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    one = len(trajs) == 1
    by_traj = CLUSTER_GRID == 'traj_rows'
    rows, cols_ = (trajs, modes) if by_traj else (modes, trajs)
    nrows, ncols = len(rows), len(cols_)
    pads = dict(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.03)
    if by_traj:
        # Grid rows: per trajectory a thin heading row and its panels, then the x label
        # and the colourbar underneath
        width = FIG_WIDTH_SINGLE
        panel_w = (width - 0.55) / ncols
        panel_h, head_h, lab_h, cb_h = panel_w * aspect, (0.0 if one else 0.2), 0.12, 0.1
        heights = ([head_h, panel_h] if not one else [panel_h]) * nrows + [lab_h, cb_h]
        fig = plt.figure(figsize=(width, sum(heights) + 0.6), constrained_layout=True)
        if hasattr(fig, 'get_layout_engine'):
            fig.get_layout_engine().set(**pads)
        else:
            fig.set_constrained_layout_pads(**pads)
        gs = fig.add_gridspec(len(heights), ncols, height_ratios=heights)
        xlab = fig.add_subplot(gs[-2, :])
        xlab.axis('off')
        xlab.text(0.5, 0.5, 'Lateral (m)', ha='center', va='center', transform=xlab.transAxes,
                  fontsize=plt.rcParams['axes.labelsize'])
        cax = fig.add_subplot(gs[-1, :])
        axes = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            r = 2 * i + 1 if not one else i
            if not one:
                head = fig.add_subplot(gs[r - 1, :])
                head.axis('off')
                head.text(0.5, 0.5, traj_label(rows[i]), ha='center', va='center',
                          transform=head.transAxes, fontsize=plt.rcParams['axes.titlesize'],
                          fontweight='bold')
            for j in range(ncols):
                ref = axes[0][0] if (i or j) else None
                axes[i][j] = fig.add_subplot(gs[r, j], sharex=ref, sharey=ref)
                if j:
                    axes[i][j].tick_params(labelleft=False)
                if i < nrows - 1:
                    axes[i][j].tick_params(labelbottom=False)
    else:
        width = FIG_WIDTH_SINGLE if one else FIG_WIDTH_DOUBLE
        panel_w = (width - 1.6) / ncols             # y labels and colourbar on the right
        height = nrows * (panel_w * aspect + 0.16) + 0.3
        fig, axes = new_figure(nrows, ncols, (width, height), pads,
                               sharex=True, sharey=True, squeeze=False)
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]
            traj, mode = (rows[i], cols_[j]) if by_traj else (cols_[j], rows[i])
            g = per_traj[traj][per_traj[traj]['mode'] == mode]
            draw_pad(ax, xlim, ylim)
            if panel_w < 1.8:   # narrow: label the outer ticks only, 0 is the centre cross
                ax.xaxis.set_major_locator(plt.FixedLocator([-0.5, 0.5]))
            for sp in speeds:
                ok = g[(g['speed'] == sp) & g['success']]
                ax.scatter(*cluster_xy(ok), s=ms(0.95) ** 2, color=cols[sp], clip_on=False,
                           marker=MODE_MARKERS[mode], ec='white', lw=axlw(0.3), alpha=0.9,
                           zorder=4)
                if CLUSTER_ELLIPSES:
                    cov_ellipse(ax, np.column_stack(cluster_xy(ok)), cols[sp])
            if len(g):
                ax.text(0.93, 0.10, f'{int(g["success"].sum())}/{len(g)}',
                        transform=ax.transAxes, ha='right', va='bottom', fontsize=fs(0.8),
                        color='0.25', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75), zorder=10)
            if i == j == 0:
                fwd_arrow(ax)
            if by_traj:
                if i == 0:                   # mode names once, under the first heading
                    ax.set_title(label(mode))
            else:
                if i == 0 and not one:
                    ax.set_title(traj_label(traj), fontweight='bold')
                if j == 0:
                    ax.set_ylabel(label(mode))
    axes[0][0].set_xlim(*xlim)
    axes[0][0].set_ylim(*ylim)
    if not by_traj:                       # by_traj: the x label sits above the colourbar
        fig.supxlabel('Lateral (m)', fontsize=plt.rcParams['axes.labelsize'])
    fig.supylabel('Longitudinal (m)', fontsize=plt.rcParams['axes.labelsize'])
    major = SPEED_TICKS[0]
    ticks = speeds if len(speeds) <= 6 else [sp for sp in speeds if sp % major == 0] or speeds
    if by_traj:
        cb = fig.colorbar(sm, cax=cax, ticks=ticks, orientation='horizontal')
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    else:
        cb = fig.colorbar(sm, ax=axes.ravel().tolist(), ticks=ticks, fraction=0.04,
                          pad=0.02, aspect=25)
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    cb.set_label(f'Car speed ({SPEED_UNIT})')
    cb.ax.tick_params(length=0)
    cb.outline.set_linewidth(axlw())
    finish(fig, out_dir, name)


def plot_clusters_pooled(per_traj, out_dir, name='landing_clusters_pooled'):
    """ Every speed pooled per trajectory, coloured by CLUSTER_POOL_COLOUR. """
    if CLUSTER_POOL_COLOUR == 'speed':
        plot_clusters_pooled_by_speed(per_traj, out_dir, name)
    else:
        plot_clusters_pooled_by_mode(per_traj, out_dir, name)


def plot_topdown(per_traj, paths, out_dir, name):
    """ Top-down view of each trajectory: the car's pad centre and the drone over the
        whole run of one representative run (TOPDOWN_MODE at TOPDOWN_SPEED, else the
        fastest speed it landed at), start (open circle) and end (filled square, a cross
        if that run failed) marked, equal axes. One panel per trajectory, side by side.
        per_traj: {trajectory: runs} """
    picks = []
    for traj in traj_order(per_traj):
        runs = per_traj[traj]
        g = runs[runs['mode'] == TOPDOWN_MODE]
        g = g if len(g) else runs
        pool = g[g['success']] if g['success'].any() else g
        speeds = sorted(pool['speed'].unique())
        sp = TOPDOWN_SPEED if TOPDOWN_SPEED in speeds else speeds[-1]
        r = pool[pool['speed'] == sp].sort_values('run').iloc[0]
        p = paths.get((traj, sp, r['mode'], r['run']), {}).get('topdown')
        if p is not None:
            picks.append((traj, sp, r['mode'], bool(r['success']), p))
    if not picks:
        return
    n = len(picks)
    width = FIG_WIDTH_DOUBLE if n > 1 else FIG_WIDTH_SINGLE
    panel_w = (width - 0.4) / n
    fig, axes = new_figure(1, n, (width, panel_w + 0.35),
                           dict(w_pad=0.03, h_pad=0.03, wspace=0.05), squeeze=False)
    car_c = cmap_colour('car')
    for ax, (traj, sp, mode, ok, p) in zip(axes[0], picks):
        dc = MODE_COLOURS[mode]
        tracks = ((p['car'], car_c, '-'), (p['drone'], dc, '--'))
        for (x, y), c, ls in tracks:
            ax.plot(x, y, color=c, ls=ls, zorder=3)
        for (x, y), c, _ in tracks:
            ax.plot(x[0], y[0], 'o', ms=ms(1.4), mfc='white', mec=c, mew=lw(), zorder=5)
            ax.plot(x[-1], y[-1], 's' if ok else 'X', ms=ms(1.4), mfc=c, mec='black',
                    mew=axlw(), zorder=6)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True)
        ax.set_title(traj_label(traj), fontweight='bold')
        ax.text(0.97, 0.04, f'{sp:g} {SPEED_UNIT}', transform=ax.transAxes, ha='right',
                va='bottom', fontsize=fs(0.8), color='0.3', zorder=10,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
        ax.set_xlabel('x (m)')
    axes[0][0].set_ylabel('y (m)')
    top_legend(fig, [
        Line2D([], [], color=car_c, label='Car (pad centre)'),
        Line2D([], [], color=MODE_COLOURS.get(TOPDOWN_MODE, '0.3'), ls='--', label='Drone'),
        Line2D([], [], ls='none', marker='o', ms=ms(1.4), mfc='white', mec='0.3', mew=lw(),
               label='Start'),
        Line2D([], [], ls='none', marker='s', ms=ms(1.4), mfc='0.5', mec='black',
               mew=axlw(), label='End')])
    finish(fig, out_dir, name)


def plot_paths_3d(runs, paths, out_dir):
    """ 3-D flight paths: per speed and frame, one panel per mode with every repeat
        overlaid. Landed runs solid, failed runs dashed; ^ marks the start (first
        detection or pad lock), o the touchdown, x where a failed run ended. """
    modes = [m for m in MODES if m in set(runs['mode'])]
    traj = runs['trajectory'].iloc[0]
    pdir = out_dir / 'paths_3d'
    pdir.mkdir(parents=True, exist_ok=True)
    car_c = cmap_colour('car')

    for frame in PATH3D_FRAMES:
        for speed, gs in runs.groupby('speed', sort=True):
            per_mode = {m: [] for m in modes}
            for r in gs.itertuples():
                p = paths.get((traj, speed, r.mode, r.run), {}).get(frame)
                if p is not None and r.mode in per_mode:
                    per_mode[r.mode].append((bool(r.success), p))
            shown = [m for m in modes if per_mode[m]]
            if not shown:
                continue

            # Shared limits so the panels compare directly
            pts = []
            for m in shown:
                for _, p in per_mode[m]:
                    pts += [np.column_stack(p['drone']), np.column_stack(p['car'])] \
                        if frame == 'world' else [np.column_stack(p)]
            pts = np.vstack(pts)
            pts = pts[np.all(np.isfinite(pts), axis=1)]
            if frame == 'pad':   # keep the car outline (and so the pad) in view
                x0, x1, y0, y1 = CAR_FOOTPRINT
                px, py = PAD_POSITION[:2]
                pts = np.vstack([pts, [[x0 - px, y0 - py, 0.0], [x1 - px, y1 - py, 0.0]]])
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            margin = 0.03 * (hi - lo) + 1e-6
            lo, hi = lo - margin, hi + margin
            rng = hi - lo
            longest = max(rng[0], rng[1])
            box = (max(rng[0], PATH3D_MIN_ASPECT * longest),
                   max(rng[1], PATH3D_MIN_ASPECT * longest),
                   min(max(rng[2], PATH3D_MIN_ASPECT * longest), PATH3D_MAX_Z_ASPECT * longest))

            n = len(shown)
            fig = plt.figure(figsize=(FIG_WIDTH_DOUBLE if n > 1 else FIG_WIDTH_SINGLE, 2.6))
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.97, wspace=0.02)
            for i, mode in enumerate(shown):
                ax = fig.add_subplot(1, n, i + 1, projection='3d')
                c = MODE_COLOURS[mode]
                if frame == 'pad':
                    hx, hy = PAD_SIZE[0] / 2, PAD_SIZE[1] / 2
                    ax.add_collection3d(Poly3DCollection(
                        [[(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)]],
                        facecolor='0.8', edgecolor='0.3', linewidths=axlw(1.2), alpha=0.7))
                    x0, x1, y0, y1 = CAR_FOOTPRINT
                    px, py = PAD_POSITION[:2]
                    ax.plot([x0 - px, x1 - px, x1 - px, x0 - px, x0 - px],
                            [y0 - py, y0 - py, y1 - py, y1 - py, y0 - py], 0,
                            color='0.55', lw=axlw())
                k = 0
                for ok, p in per_mode[mode]:
                    if frame == 'world':
                        ax.plot(*p['car'], color=car_c, lw=lw(0.75), alpha=0.45)
                        x, y, z = p['drone']
                    else:
                        x, y, z = p
                    k += ok
                    ax.plot(x, y, z, color=c, lw=lw(0.75), ls='-' if ok else '--',
                            alpha=0.8 if ok else 0.6)
                    ax.scatter(x[:1], y[:1], z[:1], marker='^', s=ms() ** 2, color=c,
                               depthshade=False)
                    ax.scatter(x[-1:], y[-1:], z[-1:], marker='o' if ok else 'x',
                               s=ms(1.1) ** 2, color=c, edgecolors='black' if ok else None,
                               linewidths=axlw(0.6) if ok else lw(), depthshade=False)

                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(lo[1], hi[1])
                ax.set_zlim(lo[2], hi[2])
                try:                       # zoom in on the box, 3.6+
                    ax.set_box_aspect(box, zoom=1.15)
                except TypeError:          # 3.5: same effect via the camera distance
                    ax.set_box_aspect(box)
                    ax.dist = 8.7
                ax.view_init(*PATH3D_VIEW[frame])
                names = (('x (m)', 'y (m)', 'z (m)') if frame == 'world' else
                         ('Longitudinal (m)', 'Lateral (m)', 'Height above pad (m)'))
                for axis, name in zip((ax.xaxis, ax.yaxis, ax.zaxis), names):
                    axis.set_label_text(name)
                    axis.set_major_locator(MaxNLocator(4))   # sparse: 3-D ticks crowd fast
                    axis.labelpad = -2
                    try:   # white panes read better in print
                        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                    except AttributeError:
                        pass
                ax.tick_params(labelsize=fs(0.85), pad=-2)
                ax.set_title(f'{label(mode)}, {speed:g} {SPEED_UNIT}  '
                             f'({k}/{len(per_mode[mode])} landed)', pad=0)

            start = 'First detection' if frame == 'world' else 'Pad lock'
            handles = [Line2D([], [], color='0.3', label='Landed'),
                       Line2D([], [], color='0.3', ls='--', label='Failed'),
                       Line2D([], [], ls='none', marker='^', color='0.3', ms=ms(1.35),
                              label=start),
                       Line2D([], [], ls='none', marker='o', color='0.3', mec='black',
                              mew=axlw(0.6), ms=ms(1.35), label='Touchdown'),
                       Line2D([], [], ls='none', marker='x', color='0.3', ms=ms(1.35),
                              label='End (failed)')]
            if frame == 'world':
                handles.append(Line2D([], [], color=car_c, label='Car (pad centre)'))
            else:
                handles.append(Rectangle((0, 0), 1, 1, fc='0.8', ec='0.3', lw=axlw(1.2),
                                         label='Landing pad'))
            top_legend(fig, handles)
            finish(fig, pdir, f'{frame}_{speed:g}kmh')


# ── LaTeX table ────────────────────────────────────────────────────────────

STATIONARY = 'Stationary'   # table group for the 0 km/h runs


def landing_table(runs):
    """ One row per trajectory, speed and mode in TABLE_MODES: landing error and time to
        land (mean ± std over successful landings) and success rate (over all runs).
        Pass each run once (no shared 0 km/h copies). With STATIONARY_ROW the 0 km/h runs
        of every trajectory form a single group reported once. """
    runs = runs[runs['mode'].isin(TABLE_MODES)]
    groups = []
    if STATIONARY_ROW:
        groups.append((STATIONARY, runs[runs['speed'] == 0]))
        runs = runs[runs['speed'] != 0]
    groups += [(traj_label(t), runs[runs['trajectory'] == t])
               for t in traj_order(runs['trajectory'])]
    rows = []
    for name, g in groups:
        for (speed, mode), gm in g.groupby(['speed', 'mode'], sort=True):
            ok = gm[gm['success']]
            e, t = ok['rest_error'].dropna(), ok['t_detect_land'].dropna()
            rows.append(dict(trajectory=name, speed=speed, mode=mode, method=label(mode),
                             n_runs=len(gm), n_success=len(ok),
                             success_pct=100 * len(ok) / len(gm),
                             error_mean=e.mean() if len(e) else np.nan,
                             error_std=e.std(ddof=1) if len(e) > 1 else np.nan,
                             t_land_mean=t.mean() if len(t) else np.nan,
                             t_land_std=t.std(ddof=1) if len(t) > 1 else np.nan))
    tab = pd.DataFrame(rows)
    if len(tab):   # keep the mode order of TABLE_MODES within each speed
        tab['_m'] = tab['mode'].map({m: i for i, m in enumerate(TABLE_MODES)})
        order = {n: i for i, (n, _) in enumerate(groups)}
        tab = (tab.assign(_g=tab['trajectory'].map(order))
               .sort_values(['_g', 'speed', '_m']).drop(columns=['_g', '_m'])
               .reset_index(drop=True))
    return tab


def latex_table(tab, tex_label=None, single=None):
    """ Compact booktabs table (needs \\usepackage{booktabs}), one row per trajectory and
        speed. With several modes each metric gets a sub-column per mode, side by side
        (TABLE_MODE_SHORT headers), best per row in bold. single: the trajectory name when
        the table covers one trajectory only; its column is then dropped. """
    dp = TABLE_DECIMALS
    modes = [m for m in TABLE_MODES if m in set(tab['mode'])]
    k = len(modes)
    env = TABLE_ENV or ('table*' if k > 1 else 'table')
    rnd = lambda v, d: float(np.round(v, d)) if np.isfinite(v) else np.nan

    def pm(m, s, d, bold=False):
        if not np.isfinite(m):
            return '--'
        core = f'{m:.{d}f}' + (f' \\pm {s:.{d}f}' if np.isfinite(s) else '')
        return f'$\\mathbf{{{core}}}$' if bold else f'${core}$'

    n = sorted(tab['n_runs'].unique())
    n_txt = f'{n[0]}' if len(n) == 1 else f'{n[0]}--{n[-1]}'
    caption = ((f'Landing performance on the {single} trajectory. ' if single else
                'Landing performance. ')
               + (TABLE_MODE_NOTE + ' ' if k > 1 and TABLE_MODE_NOTE else '')
               + 'Error and time to land: mean $\\pm$ standard deviation over successful '
               f'landings; success over all runs ($n = {n_txt}$ per case). Time to land is '
               'from first detection to touchdown.'
               + (f' {STATIONARY}: 0 {SPEED_UNIT}, shared by every trajectory.'
                  if not single and (tab['trajectory'] == STATIONARY).any() else '')
               + (' Bold: better per row.' if k > 1 and TABLE_BOLD_BEST else ''))

    lead = ([] if single else ['Trajectory']) + ['Speed']
    metrics = [('error', 'Error', '(m)'), ('success', 'Success', '(\\%)'),
               ('t_land', 'Time to land', '(s)')]
    spec = ''.join('l' if h == 'Trajectory' else 'c' for h in lead) + 'c' * (3 * k)
    nl = len(lead)
    if k > 1:   # metric names spanning their mode sub-columns, then mode names with units
        top = lead + [f'\\multicolumn{{{k}}}{{c}}{{{name} {unit}}}' for _, name, unit in metrics]
        rules = ''.join(f'\\cmidrule(lr){{{nl + 1 + i * k}-{nl + (i + 1) * k}}}'
                        for i in range(3))
        sub = [''] * (nl - 1) + [f'({SPEED_UNIT})'] \
            + [TABLE_MODE_SHORT.get(m, label(m)) for _ in metrics for m in modes]
        header = [' & '.join(top) + ' \\\\', rules, ' & '.join(sub) + ' \\\\']
    else:       # two lines: name, then unit
        header = [' & '.join(lead + [name for _, name, _ in metrics]) + ' \\\\',
                  ' & '.join([''] * (nl - 1) + [f'({SPEED_UNIT})']
                             + [unit for *_, unit in metrics]) + ' \\\\']
    out = [f'\\begin{{{env}}}[t]', '\\centering'] + ([TABLE_FONT] if TABLE_FONT else []) + [
           f'\\setlength{{\\tabcolsep}}{{{TABLE_COLSEP_PT}pt}}',
           f'\\caption{{{caption}}}', f'\\label{{{tex_label or TABLE_LABEL}}}',
           f'\\begin{{tabular}}{{@{{}}{spec}@{{}}}}', '\\toprule'] + header + ['\\midrule']

    for i, (traj, g) in enumerate(tab.groupby('trajectory', sort=False)):
        if i:
            out.append('\\midrule')
        for j, (speed, gs) in enumerate(g.groupby('speed', sort=False)):
            r = {row.mode: row for row in gs.itertuples()}
            vals = {m: dict(error=rnd(getattr(r[m], 'error_mean', np.nan), dp['error']),
                            success=rnd(getattr(r[m], 'success_pct', np.nan), dp['pct']),
                            t_land=rnd(getattr(r[m], 't_land_mean', np.nan), dp['time']))
                    if m in r else {} for m in modes}
            best = {}
            if k > 1 and TABLE_BOLD_BEST:   # judged on the values as printed; ties all bold
                for key, pick in (('error', min), ('success', max), ('t_land', min)):
                    v = [vals[m][key] for m in modes if vals[m] and np.isfinite(vals[m][key])]
                    if len(v) > 1 and len(set(v)) > 1:
                        best[key] = pick(v)
            cells = ([] if single else [traj if j == 0 else '']) + [f'{speed:g}']
            for key, _, _ in metrics:
                for m in modes:
                    if m not in r:
                        cells.append('--')
                        continue
                    row = r[m]
                    bold = key in best and vals[m][key] == best[key]
                    if key == 'success':
                        v = f'{row.success_pct:.{dp["pct"]}f}'
                        cells.append(f'\\textbf{{{v}}}' if bold else v)
                    elif key == 'error':
                        cells.append(pm(row.error_mean, row.error_std, dp['error'], bold))
                    else:
                        cells.append(pm(row.t_land_mean, row.t_land_std, dp['time'], bold))
            out.append(' & '.join(cells) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}', f'\\end{{{env}}}']
    return '\n'.join(out) + '\n'


# ── Printing ───────────────────────────────────────────────────────────────

def pm(m, s, fmt='{:.3f}'):
    if not np.isfinite(m):
        return '-'
    return fmt.format(m) + (' ± ' + fmt.replace('+', '').format(s) if np.isfinite(s) else '')


def print_summary(summary, modes_cmp):
    print('\n' + '=' * 100)
    print('ESTIMATION ERROR  (RMSE per run, mean ± std over runs; gain = raw held - filter, + = filter better)')
    print('=' * 100)
    for _, r in summary.iterrows():
        print(f"\n{r['speed']:g} {SPEED_UNIT}  |  {MODES[r['mode']]}  |  n = {r['n_runs']}"
              f" ({r['n_scored']} successful, errors over these)"
              f"  |  mean window {r['window_s_mean']:.1f} s")
        fate = ('window used fixed fallback' if CONV_FALLBACK_TO_FIXED else 'not scored') \
            if WINDOW_START == 'converged' else f"window is '{WINDOW_START}' anyway"
        print(f"  converged {pm(r['t_converged_mean'], r['t_converged_std'], '{:.1f}')} s after 3000"
              + (f"   ({r['n_not_converged']} run(s) never converged: {fate})"
                 if r['n_not_converged'] else ''))
        for src, name in (('filt', 'Filter (live)'), ('held', 'Raw fit held (live)'),
                          ('post', 'Filter @ stamp'), ('stamp', 'Raw fit @ stamp')):
            if f'{src}_lat_mean' not in r or not np.isfinite(r[f'{src}_lat_mean']):
                continue
            print(f"  {name:<20s} lat {pm(r[f'{src}_lat_mean'], r[f'{src}_lat_std'])}   "
                  f"long {pm(r[f'{src}_long_mean'], r[f'{src}_long_std'])}   "
                  f"xy {pm(r[f'{src}_xy_mean'], r[f'{src}_xy_std'])}   "
                  f"z {pm(r[f'{src}_z_mean'], r[f'{src}_z_std'])}   "
                  f"3d {pm(r.get(f'{src}_3d_mean', np.nan), r.get(f'{src}_3d_std', np.nan))}  m")
        for f in ('lat', 'long', 'xy', 'z', '3d'):
            if f'gain_{f}_mean' in r and np.isfinite(r[f'gain_{f}_mean']):
                p = r[f'gain_{f}_wilcoxon_p']
                print(f"  gain {f:<4s} {r[f'gain_{f}_mean']:+.3f} ± {r[f'gain_{f}_ci95']:.3f} m (95% CI)"
                      f"   Wilcoxon p = {'n<6' if not np.isfinite(p) else f'{p:.3f}'}")
        for f in ('lat', 'long', 'xy', 'z', '3d'):
            if f'improve_{f}_pct' in r and np.isfinite(r[f'improve_{f}_pct']):
                p = r[f'improve_{f}_wilcoxon_p']
                print(f"  improvement {f:<4s} {r[f'improve_{f}_pct']:+.0f}% "
                      f"(95% CI {r[f'improve_{f}_ci_lo']:+.0f} to {r[f'improve_{f}_ci_hi']:+.0f}%)"
                      f"   Wilcoxon p (log ratio) = {'n<6' if not np.isfinite(p) else f'{p:.3f}'}")

    print('\n' + '=' * 100)
    print('LANDING  (landing error over successful runs; times over all runs that reached the state)')
    print('=' * 100)
    hdr = f"{'speed':>6s}  {'mode':<26s} {'success':>18s} {'1st-att':>8s} {'error (m)':>14s} " \
          f"{'long (m)':>15s} {'lat (m)':>15s} {'t_acq (s)':>12s} {'t_land (s)':>12s} " \
          f"{'det-land (s)':>13s} {'attempts':>10s}"
    print(hdr)
    for _, r in summary.iterrows():
        succ = f"{r['success_n']}/{r['n_runs']} ({r['success_pct']:.0f}%)"
        print(f"{r['speed']:>6g}  {MODES[r['mode']]:<26s} {succ:>18s} "
              f"{r['first_attempt_success_pct']:>7.0f}% "
              f"{pm(r['rest_error_mean'], r['rest_error_std'], '{:.2f}'):>14s} "
              f"{pm(r['rest_long_mean'], r['rest_long_std'], '{:+.2f}'):>15s} "
              f"{pm(r['rest_lat_mean'], r['rest_lat_std'], '{:+.2f}'):>15s} "
              f"{pm(r['t_acquisition_mean'], r['t_acquisition_std'], '{:.1f}'):>12s} "
              f"{pm(r['t_landing_mean'], r['t_landing_std'], '{:.1f}'):>12s} "
              f"{pm(r['t_detect_land_mean'], r['t_detect_land_std'], '{:.1f}'):>13s} "
              f"{pm(r['attempts_mean'], r['attempts_std'], '{:.1f}'):>10s}")
    print("  success interval (95% Wilson) is in summary.csv: success_ci_lo / success_ci_hi")

    if len(modes_cmp):
        a, b = modes_cmp.attrs['modes']
        print('\n' + '=' * 100)
        print(f'FILTER RMSE: {MODES[a]} vs {MODES[b]}  (Mann-Whitney U, unpaired)')
        print('=' * 100)
        for _, r in modes_cmp.iterrows():
            parts = []
            for f in ('lat', 'long', 'xy'):
                p = r[f'{f}_mannwhitney_p']
                parts.append(f"{f} {r[f'{a}_{f}']:.3f} vs {r[f'{b}_{f}']:.3f} "
                             f"(p = {'-' if not np.isfinite(p) else f'{p:.3f}'})")
            print(f"{r['speed']:>6g} {SPEED_UNIT}   " + '   '.join(parts))


def share_stationary(all_runs, paths):
    """ Copy the 0 km/h runs into every trajectory that has none of its own, per mode,
        from a single source trajectory (the first in TRAJ_LABELS order that has them),
        so no trajectory receives the same run twice. Copies carry the source in
        shared_from; the 3-D path data is shared under the new trajectory key too. """
    still = all_runs[(all_runs['speed'] == 0) & ~all_runs['trajectory'].map(is_standalone)]
    if still.empty:
        return all_runs, paths
    added = []
    for mode in still['mode'].unique():
        sources = traj_order(still.loc[still['mode'] == mode, 'trajectory'])
        src = still[(still['mode'] == mode) & (still['trajectory'] == sources[0])]
        for traj in traj_order(all_runs['trajectory']):
            if traj in sources or is_standalone(traj):
                continue
            added.append(src.assign(trajectory=traj, shared_from=sources[0]))
            for r in src.itertuples():
                key = (r.trajectory, r.speed, r.mode, r.run)
                if key in paths:
                    paths[(traj, r.speed, r.mode, r.run)] = paths[key]
            print(f"NOTE: {len(src)} x 0 {SPEED_UNIT} {mode} run(s) from '{sources[0]}' "
                  f"shared into '{traj}'")
    if added:
        all_runs = pd.concat([all_runs] + added, ignore_index=True)
    return all_runs, paths


# ── Discovery ──────────────────────────────────────────────────────────────

def find_runs(root, trajectories=None):
    """ Yield (trajectory, speed, mode, rep, csv) for every finished run under
        <root>/<study>/<trajectory>/<NN>kmh/repNN/controller_*.csv """
    for mode in MODES:
        study = root / mode
        if not study.is_dir():
            print(f"NOTE: study '{mode}' not found ({study}), skipped")
            continue
        for traj_dir in sorted(p for p in study.iterdir() if p.is_dir()):
            if trajectories and traj_dir.name not in trajectories:
                continue
            for spd_dir in sorted(traj_dir.glob('*kmh'), key=lambda p: parse_speed(p.name)):
                for rep_dir in sorted(spd_dir.glob('rep*')):
                    tag = f"{mode}/{traj_dir.name}/{spd_dir.name}/{rep_dir.name}"
                    if ONLY_DONE and not (rep_dir / 'DONE').exists():
                        print(f"NOTE: {tag} not finished, skipped")
                        continue
                    csvs = sorted(rep_dir.glob('controller_*.csv'))
                    if not csvs:
                        print(f"NOTE: {tag} has no controller CSV (sim never booted?), skipped")
                        continue
                    if len(csvs) > 1:
                        print(f"NOTE: {tag} has {len(csvs)} controller CSVs, using {csvs[-1].name}")
                    yield traj_dir.name, parse_speed(spd_dir.name), mode, rep_dir.name, csvs[-1]


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('root', type=Path, nargs='?', default=ROOT,
                    help='run_batch.sh results root (default: ROOT in the config)')
    ap.add_argument('--trajectories', nargs='+', default=TRAJECTORIES,
                    help='only these trajectory folders (default: TRAJECTORIES, else all)')
    ap.add_argument('--out', type=Path, default=OUT_DIR,
                    help='output folder (default: OUT_DIR, else <root>/batch_results)')
    ap.add_argument('--per-run-plots', action='store_true', default=PER_RUN_PLOTS,
                    help='also run plot_errors.py on every CSV over the same window')
    args = ap.parse_args()

    root = args.root.expanduser()
    if not root.is_dir():
        print(f"Root folder not found: {root.resolve()}  (set ROOT in the config)")
        return
    out_dir = Path(args.out).expanduser() if args.out else root / 'batch_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    pe.DPI = PER_RUN_DPI

    rows, paths = [], {}
    for traj, speed, mode, rep, csv in find_runs(root, args.trajectories):
        tag = f"{traj}/{speed:g}{SPEED_UNIT}/{mode}/{rep}"
        try:
            m, window, path = analyse_run(csv)
        except Exception as e:   # one broken log shouldn't kill the batch
            print(f"WARNING: {tag} skipped: {e}")
            continue
        m.update(trajectory=traj, speed=speed, mode=mode, run=rep, csv=str(csv))
        if path:
            paths[(traj, speed, mode, rep)] = path
        info = csv.parent / 'run_info.json'      # run settings written by run_batch.sh
        if info.exists():
            try:
                cfg = json.loads(info.read_text())
                m.update({k: v for k, v in cfg.items() if k not in m and k != 'repeat'})
            except ValueError:
                print(f"NOTE: {tag} has an unreadable run_info.json, ignored")
        rows.append(m)
        flag = 'OK  ' if m['success'] else 'FAIL'
        print(f"  {flag} {tag}: window {m['window_s']:.1f} s "
              f"[{m['window_rule']}"
              + (f", conv {m['t_converged']:.1f} s" if np.isfinite(m['t_converged']) else '')
              + "], "
              f"filt lat {m['filt_lat']:.3f} m, attempts {m['attempts']}, "
              f"rest ({m['rest_long']:+.2f}, {m['rest_lat']:+.2f}, h {m['rest_height']:+.2f}) m, "
              f"tilt {m['rest_tilt_deg']:.0f}°")
        if not np.isfinite(m['window_s']) or m['window_s'] <= 0:
            print("       WARNING: empty scoring window (no 3000, filter never converged, "
                  "or the first attempt ended before the window started)")

        if args.per_run_plots and np.isfinite(window[0]):
            pe.main(csv, out_dir=out_dir / traj / 'per_run' / f'{speed:g}kmh' / mode / rep,
                    window=window, show=False)

    if not rows:
        print("No runs found. Expected <root>/<study>/<trajectory>/<NN>kmh/repNN/controller_*.csv")
        return

    lead = ['trajectory', 'speed', 'mode', 'run', 'success', 'attempts', 'shared_from']
    all_runs = pd.DataFrame(rows).assign(shared_from='')
    if SHARE_STATIONARY:
        all_runs, paths = share_stationary(all_runs, paths)
    all_runs = all_runs[lead + [c for c in all_runs.columns if c not in lead]]
    all_runs.to_csv(out_dir / 'runs_all.csv', index=False)

    if 'held_lat' not in all_runs or all_runs['held_lat'].isna().all():
        print("WARNING: no raw YOLO fit in box mode - was yolo_shadow_fit_enabled on?")

    summaries, rmse_data, traj_runs = [], {}, {}
    for traj, runs in all_runs.groupby('trajectory', sort=True):
        tdir = out_dir / traj
        tdir.mkdir(parents=True, exist_ok=True)
        summary = summarise(runs).assign(trajectory=traj)
        modes_cmp = compare_modes(runs)
        raw = pool_raw(runs[runs['mode'].isin(RMSE_MODES)]).assign(trajectory=traj)

        runs.to_csv(tdir / 'runs.csv', index=False)
        summary.to_csv(tdir / 'summary.csv', index=False)
        raw.to_csv(tdir / 'raw_pooled.csv', index=False)
        if len(modes_cmp):
            modes_cmp.to_csv(tdir / 'mode_comparison.csv', index=False)
        print(f"\nSaved: {tdir / 'runs.csv'}, summary.csv, raw_pooled.csv"
              + (", mode_comparison.csv" if len(modes_cmp) else ''))

        print('\n' + '#' * 100)
        print(f'#  TRAJECTORY: {traj.upper()}')
        print('#' * 100)
        print_summary(summary, modes_cmp)
        print_raw_check(raw)
        with plt.rc_context(PAPER_RC):
            rmse_data[traj] = (summary[summary['mode'].isin(RMSE_MODES)], raw)
            plot_rmse(*rmse_data[traj], tdir)
            if RMSE_AXES_FIGURE and list(RMSE_METRICS) != list(AXIS_METRICS):
                plot_rmse(*rmse_data[traj], tdir, AXIS_METRICS, 'rmse_axes_vs_speed')
            plot_landing(runs, tdir)
            plot_clusters(runs, tdir)
            if CLUSTER_POOL_SPEEDS:
                plot_clusters_pooled({traj: runs}, tdir)
            if PATHS_3D:
                plot_paths_3d(runs, paths, tdir)
            if PLOT_TOPDOWN:
                plot_topdown({traj: runs}, paths, tdir, 'trajectory_topdown')
        traj_runs[traj] = runs
        summaries.append(summary)

    grid_data = {t: v for t, v in rmse_data.items() if not is_standalone(t)}
    left_out = sorted(set(rmse_data) - set(grid_data))
    if left_out:
        print(f"NOTE: standalone, left out of the grid and pooled table: {', '.join(left_out)}")
    if len(grid_data) > 1:
        adir = out_dir / 'all_trajectories'
        adir.mkdir(parents=True, exist_ok=True)
        grid_runs = {t: r for t, r in traj_runs.items() if not is_standalone(t)}
        with plt.rc_context(PAPER_RC):
            plot_rmse_grid(grid_data, adir)
            if RMSE_AXES_FIGURE and list(RMSE_METRICS) != list(AXIS_METRICS):
                plot_rmse_grid(grid_data, adir, AXIS_METRICS, 'rmse_axes_vs_speed')
            plot_landing_grid(grid_runs, adir)
            if CLUSTER_POOL_SPEEDS:
                plot_clusters_pooled(grid_runs, adir, 'landing_clusters')
            if PLOT_TOPDOWN:
                plot_topdown(grid_runs, paths, adir, 'trajectories_topdown')

    summary_all = pd.concat(summaries, ignore_index=True)

    summary_all = summary_all[['trajectory'] + [c for c in summary_all.columns if c != 'trajectory']]
    summary_all.to_csv(out_dir / 'summary_all.csv', index=False)

    own = all_runs['shared_from'] == ''                             # each run once
    alone = all_runs['trajectory'].map(is_standalone)
    for traj in traj_order(all_runs.loc[alone, 'trajectory']):
        tab = landing_table(all_runs[own & (all_runs['trajectory'] == traj)])
        if tab.empty:
            continue
        tab.to_csv(out_dir / traj / 'landing_table.csv', index=False)
        (out_dir / traj / 'landing_table.tex').write_text(
            latex_table(tab, f'{TABLE_LABEL}_{traj.lower()}', single=traj_label(traj)))
        print(f"Saved: {out_dir / traj / 'landing_table.tex'} (standalone)")
    table = landing_table(all_runs[own & ~alone])
    table.to_csv(out_dir / 'landing_table.csv', index=False)
    tex = latex_table(table)
    (out_dir / 'landing_table.tex').write_text(tex)
    print('\n' + '=' * 100 + '\nLATEX TABLE, needs \\usepackage{booktabs}\n'
          + '=' * 100 + '\n' + tex)
    print(f"Saved: {out_dir / 'landing_table.tex'}, landing_table.csv")
    print(f"\nSaved: {out_dir / 'runs_all.csv'}, {out_dir / 'summary_all.csv'}")


if __name__ == '__main__':
    main()