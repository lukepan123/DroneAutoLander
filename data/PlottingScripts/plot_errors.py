"""
plot_errors.py  –  Landing pad position error analysis
======================================================
Set LOG_DIR / CSV_FILE below, then run:

    python plot_errors.py                      # CSV_FILE in LOG_DIR (newest log if None)
    python plot_errors.py controller_20261113_154652.csv   # another log in LOG_DIR
    python plot_errors.py path/to/controller_data.csv

Outputs (saved to ./plots next to the script; PDF + PNG with PAPER_STYLE, which matches
the batch_analysis.py figures, else PNG only in the original diagnostic look):
    error_vs_time.png        –  longitudinal / lateral error (car frame), height (Z)
                                error, absolute horizontal error (with pipeline
                                latency on a secondary axis), yaw error and per-state
                                UKF 1-sigma. Controller-state transitions are marked
                                on every panel; spans where the UKF is using the YOLO
                                pixel-space box update are shaded.
    trajectory_2d.png        –  2-D XY path: true / filter / raw measurements
    consistency_analysis.png –  NIS / NEES vs time and distributions
    pipeline_latency.png     –  latency per pipeline stage, one sample per measurement

Car frame convention (same as the body-frame offset below):
    +longitudinal = along the car's heading (estimate ahead of truth)
    +lateral      = to the car's left       (estimate left of truth)
All errors are  estimate - truth.

Three like-for-like comparisons are drawn:
    - Filter estimate vs truth at the UKF tick (`est_time`) the estimate is valid at,
      not the (later) controller row it was logged on.
    - Raw measurement vs truth at its own image timestamp (crosses), so pipeline
      latency doesn't show up as a fake along-track error. Compare these against the
      filter state at the same image stamp (dots, needs `yolo_post_rel_*` columns).
    - Raw measurement held until the next one arrives, vs truth at the row time
      (dashed step line). This is what a controller using raw directly would see, and
      is the fair live competitor for the filter estimate.

YOLO raw positions only exist while the UKF is in fallback (box-fit) mode; in box
mode the raw measurement is pixel edges, so no YOLO raw points are plotted there.

The YOLO box-derived yaw (`landing_pad_yolo_car_glob_yaw_raw`, logged only on frames
where it was actually fed to the UKF) is drawn on the yaw panel with its ±1σ
(`landing_pad_yolo_car_yaw_sigma_raw`). Errors beyond ±90° are a front/back flip and
are ringed in black; the summary counts them and gives the mean normalised squared
error (err/σ)², which should be about 1 if the yaw σ is honest.

Pipeline latency: new logs write one sample per measurement (`<stream>_<stage>_lag`,
NaN on every other row), and every stage is plotted on its own:
    YOLO:     transport, queue, inference, post-process, wait, UKF update
    AprilTag: transport, processing, wait, UKF update
Older SITL logs measured some stages inside a single callback while the sim clock was
frozen, so those read 0 on every sample. Such stages are detected and dropped (their
time, where it was counted at all, sits in the wait stage, which is relabelled), and
a note is printed. Legacy logs repeated the three old stage columns on every row;
those are de-duplicated on the raw measurement stamp before averaging.

Real flights:
    python plot_errors.py real.csv                       # RTK truth auto-found (fleet*.csv)
    python plot_errors.py real.csv --fleet fleet.csv --rtk-bias 2.1,-1.0
    python plot_errors.py real.csv --fleet none          # consistency + latency only

    Logs are loaded through log_loader.py (keep it next to this script), which builds
    SITL-style truth columns from an overlapping RTK telemetry log. Pad geometry comes from
    the platform (log_loader.PLATFORMS: SITL Grand Vitara pad at x = -0.80 m, z = 1.705 m;
    real Grand Vitara x = -0.79 m, z = 1.75 m) and the window defaults from WINDOW below. On RTK
    truth, the absolute-error panel shades the reported truth 1-sigma: errors inside it
    can't be resolved. Without any truth, the error and trajectory figures are skipped
    and the NIS / latency figures are still produced.

Dependencies:
    pip install pandas numpy matplotlib scipy
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.interpolate import interp1d
from scipy.stats import chi2

from log_loader import describe_truth, load_log, resolve_csv


# ── Config ─────────────────────────────────────────────────────────────────

# Log selection when no CSV is given on the command line (or by batch_analysis.py):
# CSV_FILE is a file name in LOG_DIR (or a full path); None = newest controller_*.csv in
# LOG_DIR. A bare file name on the command line is also looked up in LOG_DIR.
LOG_DIR  = '/home/luke/ros2_ws/field_2026-11-13_novideo/field_2026-11-13/pi_7inch'
CSV_FILE = 'controller_20261114_001509.csv'

# Body-frame offset applied to the true (car origin) position = landing pad top centre in
# the car frame. Set per platform in log_loader.PLATFORMS['sim' | 'real']['pad_in_car'].

OUTLIER_SPEED_THRESH = 5.0   # m/s

# Analysis window per platform, (T_START, T_END) in seconds since start of log; None
# disables a limit. With the auto start/end below on, these act as outer bounds.
WINDOW = {'sim': (10, 70), 'real': (None, None)}

# Start the analysis at the first entry into pad lock (3xxx), once the drone is over
# the car and tracking it. Before that the filter is seeding or the drone is still
# closing in. T_START still applies if later.
AUTO_START_AT_LOCK = True

# End the analysis at touchdown: the first landing-confirmed state (6xxx) or, if the
# landing never completed, the start of the last abort (5xxx). Data after either is
# the drone on the pad or climbing away, not tracking. If neither occurs (typical of real
# flights), the first of RTL (7xxx), pilot override (8xxx) or the pad being lost after
# lock (back to 2xxx) ends it instead. T_END still applies if earlier.
AUTO_END_AT_LANDING = True

# Latency statistics start at the first entry into this state (2100 = first pad
# detection, visual hold) rather than at pad lock, so seeding and the chase are
# included. They end where the error analysis ends. None = same window as the errors.
LATENCY_FROM_STATE = 2100

DPI = 500

# ── Figure style ──
# PAPER_STYLE = True gives the batch_analysis.py look: its PAPER_RC fonts and line
# weights, viridis colours (the filter is the same colour as batch_analysis's 'main'
# mode and the raw YOLO fit the same as its raw line), no figure titles or grid,
# journal widths, constrained layout, framed legends, and PDF output. Keep the values
# below equal to batch_analysis.py. False restores the original large diagnostic look.
PAPER_STYLE = True
COLOURMAP   = 'viridis'
FIG_WIDTH_SINGLE = 3.5
FIG_WIDTH_DOUBLE = 7.16
SAVE_FORMATS     = ('pdf', 'png')   # PNG at DPI for a quick look
SHOW_GRID        = False
LEGEND_BOX       = True
LEGEND_EDGE      = '0.8'
LEGEND_ROUND     = True
LEGEND_FONTSIZE  = 7.5
PAPER_RC = {
    'font.size': 9.5, 'axes.titlesize': 9.5, 'axes.labelsize': 9.5,
    'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0, 'legend.fontsize': 9.0,
    'lines.linewidth': 0.8, 'lines.markersize': 2.6, 'errorbar.capsize': 1.5,
    'axes.linewidth': 0.5, 'grid.linewidth': 0.3, 'grid.alpha': 0.3,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'xtick.minor.size': 1.2, 'xtick.minor.width': 0.4,
    'xtick.major.pad': 4.0, 'ytick.major.pad': 2.0, 'axes.labelpad': 2.0,
    'axes.titlepad': 3.0,
    'legend.frameon': False, 'legend.handlelength': 1.8, 'legend.columnspacing': 1.2,
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed TrueType fonts (journal requirement)
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
}


def cmap_colour(stop):
    """ A stop on COLOURMAP as hex (same scheme as batch_analysis.py). """
    import matplotlib
    import matplotlib.colors as mcolors
    return mcolors.to_hex(matplotlib.colormaps[COLOURMAP](stop))


if PAPER_STYLE:
    RAW_C      = cmap_colour(0.45)   # AprilTag raw measurement
    YOLO_LP_C  = cmap_colour(0.60)   # YOLO landing-pad raw (legacy logs only)
    YOLO_CAR_C = cmap_colour(0.75)   # YOLO car fit = batch_analysis raw line
    FILT_C     = cmap_colour(0.15)   # filter estimate = batch_analysis 'main' mode
    TRUE_C     = '0.15'              # true position
    BOX_MODE_C = cmap_colour(0.75)   # shading: UKF using the YOLO pixel-space box update
else:
    RAW_C        = '#e03939'   # red     – AprilTag raw measurement
    YOLO_LP_C    = '#d95f02'   # orange  – YOLO landing-pad raw (legacy logs only)
    YOLO_CAR_C   = '#7570b3'   # purple  – YOLO car fitted position
    FILT_C       = '#3a86d4'   # blue    – filter estimate
    TRUE_C       = '#2ca02c'   # green   – true position
    BOX_MODE_C   = '#7570b3'   # shading – UKF using the YOLO pixel-space box update
ALPHA        = 0.85

# Raw measurement streams: key, legend label, colour, CSV column prefix.
# Streams whose columns aren't in the CSV are skipped (YOLO LP = old logs only).
STREAMS = [
    ('april',    'AprilTag raw', RAW_C,      'landing_pad_april'),
    ('yolo_lp',  'YOLO LP raw',  YOLO_LP_C,  'landing_pad_yolo_LP'),
    ('yolo_car', 'YOLO car fit', YOLO_CAR_C, 'landing_pad_yolo_car'),
]

# ── Pipeline latency columns ──
# New logs: (stream label, column prefix, marker, [(stage suffix, stage label), ...]).
# Columns are f'{prefix}_{suffix}_lag' plus f'{prefix}_total_lag'.
LAG_STREAMS = [
    ('AprilTag', 'apriltag', 'x', [
        ('transport',  'Transport'),
        ('processing', 'Processing'),
        ('wait',       'Wait'),
        ('apply',      'UKF update'),
    ]),
    ('YOLO', 'yolo', 'o', [
        ('transport',  'Transport'),
        ('queue',      'Queue'),
        ('inference',  'Inference'),
        ('post',       'Post-process'),
        ('wait',       'Wait'),
        ('apply',      'UKF update'),
    ]),
]
# Frozen-clock fallbacks for older SITL logs. Before the fixes in yolo.py (timer in its
# own callback group) and controller.py (apply timed with perf_counter), these stages
# were measured inside one callback while the sim clock stood still, so they read 0 on
# every sample. When the probe stage reads 0 on >= FROZEN_ZERO_FRACTION of samples, the
# listed stages are dropped and the named stages relabelled. YOLO inference time then
# sat in the wait stage; the old UKF update time wasn't counted anywhere (total
# excludes it too), so it is just dropped. Post-process is legitimately ~0 ms (sub-ms
# work on a 1 ms sim clock), so it is never a probe.
FROZEN_ZERO_FRACTION = 0.95
FROZEN_RULES = {
    'YOLO': [
        ('inference', dict(drop=['Inference', 'Post-process'],
                           relabel={'Wait': 'Wait + inference'})),
        ('apply',     dict(drop=['UKF update'])),
    ],
    'AprilTag': [
        ('apply',     dict(drop=['UKF update'])),
    ],
}
# Stages folded into a neighbour when their mean is below NEGLIGIBLE_STAGE_MS, so the
# bars still sum to the total. On a 1 ms sim clock a sub-ms stage reads only 0 or 1 ms,
# so its mean is quantisation, not a measurement. Data-driven: if post-processing is
# slow on the Pi, it keeps its own bar.
NEGLIGIBLE_STAGE_MS = 0.5
MERGE_IF_NEGLIGIBLE = {'Post-process': 'Inference'}   # stage -> stage it folds into
# Legacy logs: stage columns and the raw stamp used to de-duplicate repeated rows
LEGACY_LAG = {
    'AprilTag': (['cam_to_image_lag', 'image_to_transform_lag', 'transform_to_UKF_lag'],
                 'landing_pad_april_raw_stamp'),
    'YOLO':     (['yolo_cam_to_image_lag', 'yolo_image_to_transform_lag',
                  'yolo_transform_to_UKF_lag'],
                 'landing_pad_yolo_car_raw_stamp'),
}
LEGACY_STAGE_LABELS = ['Camera → Image', 'Image → Transform', 'Transform → UKF']

STAGE_COLOURS = {
    'Transport':         '#e63946',
    'Queue':             '#f4a261',
    'Processing':        '#7570b3',
    'Inference':         '#7570b3',
    'Post-process':      '#8ecae6',
    'Wait':              '#2a9d8f',
    'Wait + inference':  '#2a9d8f',
    'Inference + post-process': '#7570b3',
    'UKF update':        '#264653',
    'Camera → Image':    '#e63946',
    'Image → Transform': '#7570b3',
    'Transform → UKF':   '#2a9d8f',
}
TOTAL_COLOURS = {'AprilTag': '#6a4c93', 'YOLO': '#e7298a'}

if PAPER_STYLE:
    # Stages in pipeline order along viridis (dark -> light, so they also separate in
    # greyscale), stopping short of the pale yellow end; totals in grey
    _STAGE_STOPS = {
        'Transport': 0.05, 'Camera → Image': 0.05,
        'Queue': 0.22,
        'Inference': 0.40, 'Processing': 0.40, 'Inference + post-process': 0.40,
        'Image → Transform': 0.40,
        'Post-process': 0.52,
        'Wait': 0.64, 'Wait + inference': 0.64, 'Transform → UKF': 0.64,
        'UKF update': 0.80,
    }
    STAGE_COLOURS = {k: cmap_colour(v) for k, v in _STAGE_STOPS.items()}
    TOTAL_COLOURS = {'YOLO': '0.30', 'AprilTag': '0.55'}

# Per-state sigma colours (7 states)
SIGMA_COLOURS = {
    'sigma_px':       '#e63946',   # red
    'sigma_py':       '#f4a261',   # orange
    'sigma_pz':       '#2a9d8f',   # teal
    'sigma_v':        '#457b9d',   # steel blue
    'sigma_a':        '#8ecae6',   # light blue
    'sigma_yaw':      '#6a4c93',   # purple
    'sigma_yaw_rate': '#a8dadc',   # pale cyan
}
SIGMA_LABELS = {
    'sigma_px':       'σ px (m)',
    'sigma_py':       'σ py (m)',
    'sigma_pz':       'σ pz (m)',
    'sigma_v':        'σ v (m/s)',
    'sigma_a':        'σ a (m/s²)',
    'sigma_yaw':      'σ yaw (rad)',
    'sigma_yaw_rate': 'σ ω (rad/s)',
}

if PAPER_STYLE:
    # Linear states (solid) and angular states (dashed) along viridis
    SIGMA_COLOURS = {k: cmap_colour(v) for k, v in [
        ('sigma_px', 0.05), ('sigma_py', 0.22), ('sigma_pz', 0.40), ('sigma_v', 0.58),
        ('sigma_a', 0.76), ('sigma_yaw', 0.15), ('sigma_yaw_rate', 0.65)]}

# Controller state groups (state_code // 1000)
STATE_LABELS = {
    0: "Pre-arm",
    1: "Take-off",
    2: "Searching for landing pad",
    3: "Maintaining pad lock",
    4: "Beginning landing descent",
    5: "Landing aborted",
    6: "Landing confirmed",
    7: "RTL",
}

# Mission-phase names for the paper figures (latency vs time). Shown without the state
# code; a '\n' wraps a long name onto a second line of the rotated label.
STATE_SHORT = {
    1: "Take-off", 2: "Wait at waypoint", 3: "Chase landing\nvehicle",
    4: "Begin landing\ndescent", 5: "Aborted", 6: "Landed", 7: "RTL", 8: "Pilot override",
}
STATE_SHORT_CODES = {   # exact codes with their own name (override the group above)
    1200: "Go to waypoint",
    2100: "Acquire landing\nvehicle",
}

# Descriptive names for the latency stages in the paper figures, keyed by the internal
# stage label: (legend name, bar label wrapped onto two lines). Colours stay keyed by
# the internal label in STAGE_COLOURS.
STAGE_NAMES = {
    'Transport':                ('Image transport',             'Image\ntransport'),
    'Queue':                    ('Detector queue',              'Detector\nqueue'),
    'Inference':                ('YOLO inference',              'YOLO\ninference'),
    'Post-process':             ('Detection post-processing',   'Post-\nprocessing'),
    'Inference + post-process': ('YOLO detection',              'YOLO\ndetection'),
    'Processing':               ('Tag detection',               'Tag\ndetection'),
    'Wait':                     ('Wait for UKF tick',           'Wait for\nUKF tick'),
    'Wait + inference':         ('Detection + wait for UKF',    'Detection\n+ wait'),
    'UKF update':               ('UKF update (incl. OOSM)',     'UKF\nupdate'),
    'Total':                    ('Total (image to estimate)',   'Total'),
}

# ── Helpers ────────────────────────────────────────────────────────────────
def clean_true_position(x, y, t, thresh=OUTLIER_SPEED_THRESH, passes=3):
    """
    Remove single-frame position glitches from the true trajectory.

    A row is flagged as an outlier when it jumps away from its neighbours and
    back: its distance from the straight line between the previous and next rows,
    divided by the step time, exceeds `thresh`. Fast but smooth relative motion (the
    drone closing on a 50 km/h car) stays on that line and is kept; a speed test on
    each side alone would flag it. Flagged points are replaced by linear
    interpolation. Multiple passes handle back-to-back glitches.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    t = np.asarray(t, dtype=float)
    bad_global = np.zeros(len(x), dtype=bool)

    for _ in range(passes):
        bad = np.zeros(len(x), dtype=bool)
        if len(x) > 2:
            span = t[2:] - t[:-2]
            w = np.where(span > 0, (t[1:-1] - t[:-2]) / np.where(span > 0, span, 1.0), 0.5)
            mx = x[:-2] + w * (x[2:] - x[:-2])
            my = y[:-2] + w * (y[2:] - y[:-2])
            dt = np.maximum(np.minimum(t[1:-1] - t[:-2], t[2:] - t[1:-1]), 1e-3)
            bad[1:-1] = np.hypot(x[1:-1] - mx, y[1:-1] - my) / dt > thresh
        bad_global |= bad

        idx  = np.arange(len(x))
        good = ~bad_global
        x = np.interp(idx, idx[good], x[good])
        y = np.interp(idx, idx[good], y[good])

    return x, y, bad_global


def clean_true_yaw(yaw, t, thresh=np.deg2rad(30), passes=3):
    """
    Remove single-frame yaw glitches from the true trajectory.

    A row is flagged when the angular velocity entering and leaving
    the row both exceed the threshold. The angle is then replaced by
    interpolation on the unwrapped yaw trajectory.
    """
    yaw = yaw.copy().astype(float)
    fin = np.isfinite(yaw)
    if fin.sum() < 3:                      # no truth (or too little to clean)
        return yaw, np.zeros(len(yaw), dtype=bool)
    yaw[fin] = np.unwrap(yaw[fin])         # NaN-safe: a NaN would poison the whole unwrap
    yaw = np.interp(np.arange(len(yaw)), np.flatnonzero(fin), yaw[fin])
    bad_global = np.zeros(len(yaw), dtype=bool)

    for _ in range(passes):
        yaw_rate = np.abs(np.diff(yaw) / np.diff(t))
        bad = np.zeros(len(yaw), dtype=bool)
        bad[1:-1] = (yaw_rate[:-1] > thresh) & (yaw_rate[1:] > thresh)
        bad_global |= bad

        idx = np.arange(len(yaw))
        good = ~bad_global
        yaw = np.interp(idx, idx[good], yaw[good])

    return np.where(fin, wrap_angle(yaw), np.nan), bad_global


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def new_measurement_mask(stamp):
    """ The CSV is logged at the controller rate, so a raw measurement is repeated
        across rows until the next one arrives. True only where the stamp changes.
    """
    stamp = np.asarray(stamp)
    is_new = np.ones(len(stamp), dtype=bool)
    is_new[1:] = stamp[1:] != stamp[:-1]
    return (stamp > 0) & is_new


def latency_streams(df):
    """ Per-measurement pipeline latency samples, one dict per stream present:
            name, marker, t (row time, s), stages [(label, ms array)], total (ms array)

        New logs fill the latency columns on one row per measurement (NaN elsewhere),
        so the samples are simply the rows where the total is finite. Legacy logs
        repeated the old three stage columns on every row; there, keep the first row of
        each raw stamp and take the total as the sum of the stages.
    """
    t_all = df['timestamp'].to_numpy()
    out = []

    def stage_values(col):
        """ A stage column, or NaN if this log predates it. """
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
        return np.full(len(df), np.nan)

    def frozen(v):
        """ True if a stage reads 0 on (nearly) every sample: a frozen sim clock. """
        v = v[np.isfinite(v)]
        return v.size >= 5 and np.mean(v == 0.0) >= FROZEN_ZERO_FRACTION

    for name, prefix, marker, stages in LAG_STREAMS:
        total_col = f'{prefix}_total_lag'
        if total_col in df.columns:
            rows = np.isfinite(df[total_col].to_numpy(dtype=float))
            if rows.any():
                vals = {suffix: stage_values(f'{prefix}_{suffix}_lag')[rows]
                        for suffix, _ in stages}
                labelled = [(label, vals[suffix]) for suffix, label in stages
                            if np.isfinite(vals[suffix]).any()]

                # Older SITL logs: drop stages timed against a frozen sim clock
                for probe, rule in FROZEN_RULES.get(name, []):
                    if probe in vals and frozen(vals[probe]):
                        drop = set(rule.get('drop', []))
                        relabel = rule.get('relabel', {})
                        print(f"NOTE: {name} {probe} lag reads 0 on every sample "
                              f"(sim clock frozen inside the callback in this log): "
                              f"dropped {sorted(drop)}"
                              + (f", relabelled {relabel}" if relabel else '') + '.')
                        labelled = [(relabel.get(lbl, lbl), v) for lbl, v in labelled
                                    if lbl not in drop]

                # Fold negligible stages into their neighbour
                for small, into in MERGE_IF_NEGLIGIBLE.items():
                    lbls = [lbl for lbl, _ in labelled]
                    if small not in lbls or into not in lbls:
                        continue
                    v_small = labelled[lbls.index(small)][1]
                    finite = v_small[np.isfinite(v_small)]
                    if finite.size and finite.mean() < NEGLIGIBLE_STAGE_MS:
                        i = lbls.index(into)
                        v_into = labelled[i][1]
                        merged = np.where(np.isfinite(v_small), v_into + v_small, v_into)
                        label = f'{into} + {small.lower()}'
                        labelled[i] = (label, merged)
                        labelled = [(l, v) for l, v in labelled if l != small]
                        print(f"NOTE: {name} {small} mean {finite.mean():.2f} ms "
                              f"< {NEGLIGIBLE_STAGE_MS} ms: merged into '{label}'.")

                out.append(dict(
                    name=name, marker=marker, t=t_all[rows], stages=labelled,
                    total=df[total_col].to_numpy(dtype=float)[rows],
                ))
            continue

        cols, stamp_col = LEGACY_LAG[name]
        if not all(c in df.columns for c in cols + [stamp_col]):
            continue
        rows = (new_measurement_mask(df[stamp_col].to_numpy())
                & df[cols].notna().all(axis=1).to_numpy())
        if not rows.any():
            continue
        vals = [df[c].to_numpy(dtype=float)[rows] for c in cols]
        out.append(dict(name=name, marker=marker, t=t_all[rows],
                        stages=list(zip(LEGACY_STAGE_LABELS, vals)),
                        total=np.sum(vals, axis=0)))

    return out


def add_state_transitions(ax, df, time_col='timestamp', label=True, short=False):
    """
    Mark controller-state changes with dotted vertical lines.

    If `label` is True the state name/code is also written next to each line
    (only do this on one panel of a stacked figure to avoid clutter). `short` uses
    STATE_SHORT names, for panels too small for the full ones.
    """
    state_col = 'controller_state'
    if state_col not in df.columns:
        return

    states = pd.to_numeric(df[state_col], errors='coerce')
    changed = states.notna() & states.ne(states.shift())

    for row_idx in np.flatnonzero(changed.to_numpy()):
        state_code = int(states.iloc[row_idx])
        if short:
            state_name = STATE_SHORT_CODES.get(
                state_code, STATE_SHORT.get(state_code // 1000, 'State'))
        else:
            state_name = STATE_LABELS.get(state_code // 1000, 'Controller state')
        transition_t = float(df[time_col].iloc[row_idx])

        ax.axvline(transition_t, color='#555555', linewidth=LW(0.9),
                   linestyle=':', alpha=0.65, zorder=0)
        if label:
            ax.annotate(
                state_name if short else f'{state_name} ({state_code})',
                xy=(transition_t, 0.99),
                xycoords=('data', 'axes fraction'),
                xytext=(3, -2),
                textcoords='offset points',
                rotation=90, ha='left', va='top', multialignment='left',
                fontsize=6 if PAPER_STYLE else 7, color='#333333',
                linespacing=1.0,
                annotation_clip=False,
                # Short labels sit over the data: a white backing keeps them legible
                bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85) if short else None,
                zorder=4,
            )


# ── Figure helpers (PAPER_STYLE switches between the two looks) ─────────────

def new_figure(nrows, ncols, paper_size, legacy_size, **kw):
    """ Subplots at journal size with constrained layout (PAPER_STYLE), else the
        original size; tight_layout is then applied in save_figure. Works on
        matplotlib 3.4+ (ROS 2 Humble ships 3.5). """
    if not PAPER_STYLE:
        return plt.subplots(nrows, ncols, figsize=legacy_size, **kw)
    fig, axes = plt.subplots(nrows, ncols, figsize=paper_size, constrained_layout=True, **kw)
    pads = dict(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.03)
    if hasattr(fig, 'get_layout_engine'):
        fig.get_layout_engine().set(**pads)
    else:
        fig.set_constrained_layout_pads(**pads)
    return fig, axes


def FS(size):
    """ Explicit font size in the diagnostic look; None (= PAPER_RC) under PAPER_STYLE. """
    return None if PAPER_STYLE else size


def LW(width):
    """ Line width, scaled to PAPER_RC['lines.linewidth'] (0.8 vs the original 1.0). """
    return width * (plt.rcParams['lines.linewidth'] if PAPER_STYLE else 1.0)


def MSZ(size):
    """ Line-marker size, scaled to PAPER_RC['lines.markersize'] (2.6 vs ~6). """
    return size * (plt.rcParams['lines.markersize'] / 6.0 if PAPER_STYLE else 1.0)


def SA(area):
    """ Scatter marker area (pt²), scaled like MSZ (area goes with size squared). """
    return area * ((plt.rcParams['lines.markersize'] / 6.0) ** 2 * 2.0 if PAPER_STYLE else 1.0)


def figure_title(fig, text, **kw):
    """ Figure title in the diagnostic look only: in a paper the caption does that job. """
    if not PAPER_STYLE:
        fig.suptitle(text, **kw)


def grid(ax, **kw):
    """ Gridlines unless PAPER_STYLE turns them off (SHOW_GRID). """
    if SHOW_GRID or not PAPER_STYLE:
        ax.grid(True, alpha=0.3, **kw)


def legend_kw(fontsize=8):
    """ batch_analysis.py legend look under PAPER_STYLE, else plain text at fontsize. """
    if not PAPER_STYLE:
        return dict(fontsize=fontsize)
    return dict(fontsize=LEGEND_FONTSIZE, frameon=LEGEND_BOX, fancybox=LEGEND_ROUND,
                edgecolor=LEGEND_EDGE, framealpha=1.0, borderpad=0.4, handletextpad=0.5)


def legend(ax, *args, fontsize=8, **kw):
    leg = ax.legend(*args, **legend_kw(fontsize), **kw)
    if PAPER_STYLE and leg is not None:
        leg.get_frame().set_linewidth(plt.rcParams['axes.linewidth'])
    return leg


def panel_title(ax, letter, text):
    """ '(a) text' at the left under PAPER_STYLE, as in batch_analysis.py. """
    if PAPER_STYLE:
        ax.set_title(f'({letter}) {text}', loc='left', fontweight='bold')
    else:
        ax.set_title(text)


def save_figure(fig, out_dir, name, legacy_rect=None):
    """ PDF (+ PNG) under PAPER_STYLE, else the original single PNG. """
    if not PAPER_STYLE and legacy_rect is not None:
        fig.tight_layout(rect=legacy_rect)
    for ext in (SAVE_FORMATS if PAPER_STYLE else ('png',)):
        path = out_dir / f'{name}.{ext}'
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {path}")


def save_legend(handles, labels, out_dir, name, width):
    """ A figure's legend on its own, saved at exactly `width` (height fitted), as in
        batch_analysis.py: include it at the same width as its figure and the text stays
        at LEGEND_FONTSIZE. Columns are dropped until it fits, so it wraps onto more rows.
    """
    fl = plt.figure(figsize=(width, 2.0))
    ncol = len(handles)
    while True:
        leg = fl.legend(handles=handles, labels=labels, ncol=ncol, loc='center',
                        borderaxespad=0.0, **legend_kw())
        leg.get_frame().set_linewidth(plt.rcParams['axes.linewidth'])
        for line in leg.get_lines():   # thin data lines read poorly as legend keys
            line.set_linewidth(max(line.get_linewidth(), 1.2))
        fl.canvas.draw()
        if ncol == 1 or leg.get_window_extent().width <= fl.bbox.width - 2:
            break
        leg.remove()
        ncol -= 1
    h = leg.get_window_extent().height / fl.dpi
    fl.set_size_inches(width, h + 2 * plt.rcParams['savefig.pad_inches'] + 0.02)
    for ext in SAVE_FORMATS:
        path = out_dir / f'{name}_legend.{ext}'
        fl.savefig(path, dpi=DPI)          # no tight crop: exact width
        print(f"Saved: {path}")
    plt.close(fl)


# ── Main ───────────────────────────────────────────────────────────────────

def main(csv_path=None, out_dir=None, window=None, show=True, fleet='auto', rtk_bias=None):
    """ Analyse one controller CSV. All arguments are optional, so running the script
        directly behaves exactly as before; batch_analysis.py calls it with them set.

    :param csv_path: CSV to read (default: CSV_FILE in LOG_DIR, else the newest
                     controller_*.csv there; a bare file name is looked up in LOG_DIR)
    :param out_dir:  Folder for the figures (default: ./plots next to this script)
    :param window:   (t_start, t_end) in seconds since the start of the log. Overrides
                     WINDOW and the auto start/end.
    :param show:     Open the figures in a window (False: save and close)
    :param fleet:    Real flights: RTK telemetry CSV, 'auto' (search next to the log) or
                     None (no truth)
    :param rtk_bias: Real flights: (E, N) RTK frame offset in m to remove, or 'auto'
    """
    with plt.rc_context(PAPER_RC if PAPER_STYLE else {}):
        _main(csv_path, out_dir, window, show, fleet, rtk_bias)


def _main(csv_path, out_dir, window, show, fleet='auto', rtk_bias=None):
    """ main() body, run inside the figure style context. """
    script_dir = Path(__file__).parent
    csv_path = Path(resolve_csv(csv_path, LOG_DIR, CSV_FILE, script_dir))

    out_dir = Path(out_dir) if out_dir is not None else script_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {csv_path.name}")
    df = load_log(csv_path, fleet=fleet, rtk_bias=rtk_bias)
    platform = df.attrs['platform']
    truth_source = df.attrs['truth_source']
    TRUE_OFFSET_X_BODY, TRUE_OFFSET_Y_BODY, TRUE_OFFSET_Z_BODY = df.attrs['pad_in_car']
    T_START, T_END = WINDOW[platform]
    print(f"Platform: {df.attrs['platform_name']} ({platform}); truth: {describe_truth(df)}")

    # Make time relative to the start of the log. Raw-measurement stamps share the
    # same clock, so keep t0 to convert them too.
    t0 = float(df['timestamp'].iloc[0])
    df['timestamp'] = df['timestamp'] - t0

    has_state = 'controller_state' in df.columns
    group = pd.to_numeric(df['controller_state'], errors='coerce') // 1000 if has_state else None
    t_all = df['timestamp']

    # ── Auto start: first entry into pad lock (3xxx) ──
    t_start = T_START
    if AUTO_START_AT_LOCK and has_state and window is None:
        locked = t_all[group == 3]
        if not len(locked):
            print("Auto start: no 3000 state in log, using T_START.")
        else:
            t_lock = float(locked.iloc[0])
            if t_start is None or t_lock > t_start:
                t_start = t_lock
                print(f"Auto start: analysis starts at t = {t_lock:.2f} s, pad lock (3000).")
            else:
                print(f"Auto start: pad lock at t = {t_lock:.2f} s is before T_START, ignored.")

    # ── Auto end: first landing confirmed (6xxx), else last abort (5xxx) ──
    t_end = T_END
    if AUTO_END_AT_LANDING and has_state and window is None:
        landed = t_all[group == 6]
        abort_starts = t_all[(group == 5) & (group.shift() != 5)]
        t_cut, why = None, ''
        if len(landed):
            t_cut, why = float(landed.iloc[0]), 'landing confirmed (6000)'
        elif len(abort_starts):
            t_cut, why = float(abort_starts.iloc[-1]), 'last landing abort (5000)'
        else:
            # Real flights: tracking ends at RTL, pilot override or a lost pad after lock
            after = t_all >= (t_start if t_start is not None else -np.inf)
            ends = []
            for g, name in ((7, 'RTL (7000)'), (8, 'pilot override (8000)')):
                hit = t_all[after & (group == g)]
                if len(hit):
                    ends.append((float(hit.iloc[0]), name))
            lost = t_all[after & (group == 2) & (group.shift() >= 3)]
            if len(lost):
                ends.append((float(lost.iloc[0]), 'pad lost after lock (2000)'))
            if ends:
                t_cut, why = min(ends)

        if t_cut is None:
            print("Auto end: no 6000/5000/7000/8000 or pad-lost state in log, using T_END.")
        elif t_start is not None and t_cut <= t_start:
            print(f"Auto end: {why} at t = {t_cut:.2f} s is before the start, ignored.")
        elif t_end is None or t_cut < t_end:
            t_end = t_cut
            print(f"Auto end: analysis ends at t = {t_cut:.2f} s, {why}.")

    # Explicit window (batch runs) replaces the limits above
    if window is not None:
        t_start, t_end = window
        print(f"Window: t = {t_start:.2f} s to {t_end:.2f} s (set by caller).")

    # ── Latency window: first entry into LATENCY_FROM_STATE -> same end as the errors ──
    # Taken from the full log before it is cut to the error window. An explicit window
    # (batch runs) applies to latency too.
    lat_start = t_start
    if LATENCY_FROM_STATE is not None and has_state and window is None:
        state_num = pd.to_numeric(df['controller_state'], errors='coerce')
        hit = t_all[state_num >= LATENCY_FROM_STATE]
        if len(hit):
            t_hit = float(hit.iloc[0])
            lat_start = t_hit if T_START is None else max(T_START, t_hit)
            end_txt = f"{t_end:.2f} s" if t_end is not None else "end of log"
            print(f"Latency window: t = {lat_start:.2f} s (state {LATENCY_FROM_STATE}) "
                  f"to {end_txt}.")
        else:
            print(f"Latency window: no {LATENCY_FROM_STATE} state in log, "
                  f"using the error window.")

    ts = df['timestamp']
    lat_mask = np.ones(len(df), dtype=bool)
    if lat_start is not None:
        lat_mask &= (ts >= lat_start).to_numpy()
    if t_end is not None:
        lat_mask &= (ts <= t_end).to_numpy()
    df_lat = df[lat_mask].reset_index(drop=True)

    if t_start is not None:
        df = df[df['timestamp'] >= t_start]
    if t_end is not None:
        df = df[df['timestamp'] <= t_end]
    df = df.reset_index(drop=True)

    # ── Truth availability: RTK logs can start/end inside the window ──────
    truth_ok = np.isfinite(df['landing_pad_rel_true_x'].to_numpy(float)) & \
        np.isfinite(df['landing_pad_glob_true_yaw'].to_numpy(float))
    has_truth = bool(truth_ok.any())
    if has_truth and not truth_ok.all():
        cov = df['timestamp'][truth_ok]
        print(f"Truth covers t = {cov.iloc[0]:.2f} - {cov.iloc[-1]:.2f} s of the window: "
              f"error analysis cropped to it.")
        df = df[(df['timestamp'] >= cov.iloc[0]) & (df['timestamp'] <= cov.iloc[-1])]
        df = df.reset_index(drop=True)
    if not has_truth:
        print("NOTE: no ground truth for this log (real flight without an RTK fleet log) - "
              "error and trajectory figures skipped; consistency (NIS) and latency kept.")
    if len(df) == 0:
        print("Nothing left in the analysis window - stopping.")
        return

    # ── Clean outlier spikes in the ground-truth trajectory ──────────────
    # Relative (pad - quad) truth is logged directly. The car's heading must be
    # the ABSOLUTE yaw: the relative X/Y are in the world-aligned local frame, while
    # the `*_rel_yaw` columns have the quad's yaw subtracted.
    t_arr = df['timestamp'].to_numpy()
    true_x, true_y, outlier_mask = clean_true_position(
        df['landing_pad_rel_true_x'].to_numpy(),
        df['landing_pad_rel_true_y'].to_numpy(),
        t_arr,
    )
    true_yaw, _ = clean_true_yaw(df['landing_pad_glob_true_yaw'].to_numpy(), t_arr)
    true_z = df['landing_pad_rel_true_z'].to_numpy(dtype=float).copy()
    true_z += TRUE_OFFSET_Z_BODY

    # ── Apply configurable body-frame XY offset ────────────────────────
    cos_yaw, sin_yaw = np.cos(true_yaw), np.sin(true_yaw)
    true_x += TRUE_OFFSET_X_BODY * cos_yaw - TRUE_OFFSET_Y_BODY * sin_yaw
    true_y += TRUE_OFFSET_X_BODY * sin_yaw + TRUE_OFFSET_Y_BODY * cos_yaw

    n_outliers = outlier_mask.sum()
    if n_outliers:
        print(f"Removed {n_outliers} outlier row(s) from true position "
              f"(interpolated over glitches at t = "
              f"{np.round(t_arr[outlier_mask], 2).tolist()})")
    if 'nees' in df.columns:
        df.loc[outlier_mask, 'nees'] = np.nan

    # ── Truth at arbitrary times (controller rows or measurement stamps) ──
    # cos/sin are interpolated rather than the wrapped angle so there's no glitch
    # when yaw crosses ±pi.
    truth = interp1d(
        t_arr, np.vstack([true_x, true_y, true_z, cos_yaw, sin_yaw]),
        kind='linear', axis=1, bounds_error=False, fill_value='extrapolate',
    )

    def errors_at(t, est_x, est_y):
        """ Car-frame (long, lat) and |XY| errors vs truth at times t, plus truth
            z and yaw at those times. """
        tx, ty, tz, c, s = truth(t)
        ex, ey = est_x - tx, est_y - ty
        return (c * ex + s * ey, -s * ex + c * ey, np.hypot(ex, ey),
                tz, np.arctan2(s, c))

    # ── Restrict to rows where a measurement exists ───────────────────────
    present = [s for s in STREAMS if f'{s[3]}_raw_stamp' in df.columns]
    measurement_mask = np.logical_or.reduce(
        [df[f'{p}_raw_stamp'].to_numpy() > 0 for *_, p in present]
    )
    df_valid = df[measurement_mask].copy().reset_index(drop=True)
    ctrl_t = df_valid['timestamp'].to_numpy()
    if len(df_valid) == 0:
        print("No detector measurements in the analysis window (car never seen) - nothing "
              "to analyse.")
        return

    # ── Filter errors (estimate valid at the UKF tick, not the log row) ───
    # est_time shares the timestamp clock but was not shifted by t0, and is 0.0
    # until the UKF loop first runs, so fall back to the row time there.
    if 'est_time' in df_valid.columns:
        est_raw = pd.to_numeric(df_valid['est_time'], errors='coerce').to_numpy(dtype=float)
        est_t = np.where(est_raw > 0, est_raw - t0, ctrl_t)
    else:
        print("WARNING: 'est_time' column not found – filter scored at the row time.")
        est_t = ctrl_t
    dt_est = (ctrl_t - est_t) * 1000.0
    print(f"Row time - estimate time: mean {np.nanmean(dt_est):.1f} ms, "
          f"max {np.nanmax(dt_est):.1f} ms")

    filt_err_long, filt_err_lat, filt_err_abs, tz, tyaw = errors_at(
        est_t,
        df_valid['landing_pad_rel_x'].to_numpy(),
        df_valid['landing_pad_rel_y'].to_numpy(),
    )
    filt_err_z = df_valid['landing_pad_rel_z'].to_numpy() - tz
    filt_yaw_err = wrap_angle(df_valid['landing_pad_glob_yaw'].to_numpy() - tyaw)

    # ── Raw measurement errors (truth taken at each measurement's own stamp) ──
    streams = []
    for key, label, colour, prefix in present:
        stamp = df_valid[f'{prefix}_raw_stamp'].to_numpy()
        x = df_valid[f'{prefix}_rel_x_raw'].to_numpy(dtype=float)
        y = df_valid[f'{prefix}_rel_y_raw'].to_numpy(dtype=float)
        valid = new_measurement_mask(stamp) & np.isfinite(x) & np.isfinite(y)
        t_meas = np.where(stamp > 0, stamp - t0, np.nan)

        e_long, e_lat, e_abs, tz, tyaw = errors_at(t_meas, x, y)
        s = dict(key=key, label=label, colour=colour, valid=valid, t=t_meas,
                 x=x, y=y, long=e_long, lat=e_lat, abs=e_abs)

        # Live baseline: the raw value exactly as a controller using it directly would
        # see it (held until the next measurement), scored against truth at row time.
        held = (stamp > 0) & np.isfinite(x) & np.isfinite(y)
        h_long, h_lat, h_abs, _, _ = errors_at(ctrl_t, x, y)
        s['held'] = held
        s['held_long'] = np.where(held, h_long, np.nan)
        s['held_lat'] = np.where(held, h_lat, np.nan)
        s['held_abs'] = np.where(held, h_abs, np.nan)

        if f'{prefix}_rel_z_raw' in df_valid.columns:
            s['z'] = df_valid[f'{prefix}_rel_z_raw'].to_numpy(dtype=float) - tz
        if f'{prefix}_glob_yaw_raw' in df_valid.columns:
            s['yaw'] = wrap_angle(df_valid[f'{prefix}_glob_yaw_raw'].to_numpy(dtype=float) - tyaw)
        if f'{prefix}_yaw_sigma_raw' in df_valid.columns:
            s['yaw_sigma'] = df_valid[f'{prefix}_yaw_sigma_raw'].to_numpy(dtype=float)
        streams.append(s)

    # ── Filter state at each YOLO image stamp (same instant as the raw point) ──
    post = None
    if 'yolo_post_rel_x' in df_valid.columns and \
            'landing_pad_yolo_car_raw_stamp' in df_valid.columns:
        p_stamp = df_valid['landing_pad_yolo_car_raw_stamp'].to_numpy()
        ppx = df_valid['yolo_post_rel_x'].to_numpy(dtype=float)
        ppy = df_valid['yolo_post_rel_y'].to_numpy(dtype=float)
        p_ok = new_measurement_mask(p_stamp) & np.isfinite(ppx) & np.isfinite(ppy)
        p_t = np.where(p_stamp > 0, p_stamp - t0, np.nan)
        p_long, p_lat, p_abs, _, _ = errors_at(p_t, ppx, ppy)
        post = dict(t=p_t, ok=p_ok, long=p_long, lat=p_lat, abs=p_abs)
        if not p_ok.any():
            post = None
    else:
        print("NOTE: 'yolo_post_rel_*' columns not found – filter-at-image-stamp "
              "comparison skipped.")

    def plot_raw(ax, field, scale=1.0, suffix=''):
        """ Scatter one error field for every raw stream that has it. """
        for s in streams:
            if field not in s:
                continue
            ok = s['valid'] & np.isfinite(s[field])
            if ok.any():
                ax.scatter(s['t'][ok], scale * s[field][ok], color=s['colour'],
                           marker='x', s=SA(22), linewidths=LW(1.0), alpha=ALPHA,
                           label=s['label'] + suffix)

    def plot_held(ax, field):
        """ Step line of the held (live) raw error for every stream that has it. """
        for s in streams:
            v = s.get(f'held_{field}')
            if v is not None and np.isfinite(v).any():
                ax.plot(ctrl_t, v, color=s['colour'], lw=LW(0.8), ls='--',
                        drawstyle='steps-post', alpha=0.7,
                        label=s['label'] + ' held (live)')

    def plot_post(ax, field):
        """ Scatter the filter error at each YOLO image stamp. """
        if post is not None and field in post:
            ok = post['ok'] & np.isfinite(post[field])
            if ok.any():
                ax.scatter(post['t'][ok], post[field][ok], color=FILT_C,
                           marker='o', s=SA(10), alpha=ALPHA, label='Filter @ image stamp')

    # YOLO pixel-space box mode (new logs only)
    has_box_mode = 'yolo_box_mode' in df_valid.columns
    box_mode = (df_valid['yolo_box_mode'].to_numpy() == 1) if has_box_mode else None

    # ── Detect which covariance / latency columns are present ─────────────
    # One sample per measurement (de-duplicated for legacy logs), over the latency
    # window (state LATENCY_FROM_STATE -> landing), not the error window
    lat_streams = latency_streams(df_lat)

    sigma_cols     = [c for c in SIGMA_COLOURS if c in df_valid.columns]
    has_per_state  = len(sigma_cols) > 0
    trace_col      = 'covar_trace' if 'covar_trace' in df_valid.columns else 'filter_covar'
    has_trace_only = (not has_per_state) and (trace_col in df_valid.columns)
    has_nis  = 'nis' in df_valid.columns
    has_nees = 'nees' in df_valid.columns and np.isfinite(df_valid['nees'].to_numpy(float)).any()

    if not lat_streams:
        print("WARNING: no AprilTag or YOLO latency columns found – latency analysis skipped.")
    if not has_per_state and not has_trace_only:
        print("WARNING: no covariance columns found – covar subplot skipped.")
    if 'controller_state' not in df.columns:
        print("WARNING: 'controller_state' column not found – state transitions skipped.")

    if has_truth:
        # ── Figure 1 – Error vs Time ─────────────────────────────────────────
        # Panels: longitudinal, lateral, height, absolute horizontal, yaw, [covariance]
        has_covar_plot = has_per_state or has_trace_only
        n_rows = 6 if has_covar_plot else 5

        fig1, axes = new_figure(n_rows, 1, (FIG_WIDTH_DOUBLE, 1.55 * n_rows),
                                (12, 3.2 * n_rows), sharex=True)
        axes = list(axes)
        figure_title(fig1, 'Landing Pad Position Error vs Time\n(true position outliers removed)',
                     fontsize=14, fontweight='bold', y=0.99)
        ax_lon, ax_lat, ax_z, ax_abs, ax_yaw = axes[:5]
        ax_cov = axes[5] if has_covar_plot else None

        # ── Panels: longitudinal, lateral and height error ────────────────────
        for letter, (ax, fe, field, ylabel, title) in zip('abc', [
            (ax_lon, filt_err_long, 'long', 'Longitudinal error (m)',
             'Longitudinal error  (along car heading, + = estimate ahead of truth)'),
            (ax_lat, filt_err_lat, 'lat', 'Lateral error (m)',
             'Lateral error  (across car heading, + = estimate left of truth)'),
            (ax_z, filt_err_z, 'z', 'Z error (m)',
             'Height (Z) error  (+ = estimate above truth)'),
        ]):
            ax.plot(est_t, fe, color=FILT_C, lw=LW(1.0), alpha=ALPHA, label='Filter estimate')
            plot_raw(ax, field)
            if field in ('long', 'lat'):
                plot_held(ax, field)
                plot_post(ax, field)
            ax.axhline(0, color='0.5', lw=LW(0.6), ls='--')
            ax.set_ylabel(ylabel, fontsize=FS(10))
            panel_title(ax, letter, title)
            grid(ax)

        # ── Panel: Absolute horizontal error + latency secondary axis ───────
        ax_abs.plot(est_t, filt_err_abs, color=FILT_C, lw=LW(1.0), alpha=ALPHA, label='Filter |error|')
        # RTK truth: errors inside the reported truth uncertainty can't be resolved
        if 'truth_sigma_xy' in df.columns and np.isfinite(df['truth_sigma_xy'].to_numpy(float)).any():
            ax_abs.fill_between(df['timestamp'], 0, df['truth_sigma_xy'], color='0.5', alpha=0.18,
                                lw=0, zorder=0, label='Truth 1σ (RTK h_acc)')
        plot_raw(ax_abs, 'abs', suffix=' |error|')
        plot_held(ax_abs, 'abs')
        plot_post(ax_abs, 'abs')
        ax_abs.set_ylim(bottom=0)
        ax_abs.set_ylabel('|error| (m)', fontsize=FS(10))
        panel_title(ax_abs, 'd', 'Absolute horizontal error (Euclidean norm, XY)')
        grid(ax_abs)

        if lat_streams:
            ax_lag = ax_abs.twinx()
            for s in lat_streams:
                # Error window only: the latency window starts earlier and would stretch
                # the shared x-axis
                m = (s['t'] >= ctrl_t[0]) & (s['t'] <= ctrl_t[-1]) if len(ctrl_t) else s['t'] < 0
                ax_lag.plot(s['t'][m], s['total'][m], color=TOTAL_COLOURS[s['name']], lw=LW(0.8),
                            alpha=0.7, ls=':', marker=s['marker'], ms=MSZ(2.5),
                            label=f"{s['name']} total latency")
            ax_lag.set_ylabel('Latency (ms)', fontsize=FS(10))
            ax_lag.tick_params(axis='y', labelcolor='0.35')
            ax_lag.set_ylim(bottom=0)
            lines_l, labels_l = ax_abs.get_legend_handles_labels()
            lines_r, labels_r = ax_lag.get_legend_handles_labels()
            legend(ax_abs, lines_l + lines_r, labels_l + labels_r, fontsize=FS(8),
                   loc='upper right', ncol=2 if PAPER_STYLE else 1)
        else:
            legend(ax_abs, fontsize=FS(8), loc='upper right')

        # ── Panel: Yaw error ───────────────────────────────────────────────
        ax_yaw.plot(est_t, np.degrees(filt_yaw_err), color=FILT_C, lw=LW(1.0), alpha=ALPHA,
                    label='Filter estimate')
        plot_raw(ax_yaw, 'yaw', scale=np.degrees(1.0))
        flip_ringed = False
        for s in streams:
            if 'yaw' not in s:
                continue
            ok = s['valid'] & np.isfinite(s['yaw'])
            if not ok.any():
                continue
            e_deg = np.degrees(s['yaw'][ok])
            # ±1σ of the yaw measurement as fed to the UKF (box yaw only)
            if 'yaw_sigma' in s:
                sig = np.degrees(s['yaw_sigma'][ok])
                ax_yaw.errorbar(s['t'][ok], e_deg, yerr=sig, fmt='none', ecolor=s['colour'],
                                elinewidth=LW(0.7), capsize=1.5, alpha=0.45)
            # Front/back flips: the measurement picked the wrong end of the car
            flip = np.abs(e_deg) > 90.0
            if flip.any():
                ax_yaw.scatter(s['t'][ok][flip], e_deg[flip], s=SA(70), facecolors='none',
                               edgecolors='k', linewidths=LW(0.9), zorder=5,
                               label=None if flip_ringed else 'Front/back flip (|err| > 90°)')
                flip_ringed = True
        ax_yaw.axhline(0, color='0.5', ls='--', lw=LW(0.8))
        for lim in (-90, 90):
            ax_yaw.axhline(lim, color='0.5', ls=':', lw=LW(0.7))
        ax_yaw.set_ylabel('Yaw error (deg)', fontsize=FS(10))
        panel_title(ax_yaw, 'e',
                    'Yaw error  (crosses = raw yaw measurements, bars = ±1σ fed to the UKF)')
        grid(ax_yaw)

        # ── Panel: per-state 1-sigma covariance ──────────────────────────────
        if has_covar_plot:
            if has_per_state:
                # Secondary axis for angular states (rad vs metres on very different scales)
                pos_plotted = [c for c in ['sigma_px', 'sigma_py', 'sigma_pz', 'sigma_v', 'sigma_a']
                               if c in sigma_cols]
                ang_plotted = [c for c in ['sigma_yaw', 'sigma_yaw_rate'] if c in sigma_cols]

                for col in pos_plotted:
                    ax_cov.plot(ctrl_t, df_valid[col].to_numpy(), color=SIGMA_COLOURS[col],
                                lw=LW(1.1), alpha=ALPHA, label=SIGMA_LABELS[col])

                if ang_plotted:
                    ax_ang = ax_cov.twinx()
                    for col in ang_plotted:
                        ax_ang.plot(ctrl_t, df_valid[col].to_numpy(), color=SIGMA_COLOURS[col],
                                    lw=LW(1.1), alpha=ALPHA, ls='--', label=SIGMA_LABELS[col])
                    ax_ang.set_ylabel('1σ (rad, rad/s)' if PAPER_STYLE
                                      else '1σ angular (rad / rad·s⁻¹)', fontsize=FS(9))
                    ax_ang.set_ylim(bottom=0)
                    lines_l, labels_l = ax_cov.get_legend_handles_labels()
                    lines_r, labels_r = ax_ang.get_legend_handles_labels()
                    legend(ax_cov, lines_l + lines_r, labels_l + labels_r,
                           fontsize=FS(8), loc='upper right', ncol=4 if PAPER_STYLE else 2)
                else:
                    legend(ax_cov, fontsize=FS(8), loc='upper right', ncol=2)

                ax_cov.set_ylabel('1σ (m, m/s)' if PAPER_STYLE
                                  else '1σ position / velocity (m, m/s)', fontsize=FS(9))
                panel_title(ax_cov, 'f', 'UKF per-state 1σ covariance  (— linear,  - - angular)')
            else:
                # Fallback: single trace scalar from old CSVs
                ax_cov.plot(ctrl_t, df_valid[trace_col].to_numpy(), color='#c0392b',
                            lw=LW(1.0), alpha=ALPHA, label='trace(P)')
                ax_cov.set_ylabel('trace(P)', fontsize=FS(10))
                panel_title(ax_cov, 'f', 'Filter covariance — trace(P)')
                legend(ax_cov, fontsize=FS(9), loc='upper right')

            ax_cov.set_ylim(bottom=0)
            grid(ax_cov)
        axes[-1].set_xlabel('Time (s)', fontsize=FS(10))

        # ── State transitions and YOLO box-mode shading on every panel ───────
        # Lines on all panels; text labels only on the top one to avoid clutter.
        for k, ax in enumerate(axes):
            add_state_transitions(ax, df, time_col='timestamp', label=(k == 0))
            if has_box_mode and box_mode.any():
                ax.fill_between(ctrl_t, 0, 1, where=box_mode, step='post',
                                transform=ax.get_xaxis_transform(),
                                color=BOX_MODE_C, alpha=0.08, lw=LW(0), zorder=0)

        box_patch = [Patch(color=BOX_MODE_C, alpha=0.2, label='YOLO box update')] \
            if has_box_mode and box_mode.any() else []
        for ax in (ax_lon, ax_lat, ax_z, ax_yaw):
            handles, labels = ax.get_legend_handles_labels()
            legend(ax, handles + box_patch, labels + [p.get_label() for p in box_patch],
                   fontsize=FS(8), loc='upper right', ncol=2 if PAPER_STYLE else 1)

        save_figure(fig1, out_dir, 'error_vs_time', legacy_rect=[0, 0, 1, 0.97])

        # ── Figure 2 – 2-D Trajectory ─────────────────────────────────────────
        fig2, ax2 = new_figure(1, 1, (FIG_WIDTH_SINGLE, FIG_WIDTH_SINGLE), (9, 8))
        figure_title(fig2, 'Landing Pad 2D Trajectory\n(true position outliers removed)',
                     fontsize=14, fontweight='bold')

        ax2.plot(true_x, true_y, color=TRUE_C, lw=LW(1.8), label='True position', zorder=3)
        ax2.plot(df_valid['landing_pad_rel_x'], df_valid['landing_pad_rel_y'],
                 color=FILT_C, lw=LW(1.2), alpha=0.85, label='Filter estimate', zorder=4)
        for s in streams:
            if s['valid'].any():
                ax2.scatter(s['x'][s['valid']], s['y'][s['valid']], c=s['colour'],
                            marker='x', s=SA(22), alpha=0.65, label=s['label'], zorder=2)

        if PAPER_STYLE:
            ax2.plot(true_x[0], true_y[0], 'o', mfc='white', mec=TRUE_C, ms=MSZ(8),
                     label='Start', zorder=5)
            ax2.plot(true_x[-1], true_y[-1], 's', color=TRUE_C, ms=MSZ(8), label='End', zorder=5)
        else:
            ax2.plot(true_x[0],  true_y[0],  'go', ms=MSZ(8), label='Start', zorder=5)
            ax2.plot(true_x[-1], true_y[-1], 'rs', ms=MSZ(8), label='End',   zorder=5)
        ax2.set_xlabel('X position (m)', fontsize=FS(11))
        ax2.set_ylabel('Y position (m)', fontsize=FS(11))
        legend(ax2, fontsize=FS(9))
        ax2.set_aspect('equal', adjustable='datalim')
        grid(ax2)
        save_figure(fig2, out_dir, 'trajectory_2d', legacy_rect=[0, 0, 1, 0.96])

    # ── Figure 3 – NIS / NEES consistency analysis ──────────────────────────
    # NIS is the last value held on every row; only count it when it changes, so
    # each measurement is counted once. Both measurement models are 4-dimensional.
    # NEES is evaluated every row over (px, py, yaw) -> 3 DOF.
    nis_bound, nees_bound = chi2.ppf(0.95, df=4), chi2.ppf(0.95, df=3)
    if has_nis:
        nis_all = df_valid['nis'].to_numpy(dtype=float)
        nis_new = np.isfinite(nis_all) & np.r_[True, nis_all[1:] != nis_all[:-1]]
        nis_t, nis = ctrl_t[nis_new], nis_all[nis_new]
    if has_nees:
        nees = df_valid['nees'].to_numpy(dtype=float)
        nees_finite = nees[np.isfinite(nees)]

    if has_nis or has_nees:
        fig3, axes3 = new_figure(2, 2, (FIG_WIDTH_DOUBLE, 4.6), (12, 8))
        figure_title(fig3, 'UKF Consistency Analysis', fontsize=14, fontweight='bold')
        ax_nis_time, ax_nees_time = axes3[0]
        ax_nis_hist, ax_nees_hist = axes3[1]

        nis_c, nees_c = ((cmap_colour(0.15), cmap_colour(0.55)) if PAPER_STYLE
                         else ('#8e44ad', '#2980b9'))
        bound_c, mean_c = ('0.15', '0.5') if PAPER_STYLE else ('red', 'green')
        for on, ax_t, ax_h, name, t_v, vals, dof, bound, colour, letters in [
            (has_nis,  ax_nis_time,  ax_nis_hist,  'NIS',
             nis_t if has_nis else None, nis if has_nis else None, 4, nis_bound, nis_c, 'ac'),
            (has_nees, ax_nees_time, ax_nees_hist, 'NEES',
             ctrl_t, nees_finite if has_nees else None, 3, nees_bound, nees_c, 'bd'),
        ]:
            if not on:
                continue
            series = nis if name == 'NIS' else nees
            ax_t.plot(t_v, series, color=colour, lw=LW(1.2), label=name,
                      marker='.' if name == 'NIS' else None, ms=MSZ(3))
            ax_t.axhline(bound, color=bound_c, linestyle='--', lw=LW(1.0),
                         label=f'95% χ²({dof})')
            ax_t.axhline(dof, color=mean_c, linestyle=':', lw=LW(1.0), label='Expected mean')
            panel_title(ax_t, letters[0], f'{name} vs time')
            ax_t.set_xlabel('Time (s)')
            ax_t.set_ylabel(name)
            grid(ax_t)
            legend(ax_t, )

            if len(vals) > 0:
                ax_h.hist(vals, bins=100, density=True, alpha=0.6, color=colour,
                          label=f'Measured {name}')
                x = np.linspace(0, max(4 * dof, vals.max()), 400)
                ax_h.plot(x, chi2.pdf(x, df=dof), color=bound_c, lw=LW(2), label=f'χ²({dof})')
                ax_h.axvline(vals.mean(), color=mean_c, linestyle='--', lw=LW(1.0),
                             label=f'Mean = {vals.mean():.2f}')
            panel_title(ax_h, letters[1], f'{name} distribution')
            ax_h.set_xlabel(name)
            ax_h.set_ylabel('Density')
            grid(ax_h)
            legend(ax_h, )

        save_figure(fig3, out_dir, 'consistency_analysis', legacy_rect=[0, 0, 1, 0.95])

    # ── Figure 4 – Pipeline latency analysis ───────────────────────────────
    # PAPER_STYLE: single column, (a) latency vs time above (b) the mean breakdown,
    # one shared y label, and the legend saved on its own (pipeline_latency_legend).
    # Otherwise the original side-by-side diagnostic figure.
    if lat_streams:
        if PAPER_STYLE:
            fig4, (ax_lag_time, ax_lag_bar) = new_figure(2, 1, (FIG_WIDTH_SINGLE, 4.4), None)
        else:
            fig4, (ax_lag_time, ax_lag_bar) = new_figure(1, 2, None, (15, 5))
        figure_title(fig4, 'Pipeline Latency Analysis', fontsize=14, fontweight='bold')
        # One stream (the usual case): no 'YOLO' prefix on every label
        prefix = (lambda s: '') if len(lat_streams) == 1 else (lambda s: f"{s['name']} ")

        def short(label, s):
            """ Legend name: descriptive STAGE_NAMES under PAPER_STYLE. """
            if not PAPER_STYLE:
                return f"{prefix(s)}{label}"
            return prefix(s) + STAGE_NAMES.get(label, (label, label))[0]

        def bar_label(label, s):
            """ Bar tick label, wrapped onto two lines to fit a journal-width panel. """
            if not PAPER_STYLE:
                return f"{prefix(s)}{label}"
            return prefix(s) + STAGE_NAMES.get(label, (label, label))[1]

        # Paper look: time measured from the start of the latency window (vehicle
        # acquisition), so the axis starts at 0. The diagnostic look keeps log time so
        # it lines up with the other figures.
        t_lag0 = float(df_lat['timestamp'].iloc[0]) if PAPER_STYLE else 0.0

        bar_specs = []   # (label, values, colour)
        for s in lat_streams:
            for label, v in s['stages']:
                ok = np.isfinite(v)
                if PAPER_STYLE:   # thin line per stage, no markers
                    ax_lag_time.plot(s['t'][ok] - t_lag0, v[ok], lw=LW(0.6), alpha=0.9,
                                     color=STAGE_COLOURS[label],
                                     label=short(label, s))
                else:
                    ax_lag_time.plot(s['t'][ok], v[ok], ls='none', marker=s['marker'],
                                     ms=MSZ(2.5), alpha=0.75, color=STAGE_COLOURS[label],
                                     label=f"{prefix(s)}{label}")
                bar_specs.append((bar_label(label, s), v, STAGE_COLOURS[label]))
            total_label = (short('Total', s) if PAPER_STYLE
                           else f"{prefix(s)}total" if prefix(s) else 'Total')
            if PAPER_STYLE:
                ax_lag_time.plot(s['t'] - t_lag0, s['total'], lw=LW(0.9),
                                 color=TOTAL_COLOURS[s['name']],
                                 label=total_label, zorder=3)
            else:
                ax_lag_time.plot(s['t'], s['total'], lw=LW(0.9), marker=s['marker'],
                                 ms=MSZ(2.5), alpha=0.9, color=TOTAL_COLOURS[s['name']],
                                 label=total_label)
            bar_specs.append((bar_label('Total', s) if PAPER_STYLE else total_label,
                              s['total'], TOTAL_COLOURS[s['name']]))

        # State markers only in the diagnostic look: the paper panel is busy enough
        if not PAPER_STYLE:
            add_state_transitions(ax_lag_time, df_lat, time_col='timestamp', label=False)
        panel_title(ax_lag_time, 'a', 'Latency vs time' if PAPER_STYLE
                    else 'Latency per measurement')
        ax_lag_time.set_xlabel('Time (s)')
        grid(ax_lag_time)
        if PAPER_STYLE:
            top = np.nanmax([np.nanmax(s['total']) for s in lat_streams])
            ax_lag_time.set_ylim(0, 1.08 * top)
            ax_lag_time.set_xlim(0.0, float(df_lat['timestamp'].iloc[-1]) - t_lag0)
        else:
            ax_lag_time.set_ylabel('Latency (ms)')
            ax_lag_time.set_ylim(bottom=0)
            legend(ax_lag_time, fontsize=7, ncol=2)

        def finite(v):
            return v[np.isfinite(v)]

        mean_lags = [finite(v).mean() if finite(v).size else np.nan for _, v, _ in bar_specs]
        std_lags = [finite(v).std() if finite(v).size else np.nan for _, v, _ in bar_specs]
        bars = ax_lag_bar.bar([lbl for lbl, _, _ in bar_specs], mean_lags,
                              yerr=std_lags, capsize=None if PAPER_STYLE else 3,
                              error_kw=dict(elinewidth=LW(1.0), capthick=LW(1.0)),
                              color=[c for *_, c in bar_specs],
                              alpha=1.0 if PAPER_STYLE else 0.85)
        panel_title(ax_lag_bar, 'b', 'Mean latency breakdown (±1σ)')
        ax_lag_bar.tick_params(axis='x', labelsize=0.72 * plt.rcParams['font.size']
                               if PAPER_STYLE else 8, length=0)
        grid(ax_lag_bar, axis='y')
        if PAPER_STYLE:
            fig4.supylabel('Latency (ms)', fontsize=plt.rcParams['axes.labelsize'])
        else:
            ax_lag_bar.set_ylabel('Latency (ms)')

        top = np.nanmax(np.add(mean_lags, std_lags)) if np.isfinite(mean_lags).any() else 1.0
        ax_lag_bar.set_ylim(0, 1.12 * top)
        for bar, value, sd in zip(bars, mean_lags, std_lags):
            if np.isfinite(value):
                y = value + (np.nan_to_num(sd) if PAPER_STYLE else 0.0) + 0.02 * top
                ax_lag_bar.text(bar.get_x() + bar.get_width() / 2, y, f'{value:.1f}',
                                ha='center', va='bottom',
                                fontsize=0.75 * plt.rcParams['font.size'] if PAPER_STYLE else 8)

        save_figure(fig4, out_dir, 'pipeline_latency', legacy_rect=[0, 0, 1, 0.95])
        if PAPER_STYLE:
            handles, labels = ax_lag_time.get_legend_handles_labels()
            save_legend(handles, labels, out_dir, 'pipeline_latency', FIG_WIDTH_SINGLE)
    else:
        print("WARNING: Pipeline latency columns missing – latency analysis skipped.")

    # ── Print summary stats ───────────────────────────────────────────────
    def rms(v):
        return np.sqrt(np.nanmean(v ** 2))

    if has_truth:
        print('\n=== Error summary (estimate - truth) ===')
        print(f'  truth: {describe_truth(df)}')
        if truth_source == 'rtk':
            sig = df['truth_sigma_xy'].to_numpy(float)
            print(f'  reported truth 1σ (horizontal) median {np.nanmedian(sig):.2f} m: '
                  f'errors below this are not resolved; heights assume a flat field')
        print('  long = along car heading (+ ahead),  lat = across car heading (+ left)')

        for s in streams:
            if not s['valid'].any():
                continue
            name = s['label']
            for field, unit in [('long', 'm'), ('lat', 'm'), ('z', 'm')]:
                if field in s:
                    vals = s[field][s['valid'] & np.isfinite(s[field])]
                    if len(vals):
                        print(f'  {name:<18s} {field:<4s} mean: {vals.mean():+.3f} {unit}   RMS: {rms(vals):.3f} {unit}')
            if 'yaw' in s:
                yaw_ok = s['valid'] & np.isfinite(s['yaw'])
                vals = s['yaw'][yaw_ok]
                if len(vals):
                    n_flip = int(np.sum(np.abs(vals) > np.pi / 2))
                    print(f'  {name:<18s} yaw  mean: {np.degrees(vals.mean()):+.2f}°   '
                          f'RMS: {np.degrees(rms(vals)):.2f}°   n: {len(vals)}   '
                          f'front/back flips: {n_flip}')
                    if 'yaw_sigma' in s:
                        sig = s['yaw_sigma'][yaw_ok]
                        good = np.isfinite(sig) & (sig > 0)
                        if good.any():
                            print(f'  {name:<18s} yaw  mean (err/σ)²: '
                                  f'{np.mean((vals[good] / sig[good]) ** 2):.2f}   (≈1 if σ is honest)')
            vals = s['abs'][s['valid']]
            print(f'  {name:<18s} |e|  mean: {vals.mean():.3f} m   max: {vals.max():.3f} m')

        print(f'  {"Filter":<18s} long mean: {np.nanmean(filt_err_long):+.3f} m   RMS: {rms(filt_err_long):.3f} m')
        print(f'  {"Filter":<18s} lat  mean: {np.nanmean(filt_err_lat):+.3f} m   RMS: {rms(filt_err_lat):.3f} m')
        print(f'  {"Filter":<18s} z    mean: {np.nanmean(filt_err_z):+.3f} m   RMS: {rms(filt_err_z):.3f} m')
        print(f'  {"Filter":<18s} yaw  mean: {np.degrees(np.nanmean(filt_yaw_err)):+.2f}°   '
              f'RMS: {np.degrees(rms(filt_yaw_err)):.2f}°')
        print(f'  {"Filter":<18s} |e|  mean: {np.nanmean(filt_err_abs):.3f} m   max: {np.nanmax(filt_err_abs):.3f} m')

        # Like-for-like pairs
        print('\n=== Like-for-like comparisons ===')
        print('  Live (control value):    filter @ est_time  vs  raw held to row time')
        for s in streams:
            if s['held'].any():
                for field in ('long', 'lat'):
                    v = s[f'held_{field}']
                    print(f'  {s["label"] + " held":<18s} {field:<4s} '
                          f'mean: {np.nanmean(v):+.3f} m   RMS: {rms(v):.3f} m')
        print('  Filtering quality:       filter @ image stamp  vs  raw @ image stamp')
        if post is not None:
            for field in ('long', 'lat'):
                v = post[field][post['ok']]
                print(f'  {"Filter @ stamp":<18s} {field:<4s} '
                      f'mean: {v.mean():+.3f} m   RMS: {rms(v):.3f} m')
        else:
            print('  (no filter-at-stamp data in this log)')

    if has_box_mode:
        yolo_new = new_measurement_mask(df_valid['landing_pad_yolo_car_raw_stamp'].to_numpy())
        n_box = int((yolo_new & box_mode).sum())
        print(f'\n=== YOLO measurement mode ===')
        print(f'  box update: {n_box}   fallback fit: {int(yolo_new.sum()) - n_box}')

    for s in lat_streams:
        win = f'state {LATENCY_FROM_STATE} -> end' if LATENCY_FROM_STATE is not None \
            else 'error window'
        print(f'\n=== {s["name"]} pipeline latency (ms, n = {len(s["total"])}, {win}) ===')
        for label, v in s['stages'] + [('Total', s['total'])]:
            v = v[np.isfinite(v)]
            if len(v):
                print(f'  {label:<26s} mean: {v.mean():6.1f}   std: {v.std():5.1f}   '
                      f'p95: {np.percentile(v, 95):6.1f}   max: {v.max():6.1f}')

    if has_per_state:
        print('\n=== UKF per-state 1σ summary ===')
        for col in sigma_cols:
            vals = df_valid[col].to_numpy()
            print(f'  {SIGMA_LABELS[col]:25s}  mean: {vals.mean():.4f}   '
                  f'max: {vals.max():.4f}   min: {vals.min():.4f}')
    elif has_trace_only:
        vals = df_valid[trace_col].to_numpy()
        print(f'\n=== {trace_col} (trace) summary ===')
        print(f'  mean: {vals.mean():.4f}   max: {vals.max():.4f}   min: {vals.min():.4f}')

    if has_nis:
        print('\n=== NIS summary (one per measurement) ===')
        if len(nis):
            print(f'  n: {len(nis)}   mean: {nis.mean():.3f}   max: {nis.max():.3f}   min: {nis.min():.3f}')
            print(f'  above 95% χ²(4) bound: {np.mean(nis > nis_bound) * 100:.1f}%')
        else:
            print('  no finite NIS entries')

    if has_nees and len(nees_finite):
        print('\n=== NEES summary ===')
        print(f'  mean: {nees_finite.mean():.3f}   max: {nees_finite.max():.3f}   '
              f'min: {nees_finite.min():.3f}')
        print(f'  above 95% χ²(3) bound: {np.mean(nees_finite > nees_bound) * 100:.1f}%')

    if show:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Landing pad estimation error analysis '
                                             '(SITL truth or RTK truth for real flights).')
    ap.add_argument('csv', nargs='?', default=None,
                    help='controller CSV, full path or a file name in LOG_DIR '
                         '(default: CSV_FILE in LOG_DIR)')
    ap.add_argument('--fleet', default='auto',
                    help="RTK telemetry CSV, 'auto' (default) or 'none'")
    ap.add_argument('--rtk-bias', default=None,
                    help="RTK frame offset 'E,N' in m to remove, or 'auto' (vision-referenced)")
    ap.add_argument('--out-dir', default=None, help='output folder (default: ./plots)')
    ap.add_argument('--no-show', action='store_true', help="save figures, don't open windows")
    a = ap.parse_args()
    bias = a.rtk_bias
    if bias is not None and bias.lower() != 'auto':
        bias = tuple(float(v) for v in bias.split(','))
    try:
        main(csv_path=a.csv, out_dir=a.out_dir, show=not a.no_show,
             fleet=None if str(a.fleet).lower() == 'none' else a.fleet, rtk_bias=bias)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)