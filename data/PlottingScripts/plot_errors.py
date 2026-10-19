"""
plot_errors.py  –  Landing pad position error analysis
======================================================
Place this script in the same folder as your CSV file and run:

    python plot_errors.py                      # newest controller_*.csv here
    python plot_errors.py path/to/controller_data.csv

Outputs (saved to ./plots next to the script):
    error_vs_time.png        –  longitudinal / lateral error (car frame), height (Z)
                                error, absolute horizontal error (with pipeline
                                latency on a secondary axis), yaw error and per-state
                                UKF 1-sigma. Controller-state transitions are marked
                                on every panel; spans where the UKF is using the YOLO
                                pixel-space box update are shaded.
    trajectory_2d.png        –  2-D XY path: true / filter / raw measurements
    consistency_analysis.png –  NIS / NEES vs time and distributions
    pipeline_latency.png     –  latency per pipeline stage

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


# ── Config ─────────────────────────────────────────────────────────────────

# Body-frame offset applied to the true XY position
TRUE_OFFSET_X_BODY = -0.5     # m, +X body direction
TRUE_OFFSET_Y_BODY = 0.0   # m, +Y body direction
TRUE_OFFSET_Z_BODY = 1.50    # m, +Z body direction

OUTLIER_SPEED_THRESH = 5.0   # m/s

# Set to None to disable either limit (seconds since start of log). With the auto
# start/end below on, these act as outer bounds.
T_START = 10
T_END   = 70

# Start the analysis at the first entry into pad lock (3xxx), once the drone is over
# the car and tracking it. Before that the filter is seeding or the drone is still
# closing in. T_START still applies if later.
AUTO_START_AT_LOCK = True

# End the analysis at touchdown: the first landing-confirmed state (6xxx) or, if the
# landing never completed, the start of the last abort (5xxx). Data after either is
# the drone on the pad or climbing away, not tracking. T_END still applies if earlier.
AUTO_END_AT_LANDING = True

DPI = 500

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

LAG_COLOURS = {
    # AprilTag pipeline
    'cam_to_image_lag': '#e63946',
    'image_to_transform_lag': '#f4a261',
    'transform_to_UKF_lag': '#2a9d8f',
    'apriltag_total_lag': '#6a4c93',

    # YOLO pipeline
    'yolo_cam_to_image_lag': '#d95f02',
    'yolo_image_to_transform_lag': '#7570b3',
    'yolo_transform_to_UKF_lag': '#1b9e77',
    'yolo_total_lag': '#e7298a',

    # Measurement age (the CSV's historical `total_lag` field)
    'total_lag': '#555555',
}

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

# ── Helpers ────────────────────────────────────────────────────────────────
def clean_true_position(x, y, t, thresh=OUTLIER_SPEED_THRESH, passes=3):
    """
    Remove single-frame position glitches from the true trajectory.

    A row is flagged as an outlier when the implied speed ENTERING that row
    AND the implied speed LEAVING that row both exceed `thresh`.  Flagged
    points are replaced by linear interpolation.  Multiple passes handle
    back-to-back glitches.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    bad_global = np.zeros(len(x), dtype=bool)

    for _ in range(passes):
        speed = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
        bad = np.zeros(len(x), dtype=bool)
        bad[1:-1] = (speed[:-1] > thresh) & (speed[1:] > thresh)
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
    yaw = np.unwrap(yaw.copy().astype(float))
    bad_global = np.zeros(len(yaw), dtype=bool)

    for _ in range(passes):
        yaw_rate = np.abs(np.diff(yaw) / np.diff(t))
        bad = np.zeros(len(yaw), dtype=bool)
        bad[1:-1] = (yaw_rate[:-1] > thresh) & (yaw_rate[1:] > thresh)
        bad_global |= bad

        idx = np.arange(len(yaw))
        good = ~bad_global
        yaw = np.interp(idx, idx[good], yaw[good])

    return wrap_angle(yaw), bad_global


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


def add_state_transitions(ax, df, time_col='timestamp', label=True):
    """
    Mark controller-state changes with dotted vertical lines.

    If `label` is True the state name/code is also written next to each line
    (only do this on one panel of a stacked figure to avoid clutter).
    """
    state_col = 'controller_state'
    if state_col not in df.columns:
        return

    states = pd.to_numeric(df[state_col], errors='coerce')
    changed = states.notna() & states.ne(states.shift())

    for row_idx in np.flatnonzero(changed.to_numpy()):
        state_code = int(states.iloc[row_idx])
        state_name = STATE_LABELS.get(state_code // 1000, 'Controller state')
        transition_t = float(df[time_col].iloc[row_idx])

        ax.axvline(transition_t, color='#555555', linewidth=0.9,
                   linestyle=':', alpha=0.65, zorder=0)
        if label:
            ax.annotate(
                f'{state_name} ({state_code})',
                xy=(transition_t, 0.99),
                xycoords=('data', 'axes fraction'),
                xytext=(3, -2),
                textcoords='offset points',
                rotation=90, ha='left', va='top',
                fontsize=7, color='#333333',
                annotation_clip=False,
            )


# ── Main ───────────────────────────────────────────────────────────────────

def main(csv_path=None, out_dir=None, window=None, show=True):
    """ Analyse one controller CSV. All arguments are optional, so running the script
        directly behaves exactly as before; batch_analysis.py calls it with them set.

    :param csv_path: CSV to read (default: first command-line argument, else the
                     hard-coded file below)
    :param out_dir:  Folder for the figures (default: ./plots next to this script)
    :param window:   (t_start, t_end) in seconds since the start of the log. Overrides
                     T_START / T_END and the auto start/end.
    :param show:     Open the figures in a window (False: save and close)
    """
    script_dir = Path(__file__).parent
    if csv_path is None:
        csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 \
            else script_dir / 'controller_20261017_222005.csv'
    csv_path = Path(csv_path)

    out_dir = Path(out_dir) if out_dir is not None else script_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {csv_path.name}")
    df = pd.read_csv(csv_path)

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

        if t_cut is None:
            print("Auto end: no 6000/5000 state in log, using T_END.")
        elif t_start is not None and t_cut <= t_start:
            print(f"Auto end: {why} at t = {t_cut:.2f} s is before the start, ignored.")
        elif t_end is None or t_cut < t_end:
            t_end = t_cut
            print(f"Auto end: analysis ends at t = {t_cut:.2f} s, {why}.")

    # Explicit window (batch runs) replaces the limits above
    if window is not None:
        t_start, t_end = window
        print(f"Window: t = {t_start:.2f} s to {t_end:.2f} s (set by caller).")

    if t_start is not None:
        df = df[df['timestamp'] >= t_start]
    if t_end is not None:
        df = df[df['timestamp'] <= t_end]
    df = df.reset_index(drop=True)

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
            s['yaw'] = wrap_angle(df_valid[f'{prefix}_glob_yaw_raw'].to_numpy() - tyaw)
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
                           marker='x', s=22, linewidths=1.0, alpha=ALPHA,
                           label=s['label'] + suffix)

    def plot_held(ax, field):
        """ Step line of the held (live) raw error for every stream that has it. """
        for s in streams:
            v = s.get(f'held_{field}')
            if v is not None and np.isfinite(v).any():
                ax.plot(ctrl_t, v, color=s['colour'], lw=0.8, ls='--',
                        drawstyle='steps-post', alpha=0.7,
                        label=s['label'] + ' held (live)')

    def plot_post(ax, field):
        """ Scatter the filter error at each YOLO image stamp. """
        if post is not None and field in post:
            ok = post['ok'] & np.isfinite(post[field])
            if ok.any():
                ax.scatter(post['t'][ok], post[field][ok], color=FILT_C,
                           marker='o', s=10, alpha=ALPHA, label='Filter @ image stamp')

    # YOLO pixel-space box mode (new logs only)
    has_box_mode = 'yolo_box_mode' in df_valid.columns
    box_mode = (df_valid['yolo_box_mode'].to_numpy() == 1) if has_box_mode else None

    # ── Detect which covariance / latency columns are present ─────────────
    apriltag_lag_cols = ['cam_to_image_lag', 'image_to_transform_lag', 'transform_to_UKF_lag']
    yolo_lag_cols = ['yolo_cam_to_image_lag', 'yolo_image_to_transform_lag', 'yolo_transform_to_UKF_lag']

    # A stream that was switched off (e.g. AprilTag toggled off) logs all-NaN latency
    def has_cols(cols):
        return all(c in df_valid.columns for c in cols) and df_valid[cols].notna().any().any()

    has_apriltag_lag = has_cols(apriltag_lag_cols)
    has_yolo_lag = has_cols(yolo_lag_cols)
    has_meas_age = has_cols(['total_lag'])

    # The CSV's `total_lag` field is measurement age, not the sum of the stages.
    if has_apriltag_lag:
        df_valid['apriltag_total_lag'] = df_valid[apriltag_lag_cols].sum(axis=1, min_count=1)
    if has_yolo_lag:
        df_valid['yolo_total_lag'] = df_valid[yolo_lag_cols].sum(axis=1, min_count=1)

    sigma_cols     = [c for c in SIGMA_COLOURS if c in df_valid.columns]
    has_per_state  = len(sigma_cols) > 0
    trace_col      = 'covar_trace' if 'covar_trace' in df_valid.columns else 'filter_covar'
    has_trace_only = (not has_per_state) and (trace_col in df_valid.columns)
    has_nis  = 'nis' in df_valid.columns
    has_nees = 'nees' in df_valid.columns

    if not has_apriltag_lag and not has_yolo_lag:
        print("WARNING: no AprilTag or YOLO latency columns found – latency overlays skipped.")
    if not has_meas_age:
        print("WARNING: 'total_lag' column not found – measurement-age diagnostics skipped.")
    if not has_per_state and not has_trace_only:
        print("WARNING: no covariance columns found – covar subplot skipped.")
    if 'controller_state' not in df.columns:
        print("WARNING: 'controller_state' column not found – state transitions skipped.")

    # ── Figure 1 – Error vs Time ─────────────────────────────────────────
    # Panels: longitudinal, lateral, height, absolute horizontal, yaw, [covariance]
    has_covar_plot = has_per_state or has_trace_only
    n_rows = 6 if has_covar_plot else 5

    fig1, axes = plt.subplots(n_rows, 1, figsize=(12, 3.2 * n_rows), sharex=True)
    axes = list(axes)
    fig1.suptitle(
        'Landing Pad Position Error vs Time\n(true position outliers removed)',
        fontsize=14, fontweight='bold', y=0.99,
    )
    ax_lon, ax_lat, ax_z, ax_abs, ax_yaw = axes[:5]
    ax_cov = axes[5] if has_covar_plot else None

    # ── Panels: longitudinal, lateral and height error ────────────────────
    for ax, fe, field, ylabel, title in [
        (ax_lon, filt_err_long, 'long', 'Longitudinal error (m)',
         'Longitudinal error  (along car heading, + = estimate ahead of truth)'),
        (ax_lat, filt_err_lat, 'lat', 'Lateral error (m)',
         'Lateral error  (across car heading, + = estimate left of truth)'),
        (ax_z, filt_err_z, 'z', 'Z error (m)',
         'Height (Z) error  (+ = estimate above truth)'),
    ]:
        ax.plot(est_t, fe, color=FILT_C, lw=1.0, alpha=ALPHA, label='Filter estimate')
        plot_raw(ax, field)
        if field in ('long', 'lat'):
            plot_held(ax, field)
            plot_post(ax, field)
        ax.axhline(0, color='0.5', lw=0.6, ls='--')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    # ── Panel: Absolute horizontal error + latency secondary axis ───────
    ax_abs.plot(est_t, filt_err_abs, color=FILT_C, lw=1.0, alpha=ALPHA, label='Filter |error|')
    plot_raw(ax_abs, 'abs', suffix=' |error|')
    plot_held(ax_abs, 'abs')
    plot_post(ax_abs, 'abs')
    ax_abs.set_ylim(bottom=0)
    ax_abs.set_ylabel('|error| (m)', fontsize=10)
    ax_abs.set_title('Absolute horizontal error (Euclidean norm, XY)', fontsize=10)
    ax_abs.grid(True, alpha=0.3)

    lag_series = [
        (has_apriltag_lag, 'apriltag_total_lag', 'AprilTag total pipeline', '--', 1.1, 0.75),
        (has_yolo_lag,     'yolo_total_lag',     'YOLO total pipeline',     '-.', 1.1, 0.75),
        (has_meas_age,     'total_lag',          'Measurement age',         ':',  0.9, 0.55),
    ]
    if any(on for on, *_ in lag_series):
        ax_lag = ax_abs.twinx()
        for on, col, label, ls, lw, a in lag_series:
            if on:
                ax_lag.plot(ctrl_t, df_valid[col].to_numpy(), color=LAG_COLOURS[col],
                            lw=lw, alpha=a, ls=ls, label=label)
        ax_lag.set_ylabel('Latency / age (ms)', fontsize=10)
        ax_lag.tick_params(axis='y', labelcolor=LAG_COLOURS['total_lag'])
        ax_lag.set_ylim(bottom=0)
        lines_l, labels_l = ax_abs.get_legend_handles_labels()
        lines_r, labels_r = ax_lag.get_legend_handles_labels()
        ax_abs.legend(lines_l + lines_r, labels_l + labels_r, fontsize=8, loc='upper right')
    else:
        ax_abs.legend(fontsize=8, loc='upper right')

    # ── Panel: Yaw error ───────────────────────────────────────────────
    ax_yaw.plot(est_t, np.degrees(filt_yaw_err), color=FILT_C, lw=1.0, alpha=ALPHA,
                label='Filter estimate')
    plot_raw(ax_yaw, 'yaw', scale=np.degrees(1.0))
    ax_yaw.axhline(0, color='0.5', ls='--', lw=0.8)
    ax_yaw.set_ylabel('Yaw error (deg)', fontsize=10)
    ax_yaw.set_title('Yaw error', fontsize=10)
    ax_yaw.grid(True, alpha=0.3)

    # ── Panel: per-state 1-sigma covariance ──────────────────────────────
    if has_covar_plot:
        if has_per_state:
            # Secondary axis for angular states (rad vs metres on very different scales)
            pos_plotted = [c for c in ['sigma_px', 'sigma_py', 'sigma_pz', 'sigma_v', 'sigma_a']
                           if c in sigma_cols]
            ang_plotted = [c for c in ['sigma_yaw', 'sigma_yaw_rate'] if c in sigma_cols]

            for col in pos_plotted:
                ax_cov.plot(ctrl_t, df_valid[col].to_numpy(), color=SIGMA_COLOURS[col],
                            lw=1.1, alpha=ALPHA, label=SIGMA_LABELS[col])

            if ang_plotted:
                ax_ang = ax_cov.twinx()
                for col in ang_plotted:
                    ax_ang.plot(ctrl_t, df_valid[col].to_numpy(), color=SIGMA_COLOURS[col],
                                lw=1.1, alpha=ALPHA, ls='--', label=SIGMA_LABELS[col])
                ax_ang.set_ylabel('1σ angular (rad / rad·s⁻¹)', fontsize=9)
                ax_ang.set_ylim(bottom=0)
                lines_l, labels_l = ax_cov.get_legend_handles_labels()
                lines_r, labels_r = ax_ang.get_legend_handles_labels()
                ax_cov.legend(lines_l + lines_r, labels_l + labels_r,
                              fontsize=8, loc='upper right', ncol=2)
            else:
                ax_cov.legend(fontsize=8, loc='upper right', ncol=2)

            ax_cov.set_ylabel('1σ position / velocity (m, m/s)', fontsize=9)
            ax_cov.set_title('UKF per-state 1σ covariance  (— linear,  - - angular)', fontsize=10)
        else:
            # Fallback: single trace scalar from old CSVs
            ax_cov.plot(ctrl_t, df_valid[trace_col].to_numpy(), color='#c0392b',
                        lw=1.0, alpha=ALPHA, label='trace(P)')
            ax_cov.set_ylabel('trace(P)', fontsize=10)
            ax_cov.set_title('Filter covariance — trace(P)', fontsize=10)
            ax_cov.legend(fontsize=9, loc='upper right')

        ax_cov.set_ylim(bottom=0)
        ax_cov.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)', fontsize=10)

    # ── State transitions and YOLO box-mode shading on every panel ───────
    # Lines on all panels; text labels only on the top one to avoid clutter.
    for k, ax in enumerate(axes):
        add_state_transitions(ax, df, time_col='timestamp', label=(k == 0))
        if has_box_mode and box_mode.any():
            ax.fill_between(ctrl_t, 0, 1, where=box_mode, step='post',
                            transform=ax.get_xaxis_transform(),
                            color=BOX_MODE_C, alpha=0.08, lw=0, zorder=0)

    box_patch = [Patch(color=BOX_MODE_C, alpha=0.2, label='YOLO box update')] \
        if has_box_mode and box_mode.any() else []
    for ax in (ax_lon, ax_lat, ax_z, ax_yaw):
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + box_patch, labels + [p.get_label() for p in box_patch],
                  fontsize=8, loc='upper right')

    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    out1 = out_dir / 'error_vs_time.png'
    fig1.savefig(out1, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {out1}")

    # ── Figure 2 – 2-D Trajectory ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 8))
    fig2.suptitle('Landing Pad 2D Trajectory\n(true position outliers removed)',
                  fontsize=14, fontweight='bold')

    ax2.plot(true_x, true_y, color=TRUE_C, lw=1.8, label='True position', zorder=3)
    ax2.plot(df_valid['landing_pad_rel_x'], df_valid['landing_pad_rel_y'],
             color=FILT_C, lw=1.2, alpha=0.85, label='Filter estimate', zorder=4)
    for s in streams:
        if s['valid'].any():
            ax2.scatter(s['x'][s['valid']], s['y'][s['valid']], c=s['colour'],
                        marker='x', s=22, alpha=0.65, label=s['label'], zorder=2)

    ax2.plot(true_x[0],  true_y[0],  'go', ms=8, label='Start', zorder=5)
    ax2.plot(true_x[-1], true_y[-1], 'rs', ms=8, label='End',   zorder=5)
    ax2.set_xlabel('X position (m)', fontsize=11)
    ax2.set_ylabel('Y position (m)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    out2 = out_dir / 'trajectory_2d.png'
    fig2.savefig(out2, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {out2}")

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
        fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
        fig3.suptitle('UKF Consistency Analysis', fontsize=14, fontweight='bold')
        ax_nis_time, ax_nees_time = axes3[0]
        ax_nis_hist, ax_nees_hist = axes3[1]

        for on, ax_t, ax_h, name, t_v, vals, dof, bound, colour in [
            (has_nis,  ax_nis_time,  ax_nis_hist,  'NIS',
             nis_t if has_nis else None, nis if has_nis else None, 4, nis_bound, '#8e44ad'),
            (has_nees, ax_nees_time, ax_nees_hist, 'NEES',
             ctrl_t, nees_finite if has_nees else None, 3, nees_bound, '#2980b9'),
        ]:
            if not on:
                continue
            series = nis if name == 'NIS' else nees
            ax_t.plot(t_v, series, color=colour, lw=1.2, label=name,
                      marker='.' if name == 'NIS' else None, ms=3)
            ax_t.axhline(bound, color='red', linestyle='--', label=f'95% χ²({dof})')
            ax_t.axhline(dof, color='green', linestyle=':', label='Expected mean')
            ax_t.set_title(f'{name} vs Time')
            ax_t.set_ylabel(name)
            ax_t.grid(True, alpha=0.3)
            ax_t.legend()

            if len(vals) > 0:
                ax_h.hist(vals, bins=100, density=True, alpha=0.6, label=f'Measured {name}')
                x = np.linspace(0, max(4 * dof, vals.max()), 400)
                ax_h.plot(x, chi2.pdf(x, df=dof), lw=2, label=f'χ²({dof})')
                ax_h.axvline(vals.mean(), linestyle='--', label=f'Mean={vals.mean():.2f}')
            ax_h.set_title(f'{name} Distribution')
            ax_h.set_xlabel(name)
            ax_h.set_ylabel('Density')
            ax_h.grid(True, alpha=0.3)
            ax_h.legend()

        fig3.tight_layout(rect=[0, 0, 1, 0.95])
        out3 = out_dir / 'consistency_analysis.png'
        fig3.savefig(out3, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {out3}")

    # ── Figure 4 – Pipeline latency analysis ───────────────────────────────
    if has_apriltag_lag or has_yolo_lag or has_meas_age:
        fig4, (ax_lag_time, ax_lag_bar) = plt.subplots(1, 2, figsize=(15, 5))
        fig4.suptitle('Pipeline Latency Analysis', fontsize=14, fontweight='bold')

        bar_specs = []   # (label, column)
        for on, prefix, cols, total, ls in [
            (has_apriltag_lag, 'AprilTag', apriltag_lag_cols, 'apriltag_total_lag', '--'),
            (has_yolo_lag,     'YOLO',     yolo_lag_cols,     'yolo_total_lag',     '-.'),
        ]:
            if not on:
                continue
            for col, stage in zip(cols, ['Camera → Image', 'Image → Transform', 'Transform → UKF']):
                ax_lag_time.plot(ctrl_t, df_valid[col].to_numpy(), lw=1.1, alpha=0.80,
                                 color=LAG_COLOURS[col], label=f'{prefix} {stage}')
                bar_specs.append((f'{prefix}\n{stage}', col))
            ax_lag_time.plot(ctrl_t, df_valid[total].to_numpy(), lw=1.4, alpha=0.90,
                             color=LAG_COLOURS[total], ls=ls, label=f'{prefix} total')
            bar_specs.append((f'{prefix}\nTotal', total))

        if has_meas_age:
            ax_lag_time.plot(ctrl_t, df_valid['total_lag'].to_numpy(), lw=0.9, alpha=0.55,
                             color=LAG_COLOURS['total_lag'], ls=':', label='Measurement age')
            bar_specs.append(('Measurement\nAge', 'total_lag'))

        ax_lag_time.set_title('Latency vs Time')
        ax_lag_time.set_xlabel('Time (s)')
        ax_lag_time.set_ylabel('Latency / age (ms)')
        ax_lag_time.grid(True, alpha=0.3)
        ax_lag_time.legend(fontsize=7, ncol=2)

        mean_lags = [df_valid[col].mean() for _, col in bar_specs]
        bars = ax_lag_bar.bar([lbl for lbl, _ in bar_specs], mean_lags,
                              color=[LAG_COLOURS[col] for _, col in bar_specs], alpha=0.85)
        ax_lag_bar.set_title('Mean Latency Breakdown')
        ax_lag_bar.set_ylabel('Mean latency / age (ms)')
        ax_lag_bar.tick_params(axis='x', labelsize=8)
        ax_lag_bar.grid(True, axis='y', alpha=0.3)

        max_mean = np.nanmax(mean_lags) if mean_lags else 1.0
        for bar, value in zip(bars, mean_lags):
            if np.isfinite(value):
                ax_lag_bar.text(bar.get_x() + bar.get_width() / 2, value + 0.02 * max_mean,
                                f'{value:.1f} ms', ha='center', va='bottom', fontsize=8)

        fig4.tight_layout(rect=[0, 0, 1, 0.95])
        out4 = out_dir / 'pipeline_latency.png'
        fig4.savefig(out4, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {out4}")
    else:
        print("WARNING: Pipeline latency columns missing – latency analysis skipped.")

    # ── Print summary stats ───────────────────────────────────────────────
    def rms(v):
        return np.sqrt(np.nanmean(v ** 2))

    print('\n=== Error summary (estimate - truth) ===')
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
            vals = s['yaw'][s['valid']]
            print(f'  {name:<18s} yaw  mean: {np.degrees(vals.mean()):+.2f}°   '
                  f'RMS: {np.degrees(rms(vals)):.2f}°')
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

    for on, col, title in [
        (has_apriltag_lag, 'apriltag_total_lag', 'AprilTag total pipeline latency'),
        (has_yolo_lag,     'yolo_total_lag',     'YOLO total pipeline latency'),
        (has_meas_age,     'total_lag',          'measurement age'),
    ]:
        if on:
            vals = df_valid[col].to_numpy(dtype=float)
            print(f'\n=== {title} summary ===')
            print(f'  mean: {np.nanmean(vals):.1f} ms   max: {np.nanmax(vals):.1f} ms   '
                  f'min: {np.nanmin(vals):.1f} ms')

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
    main()