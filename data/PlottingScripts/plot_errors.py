"""
plot_errors.py  –  Landing pad position error analysis
======================================================
Place this script in the same folder as your CSV file and run:

    python plot_errors.py

Or point it at a specific CSV:

    python plot_errors.py path/to/controller_data.csv

Outputs (saved next to the script):
    error_vs_time.png   –  X error, Y error, absolute error vs time (with
                           total_lag on a secondary axis), and per-state
                           UKF 1-sigma covariance
    trajectory_2d.png   –  2-D XY path: true / filter / raw measurements

Dependencies:
    pip install pandas numpy matplotlib scipy
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import chi2


# ── Config ─────────────────────────────────────────────────────────────────

# Body-frame offset applied to the true XY position
TRUE_OFFSET_X_BODY = 0.0   # m, +X body direction
TRUE_OFFSET_Y_BODY = -0.40   # m, +Y body direction

OUTLIER_SPEED_THRESH = 10.0   # m/s

RAW_C        = "#e03939"   # orange  – AprilTag raw measurement
YOLO_LP_C    = '#d95f02'   # orange  – YOLO landing-pad raw measurement
YOLO_CAR_C   = '#7570b3'   # purple  – YOLO car raw measurement
FILT_C       = '#3a86d4'   # blue    – filter estimate
TRUE_C       = '#2ca02c'   # green   – true position
LAG_C        = '#9b59b6'   # purple  – total_lag overlay
ALPHA        = 0.85

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
        dx = np.diff(x);  dy = np.diff(y);  dt = np.diff(t)
        speed = np.sqrt((dx / dt) ** 2 + (dy / dt) ** 2)
        bad = np.zeros(len(x), dtype=bool)
        for i in range(1, len(x) - 1):
            if speed[i - 1] > thresh and speed[i] > thresh:
                bad[i] = True
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

        dyaw = np.diff(yaw)
        dt = np.diff(t)

        yaw_rate = np.abs(dyaw / dt)

        bad = np.zeros(len(yaw), dtype=bool)

        for i in range(1, len(yaw)-1):
            if yaw_rate[i-1] > thresh and yaw_rate[i] > thresh:
                bad[i] = True

        bad_global |= bad

        idx = np.arange(len(yaw))
        good = ~bad_global

        yaw = np.interp(
            idx,
            idx[good],
            yaw[good]
        )

    # Wrap back to [-pi, pi]
    yaw = (yaw + np.pi) % (2*np.pi) - np.pi

    return yaw, bad_global


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'controller_20260928_214030.csv'
    out_dir = script_dir / 'plots'
    out_dir.mkdir(exist_ok=True)

    # Set to None to disable either limit
    T_START = 20  # seconds
    T_END   = 50  # seconds

    print(f"Reading: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # Make timestamp relative to the start of the log so T_START/T_END (and
    # every plot axis below) are in "seconds since start", not raw clock
    # time. Without this, timestamp is an absolute value and the T_START/
    # T_END window below would filter out all (or none) of the rows.
    df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]

    # Apply time window
    if T_START is not None:
        df = df[df['timestamp'] >= T_START]
    if T_END is not None:
        df = df[df['timestamp'] <= T_END]

    df = df.reset_index(drop=True)

    # ── Clean outlier spikes in the ground-truth trajectory ──────────────
    # The CSV logs the relative (quad-frame) ground truth directly as
    # landing_pad_rel_true_x/y, so there's no need to manually subtract
    # quad_true from landing_pad_true — just read it straight.
    t_arr = df['timestamp'].values
    true_x, true_y, outlier_mask = clean_true_position(
        df['landing_pad_rel_true_x'].values,
        df['landing_pad_rel_true_y'].values,
        t_arr,
    )

    true_yaw, yaw_outlier_mask = clean_true_yaw(
        df['landing_pad_rel_true_yaw'].values,
        t_arr,
    )

    # ── Apply configurable body-frame XY offset ────────────────────────
    cos_yaw = np.cos(true_yaw)
    sin_yaw = np.sin(true_yaw)

    offset_x = (
        TRUE_OFFSET_X_BODY * cos_yaw
        - TRUE_OFFSET_Y_BODY * sin_yaw
    )
    offset_y = (
        TRUE_OFFSET_X_BODY * sin_yaw
        + TRUE_OFFSET_Y_BODY * cos_yaw
    )

    true_x += offset_x
    true_y += offset_y

    df['true_x_clean'] = true_x
    df['true_y_clean'] = true_y
    df['true_yaw_clean'] = true_yaw

    n_outliers = outlier_mask.sum()
    if n_outliers:
        print(f"Removed {n_outliers} outlier row(s) from true position "
              f"(interpolated over glitches at t = "
              f"{np.round(t_arr[outlier_mask], 2).tolist()})")

    df['nees'].values[outlier_mask] = np.nan

    # ── Restrict to rows where a measurement exists ───────────────────────
    measurement_mask = (
        (df['landing_pad_april_raw_stamp'] > 0) |
        (df['landing_pad_yolo_LP_raw_stamp'] > 0) |
        (df['landing_pad_yolo_car_raw_stamp'] > 0)
    )

    df_valid = df[measurement_mask].copy().reset_index(drop=True)

    # ── Raw measurement masks ────────────────────────────────────────────
    # The CSV is logged at the controller rate, so a raw measurement can be
    # repeated across several rows while waiting for the next sensor result.
    # Only plot/count a raw measurement when its measurement stamp changes.
    def new_measurement_mask(stamp):
        stamp = np.asarray(stamp)
        is_new = np.ones(len(stamp), dtype=bool)
        if len(stamp) > 1:
            is_new[1:] = stamp[1:] != stamp[:-1]
        return (stamp > 0) & is_new

    april_stamp = df_valid['landing_pad_april_raw_stamp'].values
    yolo_lp_stamp = df_valid['landing_pad_yolo_LP_raw_stamp'].values
    yolo_car_stamp = df_valid['landing_pad_yolo_car_raw_stamp'].values

    april_valid = (
        new_measurement_mask(april_stamp) &
        np.isfinite(df_valid['landing_pad_april_rel_x_raw'].values) &
        np.isfinite(df_valid['landing_pad_april_rel_y_raw'].values)
    )

    yolo_lp_valid = (
        new_measurement_mask(yolo_lp_stamp) &
        np.isfinite(df_valid['landing_pad_yolo_LP_rel_x_raw'].values) &
        np.isfinite(df_valid['landing_pad_yolo_LP_rel_y_raw'].values)
    )

    yolo_car_valid = (
        new_measurement_mask(yolo_car_stamp) &
        np.isfinite(df_valid['landing_pad_yolo_car_rel_x_raw'].values) &
        np.isfinite(df_valid['landing_pad_yolo_car_rel_y_raw'].values)
    )

    # ── True position at each row's controller timestamp ──────────────────
    interp_x = interp1d(t_arr, true_x, kind='linear', fill_value='extrapolate')
    interp_y = interp1d(t_arr, true_y, kind='linear', fill_value='extrapolate')

    ctrl_t = df_valid['timestamp'].values
    true_at_ctrl_x = interp_x(ctrl_t)
    true_at_ctrl_y = interp_y(ctrl_t)

    # ── Compute errors ────────────────────────────────────────────────────
    april_err_x = (
        df_valid['landing_pad_april_rel_x_raw'].values - true_at_ctrl_x
    )
    april_err_y = (
        df_valid['landing_pad_april_rel_y_raw'].values - true_at_ctrl_y
    )
    april_err_abs = np.sqrt(april_err_x ** 2 + april_err_y ** 2)

    yolo_lp_err_x = (
        df_valid['landing_pad_yolo_LP_rel_x_raw'].values - true_at_ctrl_x
    )
    yolo_lp_err_y = (
        df_valid['landing_pad_yolo_LP_rel_y_raw'].values - true_at_ctrl_y
    )
    yolo_lp_err_abs = np.sqrt(yolo_lp_err_x ** 2 + yolo_lp_err_y ** 2)

    yolo_car_err_x = (
        df_valid['landing_pad_yolo_car_rel_x_raw'].values - true_at_ctrl_x
    )
    yolo_car_err_y = (
        df_valid['landing_pad_yolo_car_rel_y_raw'].values - true_at_ctrl_y
    )
    yolo_car_err_abs = np.sqrt(yolo_car_err_x ** 2 + yolo_car_err_y ** 2)

    filt_err_x   = df_valid['landing_pad_rel_x'].values - df_valid['true_x_clean'].values
    filt_err_y   = df_valid['landing_pad_rel_y'].values - df_valid['true_y_clean'].values
    filt_err_abs = np.sqrt(filt_err_x ** 2 + filt_err_y ** 2)

    def wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    true_yaw_interp = interp1d(
        t_arr,
        true_yaw,
        kind='linear',
        fill_value='extrapolate'
    )

    true_yaw_at_ctrl = true_yaw_interp(ctrl_t)

    raw_yaw_err = wrap_angle(
        df_valid["landing_pad_april_rel_yaw_raw"].values
        - true_yaw_at_ctrl
    )

    filt_yaw_err = wrap_angle(
        df_valid["landing_pad_rel_yaw"].values
        - true_yaw_at_ctrl
    )

    # ── Detect which covariance / latency columns are present ─────────────
    apriltag_lag_cols = [
        'cam_to_image_lag',
        'image_to_transform_lag',
        'transform_to_UKF_lag',
    ]
    yolo_lag_cols = [
        'yolo_cam_to_image_lag',
        'yolo_image_to_transform_lag',
        'yolo_transform_to_UKF_lag',
    ]

    has_apriltag_lag = all(c in df_valid.columns for c in apriltag_lag_cols)
    has_yolo_lag = all(c in df_valid.columns for c in yolo_lag_cols)
    has_meas_age = 'total_lag' in df_valid.columns

    # The CSV's `total_lag` field is actually measurement age
    # (_UKF_meas_age), not the sum of the processing stages.
    if has_apriltag_lag:
        df_valid['apriltag_total_lag'] = (
            df_valid[apriltag_lag_cols].sum(axis=1, min_count=1)
        )
    if has_yolo_lag:
        df_valid['yolo_total_lag'] = (
            df_valid[yolo_lag_cols].sum(axis=1, min_count=1)
        )

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

    # ── Figure 1 – Error vs Time (4 subplots) ────────────────────────────
    has_covar_plot = has_per_state or has_trace_only
    n_rows = 4

    if has_covar_plot:
        n_rows += 1

    fig1, axes = plt.subplots(
        n_rows, 1,
        figsize=(12, 3.2 * n_rows),
        sharex=True,
    )
    fig1.suptitle(
        'Landing Pad Position Error vs Time\n(true position outliers removed)',
        fontsize=14, fontweight='bold', y=0.99,
    )

    # ── Panels 0-1: X and Y error ─────────────────────────────────────────
    for ax, (re, fe, ylabel, title) in zip(
        axes[:2],
        [
            (filt_err_x, filt_err_x, 'X error (m)',  'X-axis error'),
            (filt_err_y, filt_err_y, 'Y error (m)',  'Y-axis error'),
        ],
    ):
        ax.plot(ctrl_t, fe, color=FILT_C, lw=1.0, alpha=ALPHA, label='Filter estimate')

        if april_valid.any():
            ax.scatter(
                ctrl_t[april_valid],
                (april_err_x if ax is axes[0] else april_err_y)[april_valid],
                color=RAW_C,
                marker='x',
                s=22,
                linewidths=1.0,
                alpha=ALPHA,
                label='AprilTag raw'
            )

        if yolo_lp_valid.any():
            ax.scatter(
                ctrl_t[yolo_lp_valid],
                (yolo_lp_err_x if ax is axes[0] else yolo_lp_err_y)[yolo_lp_valid],
                color=YOLO_LP_C,
                marker='x',
                s=22,
                linewidths=1.0,
                alpha=ALPHA,
                label='YOLO LP raw'
            )

        if yolo_car_valid.any():
            ax.scatter(
                ctrl_t[yolo_car_valid],
                (yolo_car_err_x if ax is axes[0] else yolo_car_err_y)[yolo_car_valid],
                color=YOLO_CAR_C,
                marker='x',
                s=22,
                linewidths=1.0,
                alpha=ALPHA,
                label='YOLO car raw'
            )

        ax.axhline(0, color='0.5', lw=0.6, ls='--')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    # ── Panel 2: Absolute error + latency secondary axis ───────────────
    ax_abs = axes[2]
    ax_abs.plot(
        ctrl_t,
        filt_err_abs,
        color=FILT_C,
        lw=1.0,
        alpha=ALPHA,
        label='Filter |error|'
    )

    if april_valid.any():
        ax_abs.scatter(
            ctrl_t[april_valid],
            april_err_abs[april_valid],
            color=RAW_C,
            marker='x',
            s=22,
            linewidths=1.0,
            alpha=ALPHA,
            label='AprilTag raw |error|'
        )

    if yolo_lp_valid.any():
        ax_abs.scatter(
            ctrl_t[yolo_lp_valid],
            yolo_lp_err_abs[yolo_lp_valid],
            color=YOLO_LP_C,
            marker='x',
            s=22,
            linewidths=1.0,
            alpha=ALPHA,
            label='YOLO LP raw |error|'
        )

    if yolo_car_valid.any():
        ax_abs.scatter(
            ctrl_t[yolo_car_valid],
            yolo_car_err_abs[yolo_car_valid],
            color=YOLO_CAR_C,
            marker='x',
            s=22,
            linewidths=1.0,
            alpha=ALPHA,
            label='YOLO car raw |error|'
        )

    ax_abs.axhline(0, color='0.5', lw=0.6, ls='--')
    ax_abs.set_ylim(bottom=0)
    ax_abs.set_ylabel('|error| (m)', fontsize=10)
    ax_abs.set_title('Absolute error (Euclidean norm)', fontsize=10)
    ax_abs.grid(True, alpha=0.3)

    if has_apriltag_lag or has_yolo_lag or has_meas_age:
        ax_lag = ax_abs.twinx()

        if has_apriltag_lag:
            ax_lag.plot(
                ctrl_t,
                df_valid['apriltag_total_lag'].values,
                color=LAG_COLOURS['apriltag_total_lag'],
                lw=1.1,
                alpha=0.75,
                ls='--',
                label='AprilTag total pipeline'
            )

        if has_yolo_lag:
            ax_lag.plot(
                ctrl_t,
                df_valid['yolo_total_lag'].values,
                color=LAG_COLOURS['yolo_total_lag'],
                lw=1.1,
                alpha=0.75,
                ls='-.',
                label='YOLO total pipeline'
            )

        if has_meas_age:
            ax_lag.plot(
                ctrl_t,
                df_valid['total_lag'].values,
                color=LAG_COLOURS['total_lag'],
                lw=0.9,
                alpha=0.55,
                ls=':',
                label='Measurement age'
            )

        ax_lag.set_ylabel('Latency / age (ms)', fontsize=10)
        ax_lag.tick_params(axis='y', labelcolor=LAG_COLOURS['total_lag'])
        ax_lag.set_ylim(bottom=0)

        lines_l, labels_l = ax_abs.get_legend_handles_labels()
        lines_r, labels_r = ax_lag.get_legend_handles_labels()
        ax_abs.legend(
            lines_l + lines_r,
            labels_l + labels_r,
            fontsize=8,
            loc='upper right'
        )
    else:
        ax_abs.legend(fontsize=8, loc='upper right')

    # ── Panel 3: Yaw error ─────────────────────────────────────────────
    ax_yaw = axes[3]
    ax_yaw.plot(
        ctrl_t,
        np.degrees(filt_yaw_err),
        color=FILT_C,
        lw=1.0,
        alpha=ALPHA,
        label='Filter estimate'
    )

    if april_valid.any():
        ax_yaw.scatter(
            ctrl_t[april_valid],
            np.degrees(raw_yaw_err[april_valid]),
            color=RAW_C,
            marker='x',
            s=22,
            linewidths=1.0,
            alpha=ALPHA,
            label='AprilTag raw'
        )

    ax_yaw.axhline(
        0,
        color='0.5',
        ls='--',
        lw=0.8
    )

    ax_yaw.set_ylabel('Yaw error (deg)', fontsize=10)
    ax_yaw.set_title('Yaw error', fontsize=10)
    ax_yaw.grid(True, alpha=0.3)
    ax_yaw.legend(fontsize=9, loc='upper right')

    # ── Panel 4: per-state 1-sigma covariance ────────────────────────────
    if has_covar_plot:
        ax_cov = axes[4]

        if has_per_state:
            # Plot each state's 1-sigma on the same axes, with a secondary
            # axis for angular states (rad vs metres on very different scales)
            pos_cols = ['sigma_px', 'sigma_py', 'sigma_pz',
                        'sigma_v',  'sigma_a']
            ang_cols = ['sigma_yaw', 'sigma_yaw_rate']

            pos_plotted = [c for c in pos_cols if c in sigma_cols]
            ang_plotted = [c for c in ang_cols if c in sigma_cols]

            for col in pos_plotted:
                ax_cov.plot(
                    ctrl_t, df_valid[col].values,
                    color=SIGMA_COLOURS[col], lw=1.1, alpha=ALPHA,
                    label=SIGMA_LABELS[col],
                )

            if ang_plotted:
                ax_ang = ax_cov.twinx()
                for col in ang_plotted:
                    ax_ang.plot(
                        ctrl_t, df_valid[col].values,
                        color=SIGMA_COLOURS[col], lw=1.1, alpha=ALPHA,
                        ls='--', label=SIGMA_LABELS[col],
                    )
                ax_ang.set_ylabel('1σ angular (rad / rad·s⁻¹)', fontsize=9)
                ax_ang.set_ylim(bottom=0)

                # Merge legends from both axes
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
            covar_vals = df_valid[trace_col].values
            ax_cov.plot(ctrl_t, covar_vals, color='#c0392b', lw=1.0, alpha=ALPHA,
                        label='trace(P)')
            ax_cov.set_ylabel('trace(P)', fontsize=10)
            ax_cov.set_title('Filter covariance — trace(P)', fontsize=10)
            ax_cov.legend(fontsize=9, loc='upper right')

        ax_cov.set_ylim(bottom=0)
        ax_cov.grid(True, alpha=0.3)
        ax_cov.set_xlabel('Time (s)', fontsize=10)
    else:
        axes[-1].set_xlabel('Time (s)', fontsize=10)

    fig1.tight_layout(rect=[0, 0, 1, 0.97])

    out1 = out_dir / 'error_vs_time.png'
    fig1.savefig(out1, dpi=500, bbox_inches='tight')
    print(f"Saved: {out1}")

    # ── Figure 2 – 2-D Trajectory ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 8))
    fig2.suptitle(
        'Landing Pad 2D Trajectory\n(true position outliers removed)',
        fontsize=14, fontweight='bold',
    )

    ax2.plot(true_x, true_y,
             color=TRUE_C, lw=1.8, label='True position', zorder=3)
    ax2.plot(df_valid['landing_pad_rel_x'], df_valid['landing_pad_rel_y'],
             color=FILT_C, lw=1.2, alpha=0.85, label='Filter estimate', zorder=4)

    if april_valid.any():
        ax2.scatter(
            df_valid.loc[april_valid, 'landing_pad_april_rel_x_raw'],
            df_valid.loc[april_valid, 'landing_pad_april_rel_y_raw'],
            c=RAW_C, marker='x', s=22, alpha=0.65,
            label='AprilTag raw', zorder=2
        )

    if yolo_lp_valid.any():
        ax2.scatter(
            df_valid.loc[yolo_lp_valid, 'landing_pad_yolo_LP_rel_x_raw'],
            df_valid.loc[yolo_lp_valid, 'landing_pad_yolo_LP_rel_y_raw'],
            c=YOLO_LP_C, marker='x', s=22, alpha=0.65,
            label='YOLO LP raw', zorder=2
        )

    if yolo_car_valid.any():
        ax2.scatter(
            df_valid.loc[yolo_car_valid, 'landing_pad_yolo_car_rel_x_raw'],
            df_valid.loc[yolo_car_valid, 'landing_pad_yolo_car_rel_y_raw'],
            c=YOLO_CAR_C, marker='x', s=22, alpha=0.65,
            label='YOLO car raw', zorder=2
        )

    ax2.plot(true_x[0],  true_y[0],  'go', ms=8, label='Start', zorder=5)
    ax2.plot(true_x[-1], true_y[-1], 'rs', ms=8, label='End',   zorder=5)

    ax2.set_xlabel('X position (m)', fontsize=11)
    ax2.set_ylabel('Y position (m)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    out2 = out_dir / 'trajectory_2d.png'
    fig2.savefig(out2, dpi=500, bbox_inches='tight')
    print(f"Saved: {out2}")

        # ── Figure 3 – NIS / NEES consistency analysis ──────────────────────────
    if has_nis or has_nees:

        fig3, axes3 = plt.subplots(
            2, 2,
            figsize=(12, 8)
        )

        fig3.suptitle(
            'UKF Consistency Analysis',
            fontsize=14,
            fontweight='bold'
        )

        # Flatten axes
        ax_nis_time   = axes3[0, 0]
        ax_nees_time  = axes3[0, 1]
        ax_nis_hist   = axes3[1, 0]
        ax_nees_hist  = axes3[1, 1]

        # ── NIS vs time ───────────────────────────────────────────────
        if has_nis:
            nis = df_valid['nis'].values
            nis_valid = np.isfinite(nis)

            ax_nis_time.plot(
                ctrl_t[nis_valid],
                nis[nis_valid],
                color='#8e44ad',
                lw=1.2,
                label='NIS'
            )

            ax_nis_time.axhline(
                9.49,
                color='red',
                linestyle='--',
                label='95% χ²(4)'
            )

            ax_nis_time.axhline(
                4,
                color='green',
                linestyle=':',
                label='Expected mean'
            )

            ax_nis_time.set_title('NIS vs Time')
            ax_nis_time.set_ylabel('NIS')
            ax_nis_time.grid(True, alpha=0.3)
            ax_nis_time.legend()

        # ── NEES vs time ──────────────────────────────────────────────
        if has_nees:
            nees = df_valid['nees'].values

            ax_nees_time.plot(
                ctrl_t,
                nees,
                color='#2980b9',
                lw=1.2,
                label='NEES'
            )

            ax_nees_time.axhline(
                9.348,
                color='red',
                linestyle='--',
                label='95% χ²(3)'
            )

            ax_nees_time.axhline(
                3,
                color='green',
                linestyle=':',
                label='Expected mean'
            )

            ax_nees_time.set_title('NEES vs Time')
            ax_nees_time.set_ylabel('NEES')
            ax_nees_time.grid(True, alpha=0.3)
            ax_nees_time.legend()

        # ── NIS histogram ─────────────────────────────────────────────
        if has_nis:
            nis = df_valid['nis'].values
            nis_valid = nis[np.isfinite(nis)]

            if len(nis_valid) > 0:
                ax_nis_hist.hist(
                    nis_valid,
                    bins=100,
                    density=True,
                    alpha=0.6,
                    label='Measured NIS'
                )

                x = np.linspace(
                    0,
                    max(15, nis_valid.max()),
                    400
                )

                ax_nis_hist.plot(
                    x,
                    chi2.pdf(x, df=4),
                    lw=2,
                    label='χ²(4)'
                )

                ax_nis_hist.axvline(
                    nis_valid.mean(),
                    linestyle='--',
                    label=f'Mean={nis_valid.mean():.2f}'
                )

            ax_nis_hist.set_title('NIS Distribution')
            ax_nis_hist.set_xlabel('NIS')
            ax_nis_hist.set_ylabel('Density')
            ax_nis_hist.grid(True, alpha=0.3)
            ax_nis_hist.legend()

        # ── NEES histogram ────────────────────────────────────────────
        if has_nees:
            nees = df_valid['nees'].values
            nees_valid = nees[np.isfinite(nees)]

            if len(nees_valid) > 0:
                ax_nees_hist.hist(
                    nees_valid,
                    bins=100,
                    density=True,
                    alpha=0.6,
                    label='Measured NEES'
                )

                x = np.linspace(
                    0,
                    max(20, nees_valid.max()),
                    400
                )

                ax_nees_hist.plot(
                    x,
                    chi2.pdf(x, df=3),
                    lw=2,
                    label='χ²(3)'
                )

                ax_nees_hist.axvline(
                    nees_valid.mean(),
                    linestyle='--',
                    label=f'Mean={nees_valid.mean():.2f}'
                )

            ax_nees_hist.set_title('NEES Distribution')
            ax_nees_hist.set_xlabel('NEES')
            ax_nees_hist.set_ylabel('Density')
            ax_nees_hist.grid(True, alpha=0.3)
            ax_nees_hist.legend()

        fig3.tight_layout(rect=[0, 0, 1, 0.95])

        out3 = out_dir / 'consistency_analysis.png'
        fig3.savefig(
            out3,
            dpi=500,
            bbox_inches='tight'
        )

        print(f"Saved: {out3}")

    # ── Figure 4 – Pipeline latency analysis ───────────────────────────────
    if has_apriltag_lag or has_yolo_lag or has_meas_age:

        # --------------------------------------------------------------
        # Subplot 1: latency vs time
        # Subplot 2: mean latency breakdown
        # --------------------------------------------------------------

        fig5, (ax_lag_time, ax_lag_bar) = plt.subplots(
            1,
            2,
            figsize=(15, 5)
        )

        fig5.suptitle(
            'Pipeline Latency Analysis',
            fontsize=14,
            fontweight='bold'
        )

        # ── Latency vs time ───────────────────────────────────────────
        if has_apriltag_lag:
            for col in apriltag_lag_cols:
                ax_lag_time.plot(
                    ctrl_t,
                    df_valid[col].values,
                    lw=1.1,
                    alpha=0.80,
                    color=LAG_COLOURS[col],
                    label='AprilTag ' + col.replace('_lag', '').replace('_', ' ')
                )

            ax_lag_time.plot(
                ctrl_t,
                df_valid['apriltag_total_lag'].values,
                lw=1.4,
                alpha=0.90,
                color=LAG_COLOURS['apriltag_total_lag'],
                ls='--',
                label='AprilTag total'
            )

        if has_yolo_lag:
            for col in yolo_lag_cols:
                ax_lag_time.plot(
                    ctrl_t,
                    df_valid[col].values,
                    lw=1.1,
                    alpha=0.80,
                    color=LAG_COLOURS[col],
                    label='YOLO ' + col.replace('yolo_', '').replace('_lag', '').replace('_', ' ')
                )

            ax_lag_time.plot(
                ctrl_t,
                df_valid['yolo_total_lag'].values,
                lw=1.4,
                alpha=0.90,
                color=LAG_COLOURS['yolo_total_lag'],
                ls='-.',
                label='YOLO total'
            )

        if has_meas_age:
            ax_lag_time.plot(
                ctrl_t,
                df_valid['total_lag'].values,
                lw=0.9,
                alpha=0.55,
                color=LAG_COLOURS['total_lag'],
                ls=':',
                label='Measurement age'
            )

        ax_lag_time.set_title('Latency vs Time')
        ax_lag_time.set_xlabel('Time (s)')
        ax_lag_time.set_ylabel('Latency / age (ms)')
        ax_lag_time.grid(True, alpha=0.3)
        ax_lag_time.legend(fontsize=7, ncol=2)

        # ── Mean breakdown bar chart ──────────────────────────────────
        labels = []
        mean_lags = []
        bar_colors = []

        if has_apriltag_lag:
            labels.extend([
                'AprilTag\nCamera → Image',
                'AprilTag\nImage → Transform',
                'AprilTag\nTransform → UKF',
                'AprilTag\nTotal',
            ])
            mean_lags.extend([
                df_valid['cam_to_image_lag'].mean(),
                df_valid['image_to_transform_lag'].mean(),
                df_valid['transform_to_UKF_lag'].mean(),
                df_valid['apriltag_total_lag'].mean(),
            ])
            bar_colors.extend([
                LAG_COLOURS['cam_to_image_lag'],
                LAG_COLOURS['image_to_transform_lag'],
                LAG_COLOURS['transform_to_UKF_lag'],
                LAG_COLOURS['apriltag_total_lag'],
            ])

        if has_yolo_lag:
            labels.extend([
                'YOLO\nCamera → Image',
                'YOLO\nImage → Transform',
                'YOLO\nTransform → UKF',
                'YOLO\nTotal',
            ])
            mean_lags.extend([
                df_valid['yolo_cam_to_image_lag'].mean(),
                df_valid['yolo_image_to_transform_lag'].mean(),
                df_valid['yolo_transform_to_UKF_lag'].mean(),
                df_valid['yolo_total_lag'].mean(),
            ])
            bar_colors.extend([
                LAG_COLOURS['yolo_cam_to_image_lag'],
                LAG_COLOURS['yolo_image_to_transform_lag'],
                LAG_COLOURS['yolo_transform_to_UKF_lag'],
                LAG_COLOURS['yolo_total_lag'],
            ])

        if has_meas_age:
            labels.append('Measurement\nAge')
            mean_lags.append(df_valid['total_lag'].mean())
            bar_colors.append(LAG_COLOURS['total_lag'])

        bars = ax_lag_bar.bar(
            labels,
            mean_lags,
            color=bar_colors,
            alpha=0.85
        )

        ax_lag_bar.set_title('Mean Latency Breakdown')
        ax_lag_bar.set_ylabel('Mean latency / age (ms)')
        ax_lag_bar.tick_params(axis='x', labelsize=8)
        ax_lag_bar.grid(True, axis='y', alpha=0.3)

        # Add values above bars
        max_mean = max(mean_lags) if mean_lags else 1.0
        for bar, value in zip(bars, mean_lags):
            ax_lag_bar.text(
                bar.get_x() + bar.get_width()/2,
                value + 0.02 * max_mean,
                f'{value:.1f} ms',
                ha='center',
                va='bottom',
                fontsize=8
            )

        fig5.tight_layout(rect=[0, 0, 1, 0.95])

        out5 = out_dir / 'pipeline_latency.png'
        fig5.savefig(
            out5,
            dpi=500,
            bbox_inches='tight'
        )

        print(f"Saved: {out5}")

    else:
        print(
            "WARNING: Pipeline latency columns missing "
            "– latency analysis skipped."
        )

    # ── Print summary stats ───────────────────────────────────────────────
    print('\n=== Error summary ===')

    if april_valid.any():
        vals = april_err_x[april_valid]
        print(f'  April raw X   mean: {vals.mean():+.3f} m   RMS: {np.sqrt(np.mean(vals**2)):.3f} m')
        vals = april_err_y[april_valid]
        print(f'  April raw Y   mean: {vals.mean():+.3f} m   RMS: {np.sqrt(np.mean(vals**2)):.3f} m')
        vals = raw_yaw_err[april_valid]
        print(f'  April raw yaw mean: {np.mean(vals):+.4f} rad '
            f'({np.degrees(np.mean(vals)):+.2f}°)   '
            f'RMS: {np.sqrt(np.mean(vals**2)):.4f} rad '
            f'({np.degrees(np.sqrt(np.mean(vals**2))):.2f}°)')
        vals = april_err_abs[april_valid]
        print(f'  April raw |e|  mean: {vals.mean():.3f} m   max: {vals.max():.3f} m')

    if yolo_lp_valid.any():
        vals = yolo_lp_err_x[yolo_lp_valid]
        print(f'  YOLO LP raw X mean: {vals.mean():+.3f} m   RMS: {np.sqrt(np.mean(vals**2)):.3f} m')
        vals = yolo_lp_err_y[yolo_lp_valid]
        print(f'  YOLO LP raw Y mean: {vals.mean():+.3f} m   RMS: {np.sqrt(np.mean(vals**2)):.3f} m')
        vals = yolo_lp_err_abs[yolo_lp_valid]
        print(f'  YOLO LP raw |e| mean: {vals.mean():.3f} m   max: {vals.max():.3f} m')

    if yolo_car_valid.any():
        vals = yolo_car_err_x[yolo_car_valid]
        print(f'  YOLO car raw X mean: {vals.mean():+.3f} m   RMS: {np.sqrt(np.mean(vals**2)):.3f} m')
        vals = yolo_car_err_y[yolo_car_valid]
        print(f'  YOLO car raw Y mean: {vals.mean():+.3f} m   RMS: {np.sqrt(np.mean(vals**2)):.3f} m')
        vals = yolo_car_err_abs[yolo_car_valid]
        print(f'  YOLO car raw |e| mean: {vals.mean():.3f} m   max: {vals.max():.3f} m')

    print(f'  Filt  X   mean: {filt_err_x.mean():+.3f} m   RMS: {np.sqrt(np.mean(filt_err_x**2)):.3f} m')
    print(f'  Filt  Y   mean: {filt_err_y.mean():+.3f} m   RMS: {np.sqrt(np.mean(filt_err_y**2)):.3f} m')
    print(f'  Filt  yaw mean: {np.mean(filt_yaw_err):+.4f} rad '
        f'({np.degrees(np.mean(filt_yaw_err)):+.2f}°)   '
        f'RMS: {np.sqrt(np.mean(filt_yaw_err**2)):.4f} rad '
        f'({np.degrees(np.sqrt(np.mean(filt_yaw_err**2))):.2f}°)')
    print(f'  Filt |e|  mean: {filt_err_abs.mean():.3f} m   max: {filt_err_abs.max():.3f} m')

    if has_apriltag_lag:
        lag_vals = df_valid['apriltag_total_lag'].values
        print(f'\n=== AprilTag total pipeline latency summary ===')
        print(f'  mean: {lag_vals.mean():.1f} ms   max: {lag_vals.max():.1f} ms   '
              f'min: {lag_vals.min():.1f} ms')

    if has_yolo_lag:
        lag_vals = df_valid['yolo_total_lag'].values
        print(f'\n=== YOLO total pipeline latency summary ===')
        print(f'  mean: {lag_vals.mean():.1f} ms   max: {lag_vals.max():.1f} ms   '
              f'min: {lag_vals.min():.1f} ms')

    if has_meas_age:
        lag_vals = df_valid['total_lag'].values
        print(f'\n=== measurement age summary ===')
        print(f'  mean: {lag_vals.mean():.1f} ms   max: {lag_vals.max():.1f} ms   '
              f'min: {lag_vals.min():.1f} ms')

    if has_per_state:
        print(f'\n=== UKF per-state 1σ summary (mean ± std) ===')
        for col in sigma_cols:
            vals = df_valid[col].values
            print(f'  {SIGMA_LABELS[col]:25s}  mean: {vals.mean():.4f}   '
                  f'max: {vals.max():.4f}   min: {vals.min():.4f}')
    elif has_trace_only:
        covar_vals = df_valid[trace_col].values
        print(f'\n=== {trace_col} (trace) summary ===')
        print(f'  mean: {covar_vals.mean():.4f}   max: {covar_vals.max():.4f}   '
              f'min: {covar_vals.min():.4f}')
        
    if has_nis:
        nis = df_valid['nis'].values
        nis_valid = nis[np.isfinite(nis)]

        print('\n=== NIS summary ===')

        if len(nis_valid) > 0:
            print(f'  mean: {nis_valid.mean():.3f}')
            print(f'  max : {nis_valid.max():.3f}')
            print(f'  min : {nis_valid.min():.3f}')
        else:
            print('  no finite NIS entries')
    
    if has_nees:
        nees = df_valid['nees'].values

        print('\n=== NEES summary ===')
        print(f'  mean: {np.nanmean(nees):.3f}')
        print(f'  max : {np.nanmax(nees):.3f}')
        print(f'  min : {np.nanmin(nees):.3f}')

        upper = chi2.ppf(0.95, df=4)
        frac_bad = np.nanmean(nees > upper)
        print(f"NEES above 95% bound: {frac_bad*100:.1f}%")

    plt.show()


if __name__ == '__main__':
    main()