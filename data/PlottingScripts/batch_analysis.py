"""
batch_analysis.py  –  Trial-set statistics for the vision landing runs
=====================================================================
Wraps plot_errors.py (keep both in the same folder) and runs every controller CSV in
a trajectory folder laid out as

    <root>/spd_xx/box_measurement/*.csv            (UKF with OOSM latency compensation)
    <root>/spd_xx/no_latency_compensation/*.csv    (UKF with OOSM disabled)

Usage:
    Set ROOT (and optionally OUT_DIR / PER_RUN_PLOTS) in the config below, then
        python batch_analysis.py
    Command-line arguments still work and override the config:
        python batch_analysis.py some_path/Straight --per-run-plots --out results_dir

Per run it computes
    Estimation error (estimate - truth, car frame), scored over the window
        [start,  first 5000 or 6000]
    i.e. once the filter has settled, up to the end of the FIRST landing attempt. Data
    from any re-attempt is ignored. The start is set by WINDOW_START:
        'converged'  the first time after 3000 that the UKF reports itself converged
                     (box update active, sigma_xy and sigma_yaw under thresholds) for
                     CONV_DWELL_S. Truth-free, so it can't cherry-pick good stretches.
                     Falls back to 'fixed' if it never converges (CONV_FALLBACK_TO_FIXED).
        'fixed'      first 3000 + SETTLE_S
    Same scoring as plot_errors.py:
        filt   filter estimate at the UKF tick it is valid at (est_time)
        held   YOLO car fit held until the next one, at the row time (the live competitor)
        stamp  YOLO car fit at its own image stamp
        post   filter state at the YOLO image stamp (NaN when OOSM is off)
    Landing outcome, from the drone's final resting pose (median of the last REST_AVG_S)
        success  = upright (FCU roll/pitch within UPRIGHT_MAX_TILT_DEG) AND its centre is
                   over the pad AND its height above the pad top is in REST_HEIGHT_RANGE
                   (rules out the hood, the roof behind the pad and the ground)
        landing error (long / lat / radial) from the pad centre, successful runs only
        time to acquisition = first 4000 - first REF_STATE
        time to landing     = first 6000 - first REF_STATE  (includes any re-attempts)
        attempts            = number of entries into 4000

Outputs (in <root>/batch_results unless --out is given):
    runs.csv              one row per CSV, every metric above
    summary.csv           per speed and mode: mean / std over runs, paired filter-vs-raw
                          test, success rate with 95% Wilson interval
    mode_comparison.csv   per speed: OOSM vs no compensation filter RMSE (Mann-Whitney U)
    landing_clusters.png  final resting positions in the pad frame, one panel per mode
    rmse_vs_speed.png     lateral / longitudinal / height RMSE vs speed
    landing_vs_speed.png  success rate, landing error, acquisition and landing time vs speed
    per_run/...           plot_errors.py figures for each run (only with --per-run-plots)

Dependencies:
    pip install pandas numpy matplotlib scipy
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # batch job: save figures, never open windows
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from scipy import stats
from scipy.interpolate import interp1d

import plot_errors as pe


# ── Config ─────────────────────────────────────────────────────────────────

# Trajectory folder holding the spd_xx sub-folders
ROOT = Path(r'Documents/Thesis/DroneAutoLander/data/FinalResults/Straight')
OUT_DIR = None          # None -> ROOT/batch_results
PER_RUN_PLOTS = False   # True: also save plot_errors.py figures for every run (slow)

# Sub-folder name -> label. Folders not listed here are skipped.
MODES = {
    'box_measurement':         'OOSM (latency compensated)',
    'no_latency_compensation': 'No latency compensation',
}
MODE_COLOURS = {'box_measurement': pe.FILT_C, 'no_latency_compensation': '#d95f02'}
MODE_STYLES  = {'box_measurement': '-',       'no_latency_compensation': '--'}
MODE_MARKERS = {'box_measurement': 'o',       'no_latency_compensation': 's'}

# Start of the error window (end is always the first 5000/6000)
WINDOW_START = 'converged'   # 'converged' (filter's own covariance) or 'fixed' (3000 + SETTLE_S)
SETTLE_S     = 5.0           # 'fixed': s after the first 3000

# 'converged': all of these must hold continuously for CONV_DWELL_S
CONV_REQUIRE_BOX_MODE  = True   # UKF using the pixel-space box update (yaw known)
CONV_POS_SIGMA         = 0.50   # m, hypot(sigma_px, sigma_py) below this
CONV_YAW_SIGMA         = 0.20   # rad, sigma_yaw below this (box mode itself needs < 0.25)
CONV_DWELL_S           = 2.0    # s the criteria must hold; the window starts at its end
CONV_MIN_DELAY_S       = 0.0    # s, never start earlier than first 3000 + this
CONV_FALLBACK_TO_FIXED = True   # never converged -> use 'fixed'; False -> run scores NaN
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

# Speed-plot options
SPEED_UNIT      = 'km/h'   # label only: speeds are read from the spd_xx folder names as-is
OFFSET_POINTS   = False    # True: nudge overlapping series sideways so their bars don't overlap
SHOW_ERROR_BARS = False     # ±1σ over runs (success rate: 95% Wilson interval)
SHOW_ELLIPSES   = False     # 1σ spread ellipses on the landing cluster plot

# Landing cluster axis limits (pad frame, m). Points outside are drawn as arrowheads on
# the edge, pointing the way they went, and counted in the panel title.
#   'car'   car footprint + 0.4 m margin
#   'auto'  stretch to fit every landing, however far off the car
#   ((x_min, x_max), (y_min, y_max))  custom, e.g. ((-1.5, 1.5), (-1.0, 1.0))
CLUSTER_LIMITS = 'car'

PER_RUN_DPI = 200    # plot_errors.py figures in --per-run-plots (its own default is 500)
DPI = 300


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
    """ 'spd_8' -> 8.0, 'spd_08' -> 8.0, 'spd_7p5' / 'spd_7.5' -> 7.5, else NaN. """
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
    """ First time after pad lock at which the UKF's own convergence criteria have held
        for CONV_DWELL_S (the window starts there, at the end of the dwell). Uses only
        what the filter knew, never the truth. NaN if it never happens before t_end.
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

    t_min = t_3000 + CONV_MIN_DELAY_S
    run_start = None
    for k in np.flatnonzero((t >= t_3000) & (t <= t_end)):
        if not ok[k]:
            run_start = None
            continue
        if run_start is None:
            run_start = t[k]
        if t[k] - run_start >= CONV_DWELL_S and t[k] >= t_min:
            return float(t[k])
    return np.nan


def analyse_run(csv_path):
    """ All metrics for one controller CSV. Returns (metrics dict, scoring window). """
    df = pd.read_csv(csv_path)
    t0 = float(df['timestamp'].iloc[0])
    t = df['timestamp'].to_numpy(dtype=float) - t0
    state = pd.to_numeric(df['controller_state'], errors='coerce').to_numpy()
    group = state // 1000

    out = {}

    # ── State timeline ──
    t_ref = first_time(t, state == REF_STATE) if REF_STATE % 1000 \
        else first_time(t, group == REF_STATE // 1000)
    t_3000 = first_time(t, group == 3)
    t_4000 = first_time(t, group == 4)
    t_6000 = first_time(t, group == 6)
    after_lock = np.isfinite(t_3000) & (t >= t_3000)
    t_first_end = first_time(t, after_lock & ((group == 5) | (group == 6)))

    entries_4000 = (group == 4) & np.r_[True, group[1:] != group[:-1]]
    out['attempts'] = int(entries_4000.sum())
    out['reached_6000'] = bool(np.isfinite(t_6000))
    out['t_acquisition'] = t_4000 - t_ref
    out['t_landing'] = t_6000 - t_ref

    # ── Scoring window: settled tracking up to the end of the first attempt ──
    w_end = t_first_end if np.isfinite(t_first_end) else t[-1]
    t_conv = converged_time(df, t, t_3000, w_end)   # always computed: also a result in its own right
    out['t_converged'] = t_conv - t_3000   # time from pad lock to filter convergence
    if WINDOW_START == 'converged' and np.isfinite(t_conv):
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
        out[f'{prefix}_n'] = int(np.isfinite(lat).sum())

    in_win = (t >= w_start) & (t <= w_end) if np.isfinite(w_start) else np.zeros(len(t), bool)

    # Filter estimate, valid at the UKF tick (est_time), not the log row
    est_raw = pd.to_numeric(df.get('est_time', pd.Series(np.zeros(len(t)))),
                            errors='coerce').to_numpy(dtype=float)
    est_t = np.where(est_raw > 0, est_raw - t0, t)
    m = in_win
    long, lat, tz = car_frame(est_t[m], df['landing_pad_rel_x'].to_numpy(float)[m],
                              df['landing_pad_rel_y'].to_numpy(float)[m])
    score('filt', long, lat, df['landing_pad_rel_z'].to_numpy(float)[m] - tz)

    # YOLO car fit (fallback fit or shadow fit), held and at its own stamp
    pre = 'landing_pad_yolo_car'
    if f'{pre}_raw_stamp' in df.columns:
        stamp = df[f'{pre}_raw_stamp'].to_numpy(dtype=float)
        rx = df[f'{pre}_rel_x_raw'].to_numpy(dtype=float)
        ry = df[f'{pre}_rel_y_raw'].to_numpy(dtype=float)
        rz = df[f'{pre}_rel_z_raw'].to_numpy(dtype=float)
        t_meas = np.where(stamp > 0, stamp - t0, np.nan)

        held = in_win & (stamp > 0) & np.isfinite(rx) & np.isfinite(ry)
        long, lat, tz = car_frame(t[held], rx[held], ry[held])
        score('held', long, lat, rz[held] - tz)

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
    return out, window


# ── Aggregation ────────────────────────────────────────────────────────────

ERROR_KEYS = [f'{src}_{f}' for src in ('filt', 'held', 'stamp', 'post')
              for f in ('lat', 'long', 'xy', 'z', 'lat_bias')]


def summarise(runs):
    rows = []
    for (speed, mode), g in runs.groupby(['speed', 'mode'], sort=True):
        r = dict(speed=speed, mode=mode, n_runs=len(g))
        for k in ERROR_KEYS:
            if k in g:
                r[f'{k}_mean'], r[f'{k}_std'] = g[k].mean(), g[k].std(ddof=1)

        # Paired, per run: raw held minus filter (positive = filter better)
        for f in ('lat', 'long', 'xy', 'z'):
            if f'held_{f}' in g:
                d, ci, p, n = paired_stats(g[f'held_{f}'], g[f'filt_{f}'])
                r[f'gain_{f}_mean'], r[f'gain_{f}_ci95'], r[f'gain_{f}_wilcoxon_p'] = d, ci, p
                pct, lo, hi, p_log, _ = improvement_stats(g[f'filt_{f}'], g[f'held_{f}'])
                r[f'improve_{f}_pct'], r[f'improve_{f}_ci_lo'], r[f'improve_{f}_ci_hi'] = pct, lo, hi
                r[f'improve_{f}_wilcoxon_p'] = p_log

        k, n = int(g['success'].sum()), len(g)
        lo, hi = wilson(k, n)
        r.update(success_n=k, success_pct=100 * k / n, success_ci_lo=100 * lo, success_ci_hi=100 * hi,
                 first_attempt_success_pct=100 * g['first_attempt_success'].mean())

        ok = g[g['success']]
        for col in ('rest_error', 'rest_long', 'rest_lat'):
            r[f'{col}_mean'], r[f'{col}_std'] = ok[col].mean(), ok[col].std(ddof=1)
        for col in ('t_acquisition', 't_landing', 'attempts'):
            r[f'{col}_mean'], r[f'{col}_std'] = g[col].mean(), g[col].std(ddof=1)
        r['attempts_max'] = g['attempts'].max()
        r['t_converged_mean'], r['t_converged_std'] = g['t_converged'].mean(), g['t_converged'].std(ddof=1)
        r['n_not_converged'] = int(g['t_converged'].isna().sum())   # excluded from t_converged stats
        r['window_s_mean'] = g['window_s'].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def compare_modes(runs):
    """ Per speed: filter RMSE with vs without latency compensation (different runs, so
        unpaired: Mann-Whitney U). """
    rows = []
    a_mode, b_mode = list(MODES)[:2]
    for speed, g in runs.groupby('speed'):
        r = dict(speed=speed)
        for f in ('lat', 'long', 'xy'):
            a = g.loc[g['mode'] == a_mode, f'filt_{f}'].dropna()
            b = g.loc[g['mode'] == b_mode, f'filt_{f}'].dropna()
            r[f'{a_mode}_{f}'], r[f'{b_mode}_{f}'] = a.mean(), b.mean()
            r[f'{f}_mannwhitney_p'] = (stats.mannwhitneyu(a, b).pvalue
                                       if len(a) >= 3 and len(b) >= 3 else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


# ── Figures ────────────────────────────────────────────────────────────────

def speed_colours(speeds):
    """ One hue, light -> dark with speed (sequential: speed is a magnitude). """
    speeds = sorted(speeds)
    shades = np.linspace(0.40, 0.95, len(speeds)) if len(speeds) > 1 else [0.75]
    return {s: plt.cm.Blues(v) for s, v in zip(speeds, shades)}


def cov_ellipse(ax, pts, colour):
    """ 1-sigma covariance ellipse of the successful landings for one speed. """
    if len(pts) < 3:
        return
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(pts.mean(axis=0), 2 * np.sqrt(vals[1]), 2 * np.sqrt(vals[0]),
                         angle=ang, fill=False, ec=colour, lw=1.4, zorder=3))


def cluster_limits(runs):
    """ (xlim, ylim) for the landing cluster plot, from CLUSTER_LIMITS. """
    px, py = PAD_POSITION[:2]
    x0, x1, y0, y1 = CAR_FOOTPRINT
    car = ((x0 - px - 0.4, x1 - px + 0.4), (y0 - py - 0.4, y1 - py + 0.4))
    if CLUSTER_LIMITS == 'car':
        return car
    if CLUSTER_LIMITS == 'auto':
        xs = np.r_[car[0], runs['rest_long'].dropna() - 0.4, runs['rest_long'].dropna() + 0.4]
        ys = np.r_[car[1], runs['rest_lat'].dropna() - 0.4, runs['rest_lat'].dropna() + 0.4]
        return (xs.min(), xs.max()), (ys.min(), ys.max())
    (a, b), (c, d) = CLUSTER_LIMITS
    return (float(a), float(b)), (float(c), float(d))


def scatter_clipped(ax, x, y, colour, failed, xlim, ylim):
    """ Scatter landings; those outside the view become edge arrowheads. Returns the
        number drawn off-chart. """
    x, y = np.asarray(x, float), np.asarray(y, float)
    inside = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    if failed:
        ax.scatter(x[inside], y[inside], s=40, color=colour, marker='x', lw=1.4, zorder=4)
    else:
        ax.scatter(x[inside], y[inside], s=36, color=colour, ec='white', lw=0.6, zorder=4)

    inset_x, inset_y = 0.02 * (xlim[1] - xlim[0]), 0.02 * (ylim[1] - ylim[0])
    for xi, yi in zip(x[~inside], y[~inside]):
        cx = np.clip(xi, xlim[0] + inset_x, xlim[1] - inset_x)
        cy = np.clip(yi, ylim[0] + inset_y, ylim[1] - inset_y)
        # Point the arrowhead along the axis it left by the most (in view widths)
        dx = (xi - cx) / (xlim[1] - xlim[0])
        dy = (yi - cy) / (ylim[1] - ylim[0])
        marker = ('>' if dx > 0 else '<') if abs(dx) >= abs(dy) else ('^' if dy > 0 else 'v')
        ax.scatter(cx, cy, s=55, marker=marker, zorder=5, lw=1.2,
                   fc='none' if failed else colour, ec=colour)
    return int((~inside).sum())


def plot_clusters(runs, out_dir):
    modes = [m for m in MODES if m in set(runs['mode'])]
    cols = speed_colours(runs['speed'].unique())
    xlim, ylim = cluster_limits(runs)
    # Panel width follows the view's aspect ratio (equal axes), so custom views don't
    # leave large gaps between panels
    aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    panel_w = float(np.clip(3.0 * aspect, 2.8, 7.5))
    fig, axes = plt.subplots(1, len(modes), figsize=(panel_w * len(modes) + 2.4, 3.9),
                             sharex=True, sharey=True, squeeze=False)
    px, py = PAD_POSITION[:2]
    for ax, mode in zip(axes[0], modes):
        x0, x1, y0, y1 = CAR_FOOTPRINT
        ax.add_patch(Rectangle((x0 - px, y0 - py), x1 - x0, y1 - y0, fc='0.93', ec='0.6',
                               lw=0.8, zorder=0, label='Car body'))
        ax.add_patch(Rectangle((-PAD_SIZE[0] / 2, -PAD_SIZE[1] / 2), *PAD_SIZE, fc='none',
                               ec='0.25', lw=1.2, zorder=1, label='Landing pad'))
        # Direction of travel, pinned to the panel corner so it shows in any view
        ax.annotate('', xy=(0.17, 0.93), xytext=(0.03, 0.93), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='0.45', lw=1.0))
        ax.text(0.10, 0.95, 'fwd', transform=ax.transAxes, ha='center', va='bottom',
                fontsize=8, color='0.4')
        ax.plot(0, 0, marker='+', color='0.25', ms=10, mew=1.2, zorder=2)

        g = runs[runs['mode'] == mode]
        n_off = 0
        for speed, gs in g.groupby('speed'):
            ok, bad = gs[gs['success']], gs[~gs['success']]
            n_off += scatter_clipped(ax, ok['rest_long'], ok['rest_lat'], cols[speed], False, xlim, ylim)
            n_off += scatter_clipped(ax, bad['rest_long'], bad['rest_lat'], cols[speed], True, xlim, ylim)
            if SHOW_ELLIPSES:
                cov_ellipse(ax, ok[['rest_long', 'rest_lat']].to_numpy(), cols[speed])

        ax.set_title(MODES[mode] + (f'\n{n_off} landing(s) outside view' if n_off else ''),
                     fontsize=11)
        ax.set_xlabel('Longitudinal from pad centre (m)  (+ = forward)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel('Lateral from pad centre (m)  (+ = left)')

    axes[0][0].set_xlim(*xlim)    # shared axes: sets every panel
    axes[0][0].set_ylim(*ylim)

    handles = [Line2D([], [], ls='none', marker='o', color=cols[s], mec='white', ms=7,
                      label=f'{s:g} {SPEED_UNIT}') for s in sorted(cols)]
    handles += [Line2D([], [], ls='none', marker='x', color='0.3', ms=7, mew=1.4, label='Failed'),
                Line2D([], [], ls='none', marker='>', color='0.3', ms=7, label='Outside view (at edge)'),
                Line2D([], [], color='0.3', lw=1.4, label='1σ spread (successes)'),
                Rectangle((0, 0), 1, 1, fc='none', ec='0.25', label='Landing pad')]
    axes[0][-1].legend(handles=handles, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.suptitle('Final resting positions (pad frame)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / 'landing_clusters.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_rmse(summary, out_dir):
    modes = [m for m in MODES if m in set(summary['mode'])]
    fig, axes2 = plt.subplots(2, 3, figsize=(15, 8.6), sharex=True)
    axes, axes_imp = axes2[0], axes2[1]
    speeds = sorted(summary['speed'].unique())
    step = 0.02 * max(np.ptp(summary['speed']), 1.0) if OFFSET_POINTS else 0.0
    offsets = {('filt', m): (2 * i) * step for i, m in enumerate(modes)}
    offsets.update({('held', m): (2 * i + 1) * step for i, m in enumerate(modes)})
    for ax, f, title in zip(axes, ('lat', 'long', 'z'),
                            ('Lateral RMSE', 'Longitudinal RMSE', 'Height RMSE')):
        for mode in modes:
            g = summary[summary['mode'] == mode].sort_values('speed')
            for src, colour, name in (('filt', pe.FILT_C, 'Filter'),
                                      ('held', pe.YOLO_CAR_C, 'Raw Measurement')):
                col = f'{src}_{f}_mean'
                if col not in g or not g[col].notna().any():
                    continue
                ax.errorbar(g['speed'] + offsets[(src, mode)], g[col], yerr=g[f'{src}_{f}_std'] if SHOW_ERROR_BARS else None,
                            color=colour, ls=MODE_STYLES[mode], marker=MODE_MARKERS[mode],
                            ms=5, lw=1.4, capsize=3, label=f'{name}, {MODES[mode]}')
        ax.set_title(title, fontsize=11)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('RMSE (m), mean ± 1σ over runs' if SHOW_ERROR_BARS else 'RMSE (m), mean over runs')
    axes[0].legend(fontsize=8)

    # Bottom row: relative improvement of the filter over the raw fit held (per-run log ratio)
    for ax, f, title in zip(axes_imp, ('lat', 'long', 'z'),
                            ('Lateral % Improvement - Filter vs Raw Measurement',
                             'Longitudinal % Improvement - Filter vs Raw Measurement',
                             'Height % Improvement - Filter vs Raw Measurement')):
        for i, mode in enumerate(modes):
            g = summary[summary['mode'] == mode].sort_values('speed')
            col = f'improve_{f}_pct'
            if col not in g or not g[col].notna().any():
                continue
            yerr = (np.vstack([g[col] - g[f'improve_{f}_ci_lo'], g[f'improve_{f}_ci_hi'] - g[col]])
                    if SHOW_ERROR_BARS else None)
            ax.errorbar(g['speed'] + i * step, g[col], yerr=yerr, color=MODE_COLOURS[mode],
                        ls=MODE_STYLES[mode], marker=MODE_MARKERS[mode], ms=5, lw=1.4,
                        capsize=3, label=MODES[mode])
        ax.axhline(0, color='0.4', lw=0.8, ls=':')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f'Car speed ({SPEED_UNIT})')
        ax.set_xticks(speeds)
        ax.grid(True, alpha=0.3)
    axes_imp[0].set_ylabel('Filter vs Raw Measurement (%), + = filter better'
                           + ('\n(geometric mean, 95% CI)' if SHOW_ERROR_BARS else '\n(geometric mean)'))
    axes_imp[0].legend(fontsize=8)
    start = ('filter converged' if WINDOW_START == 'converged' else f'3000 + {SETTLE_S:g} s')
    fig.suptitle(f'Estimation error vs speed  (window: {start} to first 5000/6000)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = out_dir / 'rmse_vs_speed.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_landing(summary, out_dir):
    modes = [m for m in MODES if m in set(summary['mode'])]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
    # Top row: outcome. Bottom row: timeline after pad lock, in the order things happen.
    panels = [
        (axes[0, 0], 'success_pct', None, 'Success rate (%)', 'Landing Success rate'),
        (axes[0, 1], 'rest_error_mean', 'rest_error_std', 'Landing error (m)',
         'Landing error from pad centre (successful runs)'),
        (axes[0, 2], 'attempts_mean', 'attempts_std', 'Attempts', 'Landing attempts'),
        (axes[1, 0], 't_converged_mean', 't_converged_std', 'Time (s)',
         'Time to filter convergence (from first landing pad lock)'),
        (axes[1, 1], 't_acquisition_mean', 't_acquisition_std', 'Time (s)',
         f'Time from first landing pad lock to start of descent'),
        (axes[1, 2], 't_landing_mean', 't_landing_std', 'Time (s)',
         f'Time from first landing pad lock to complete landing'),
    ]
    step = 0.02 * max(np.ptp(summary['speed']), 1.0) if OFFSET_POINTS else 0.0
    for ax, col, err, ylabel, title in panels:
        for i, mode in enumerate(modes):
            g = summary[summary['mode'] == mode].sort_values('speed')
            if col == 'success_pct':
                yerr = np.vstack([g[col] - g['success_ci_lo'], g['success_ci_hi'] - g[col]])
            else:
                yerr = g[err]
            if not SHOW_ERROR_BARS:
                yerr = None
            ax.errorbar(g['speed'] + i * step, g[col], yerr=yerr, color=MODE_COLOURS[mode],
                        ls=MODE_STYLES[mode], marker=MODE_MARKERS[mode], ms=5, lw=1.4,
                        capsize=3, label=MODES[mode])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=105 if col == 'success_pct' else None)
    for ax in axes[1]:
        ax.set_xlabel(f'Car speed ({SPEED_UNIT})')
        ax.set_xticks(sorted(summary['speed'].unique()))
    axes[0, 0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Landing performance vs speed', fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = out_dir / 'landing_vs_speed.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


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
              f"  |  mean window {r['window_s_mean']:.1f} s")
        fate = ('window used fixed fallback' if CONV_FALLBACK_TO_FIXED else 'not scored') \
            if WINDOW_START == 'converged' else 'window is fixed anyway'
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
                  f"z {pm(r[f'{src}_z_mean'], r[f'{src}_z_std'])}  m")
        for f in ('lat', 'long', 'xy', 'z'):
            if f'gain_{f}_mean' in r and np.isfinite(r[f'gain_{f}_mean']):
                p = r[f'gain_{f}_wilcoxon_p']
                print(f"  gain {f:<4s} {r[f'gain_{f}_mean']:+.3f} ± {r[f'gain_{f}_ci95']:.3f} m (95% CI)"
                      f"   Wilcoxon p = {'n<6' if not np.isfinite(p) else f'{p:.3f}'}")
        for f in ('lat', 'long', 'xy', 'z'):
            if f'improve_{f}_pct' in r and np.isfinite(r[f'improve_{f}_pct']):
                p = r[f'improve_{f}_wilcoxon_p']
                print(f"  improvement {f:<4s} {r[f'improve_{f}_pct']:+.0f}% "
                      f"(95% CI {r[f'improve_{f}_ci_lo']:+.0f} to {r[f'improve_{f}_ci_hi']:+.0f}%)"
                      f"   Wilcoxon p (log ratio) = {'n<6' if not np.isfinite(p) else f'{p:.3f}'}")

    print('\n' + '=' * 100)
    print('LANDING  (landing error over successful runs; times over all runs that reached the state)')
    print('=' * 100)
    hdr = f"{'speed':>6s}  {'mode':<26s} {'success':>18s} {'1st-att':>8s} {'error (m)':>14s} " \
          f"{'long (m)':>15s} {'lat (m)':>15s} {'t_acq (s)':>12s} {'t_land (s)':>12s} {'attempts':>10s}"
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
              f"{pm(r['attempts_mean'], r['attempts_std'], '{:.1f}'):>10s}")
    print("  success interval (95% Wilson) is in summary.csv: success_ci_lo / success_ci_hi")

    if len(modes_cmp):
        a, b = list(MODES)[:2]
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


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('root', type=Path, nargs='?', default=ROOT,
                    help='trajectory folder (default: ROOT in the config)')
    ap.add_argument('--out', type=Path, default=OUT_DIR,
                    help='output folder (default: OUT_DIR, else <root>/batch_results)')
    ap.add_argument('--per-run-plots', action='store_true', default=PER_RUN_PLOTS,
                    help='also run plot_errors.py on every CSV over the same window')
    args = ap.parse_args()

    if not args.root.is_dir():
        print(f"Root folder not found: {args.root.resolve()}  (set ROOT in the config)")
        return
    out_dir = Path(args.out) if args.out else args.root / 'batch_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    pe.DPI = PER_RUN_DPI

    rows = []
    for spd_dir in sorted(args.root.glob('spd_*'), key=lambda p: parse_speed(p.name)):
        speed = parse_speed(spd_dir.name)
        for mode in MODES:
            csvs = sorted((spd_dir / mode).glob('*.csv'))
            if not csvs:
                print(f"NOTE: no CSVs in {spd_dir.name}/{mode}")
                continue
            for csv in csvs:
                try:
                    m, window = analyse_run(csv)
                except Exception as e:   # one broken log shouldn't kill the batch
                    print(f"WARNING: {spd_dir.name}/{mode}/{csv.name} skipped: {e}")
                    continue
                m.update(speed=speed, mode=mode, run=csv.name)
                rows.append(m)
                flag = 'OK  ' if m['success'] else 'FAIL'
                print(f"  {flag} {spd_dir.name}/{mode}/{csv.name}: window {m['window_s']:.1f} s "
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
                    pe.main(csv, out_dir=out_dir / 'per_run' / spd_dir.name / mode / csv.stem,
                            window=window, show=False)

    if not rows:
        print("No runs found. Expected <root>/spd_xx/<mode>/*.csv")
        return

    lead = ['speed', 'mode', 'run', 'success', 'attempts']
    runs = pd.DataFrame(rows)
    runs = runs[lead + [c for c in runs.columns if c not in lead]]
    summary = summarise(runs)
    modes_cmp = compare_modes(runs)

    runs.to_csv(out_dir / 'runs.csv', index=False)
    summary.to_csv(out_dir / 'summary.csv', index=False)
    modes_cmp.to_csv(out_dir / 'mode_comparison.csv', index=False)
    print(f"\nSaved: {out_dir / 'runs.csv'}, summary.csv, mode_comparison.csv")

    if 'held_lat' not in runs or runs['held_lat'].isna().all():
        print("WARNING: no raw YOLO fit in box mode - was yolo_shadow_fit_enabled on?")

    print_summary(summary, modes_cmp)
    plot_clusters(runs, out_dir)
    plot_rmse(summary, out_dir)
    plot_landing(summary, out_dir)


if __name__ == '__main__':
    main()