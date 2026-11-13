"""
plot_error_compare.py  –  3-D estimation error over time for two (or more) runs
================================================================================
Shows where in a run the estimation error comes from: each run is drawn on its own
panel against time to touchdown, so the descents line up at t = 0 whatever the chase
length. Uses plot_errors.py (same folder) for the truth cleaning, pad offset and the
batch_analysis paper style, so the numbers match the other figures.

    python plot_error_compare.py                       # the runs listed in RUNS below
    python plot_error_compare.py runA.csv runB.csv --labels "20 km/h" "55 km/h"

Relative CSV paths in RUNS are taken from this script's folder.

Per run:
    filter   3-D error of the UKF estimate at the tick it is valid at (est_time)
    raw      3-D error of the YOLO car fit held to the row time (the live competitor),
             drawn only while it is fresh
Hatched: the descent (first 4xxx to touchdown). Shaded: spans with no fresh detection
(latest frame older than MAX_MEAS_AGE_S), where the filter is coasting. RMSE per phase is
printed, over the same rows batch_analysis scores (pad in view, fresh frame, raw fit
finite).

Outputs (./plots next to this script): error_compare.pdf/.png and error_compare_legend.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

import plot_errors as pe

# ── Config ─────────────────────────────────────────────────────────────────
# Runs to compare when no CSVs are given on the command line: (CSV, panel label), one
# panel each, top to bottom. Paths relative to this script's folder, or absolute.
RUNS = [
    ('controller_20261021_062021.csv', '20 km/h'),
    ('controller_20261107_043123.csv', '55 km/h'),
]

START_STATE    = 3000    # window start: first entry into this state (3000 = chase)
DESCENT_STATE  = 4000    # descent starts at the first entry into this group
LAND_STATE     = 6000    # touchdown (window end)
MAX_MEAS_AGE_S = 0.25    # same as batch_analysis: older latest frame = no detection
SHOW_RAW       = True    # also draw the raw held measurement
PANEL_HEIGHT   = 1.45    # in, per run (single-column width from plot_errors)
PANEL_TITLE    = '3-D error over time at {label}'   # {label} = the run's label in RUNS

DESCENT_C = '0.72'       # descent: hatched in this colour
HATCH_LW  = 0.4          # pt, hatch line width (matplotlib default 1.0 is heavy)
DROPOUT_C = '0.88'       # no detection: solid fill


def first_time(t, mask):
    idx = np.flatnonzero(mask)
    return float(t[idx[0]]) if idx.size else np.nan


def run_errors(csv_path):
    """ 3-D filter and raw errors for one run, with its phase times. """
    df = pd.read_csv(csv_path)
    t0 = float(df['timestamp'].iloc[0])
    t = df['timestamp'].to_numpy(float) - t0
    state = pd.to_numeric(df['controller_state'], errors='coerce').to_numpy()
    t_start = first_time(t, state >= START_STATE)
    t_desc = first_time(t, (state // 1000) == DESCENT_STATE // 1000)
    t_land = first_time(t, (state // 1000) == LAND_STATE // 1000)
    if not np.isfinite(t_start) or not np.isfinite(t_land):
        raise SystemExit(f"{csv_path}: no {START_STATE} or no {LAND_STATE} state, can't align")

    # Truth exactly as plot_errors / batch_analysis: cleaned from the window start, pad offset
    sl = t >= t_start
    ts = t[sl]
    tx, ty, _ = pe.clean_true_position(df['landing_pad_rel_true_x'].to_numpy()[sl],
                                       df['landing_pad_rel_true_y'].to_numpy()[sl], ts)
    yaw, _ = pe.clean_true_yaw(df['landing_pad_glob_true_yaw'].to_numpy()[sl], ts)
    tz = df['landing_pad_rel_true_z'].to_numpy(float)[sl] + pe.TRUE_OFFSET_Z_BODY
    c, s = np.cos(yaw), np.sin(yaw)
    tx = tx + pe.TRUE_OFFSET_X_BODY * c - pe.TRUE_OFFSET_Y_BODY * s
    ty = ty + pe.TRUE_OFFSET_X_BODY * s + pe.TRUE_OFFSET_Y_BODY * c
    truth = interp1d(ts, np.vstack([tx, ty, tz]), axis=1, bounds_error=False,
                     fill_value='extrapolate')

    def err3(tq, x, y, z):
        X, Y, Z = truth(tq)
        return np.sqrt((x - X) ** 2 + (y - Y) ** 2 + (z - Z) ** 2)

    est = pd.to_numeric(df.get('est_time', pd.Series(np.zeros(len(t)))),
                        errors='coerce').to_numpy(float)
    est_t = np.where(est > 0, est - t0, t)
    e_filt = err3(est_t, df['landing_pad_rel_x'].to_numpy(float),
                  df['landing_pad_rel_y'].to_numpy(float), df['landing_pad_rel_z'].to_numpy(float))
    pre = 'landing_pad_yolo_car'
    e_raw = err3(t, df[f'{pre}_rel_x_raw'].to_numpy(float), df[f'{pre}_rel_y_raw'].to_numpy(float),
                 df[f'{pre}_rel_z_raw'].to_numpy(float))

    stamp = df[f'{pre}_raw_stamp'].to_numpy(float)
    age = t - np.where(stamp > 0, stamp - t0, np.nan)
    fresh = age <= MAX_MEAS_AGE_S
    found = (pd.to_numeric(df['pad_found'], errors='coerce').to_numpy() == 1
             if 'pad_found' in df.columns else np.ones(len(t), bool))
    in_win = (t >= t_start) & (t <= t_land)
    scored = in_win & fresh & found & np.isfinite(e_raw)

    return dict(name=Path(csv_path).name, t=t, tt=t - t_land, in_win=in_win, fresh=fresh,
                scored=scored, e_filt=e_filt, e_raw=e_raw,
                t_start=t_start, t_desc=t_desc, t_land=t_land)


def rmse(v, m):
    return float(np.sqrt(np.nanmean(v[m] ** 2))) if m.any() else np.nan


def phase_stats(r):
    """ RMSE over the chase (start -> descent), the descent and the whole window. """
    t, m = r['t'], r['scored']
    desc = np.isfinite(r['t_desc'])
    chase = m & (t < r['t_desc']) if desc else m
    descent = m & (t >= r['t_desc']) if desc else np.zeros_like(m)
    return {k: dict(filt=rmse(r['e_filt'], mm), raw=rmse(r['e_raw'], mm), s=float(mm.sum()) * 0.05)
            for k, mm in (('chase', chase), ('descent', descent), ('all', m))}


def spans(t, mask):
    """ (t_begin, t_end) of each run of True in mask. """
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(t[a], t[min(b, len(t) - 1)]) for a, b in zip(e[::2], e[1::2])]


def plot(runs, labels, out_dir):
    rc = dict(pe.PAPER_RC) if pe.PAPER_STYLE else {}
    rc['hatch.linewidth'] = HATCH_LW
    with plt.rc_context(rc):
        fig, axes = pe.new_figure(len(runs), 1,
                                  (pe.FIG_WIDTH_SINGLE, PANEL_HEIGHT * len(runs) + 0.35),
                                  (10, 3.0 * len(runs)), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        y_max = max(np.nanmax(r['e_filt'][r['in_win']]) for r in runs)

        for k, (ax, r, label) in enumerate(zip(axes, runs, labels)):
            w, tt = r['in_win'], r['tt']
            # Descent and no-detection shading
            if np.isfinite(r['t_desc']):
                ax.axvspan(r['t_desc'] - r['t_land'], 0.0, facecolor='none',
                           edgecolor=DESCENT_C, hatch='////', lw=0, zorder=0)
            for a, b in spans(tt, w & ~r['fresh']):
                ax.axvspan(a, b, color=DROPOUT_C, lw=0, zorder=0.5)

            if SHOW_RAW:
                raw = np.where(w & r['fresh'], r['e_raw'], np.nan)
                ax.plot(tt, raw, color=pe.YOLO_CAR_C, lw=pe.LW(0.8), ls=':', zorder=2)
            ax.plot(tt[w], r['e_filt'][w], color=pe.FILT_C, lw=pe.LW(1.1), zorder=3)

            pe.panel_title(ax, 'abcdefgh'[k], PANEL_TITLE.format(label=label))
            pe.grid(ax)

        axes[0].set_ylim(0.0, 1.1 * y_max)
        axes[0].set_xlim(min(r['tt'][r['in_win']][0] for r in runs), 0.0)
        axes[-1].set_xlabel('Time to touchdown (s)')
        if pe.PAPER_STYLE and len(runs) > 1:
            fig.supylabel('3-D error (m)', fontsize=plt.rcParams['axes.labelsize'])
        else:
            for ax in axes:
                ax.set_ylabel('3-D error (m)')

        handles = [Line2D([], [], color=pe.FILT_C, lw=pe.LW(1.1), label='UKF')]
        if SHOW_RAW:
            handles.append(Line2D([], [], color=pe.YOLO_CAR_C, lw=pe.LW(0.8), ls=':',
                                  label='Raw'))
        handles += [Patch(facecolor='white', edgecolor=DESCENT_C, hatch='////',
                          label='Descent'),
                    Patch(color=DROPOUT_C, label='No detection')]

        pe.save_figure(fig, out_dir, 'error_compare', legacy_rect=[0, 0, 1, 1])
        if pe.PAPER_STYLE:
            pe.save_legend(handles, [h.get_label() for h in handles], out_dir,
                           'error_compare', pe.FIG_WIDTH_SINGLE)
        else:
            pe.legend(axes[0], handles=handles, loc='upper left')
            pe.save_figure(fig, out_dir, 'error_compare')
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('csvs', nargs='*',
                   help='controller CSVs, one panel each, top to bottom (default: RUNS)')
    p.add_argument('--labels', nargs='+', help='panel labels (default: file names)')
    p.add_argument('--out', default=None, help='output folder (default: ./plots)')
    args = p.parse_args()

    script_dir = Path(__file__).parent
    if args.csvs:
        csvs = [Path(c) for c in args.csvs]
        labels = args.labels or [c.name for c in csvs]
    else:
        csvs = [Path(c) if Path(c).is_absolute() else script_dir / c for c, _ in RUNS]
        labels = args.labels or [label for _, label in RUNS]
    if len(labels) != len(csvs):
        raise SystemExit('--labels needs one label per CSV')
    missing = [str(c) for c in csvs if not c.exists()]
    if missing:
        raise SystemExit('CSV not found: ' + ', '.join(missing))
    out_dir = Path(args.out) if args.out else script_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [run_errors(c) for c in csvs]
    print(f"{'run':<28s} {'phase':<8s} {'s':>5s} {'filter':>8s} {'raw':>8s} {'reduction':>10s}")
    for r, label in zip(runs, labels):
        for phase, v in phase_stats(r).items():
            red = 100 * (1 - v['filt'] / v['raw']) if v['raw'] else np.nan
            print(f"{label:<28s} {phase:<8s} {v['s']:5.1f} {v['filt']:8.3f} {v['raw']:8.3f} {red:9.0f}%")
    plot(runs, labels, out_dir)


if __name__ == '__main__':
    main()