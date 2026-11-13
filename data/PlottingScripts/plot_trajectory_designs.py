"""
plot_trajectory_designs.py  -  The test trajectories as designed, for the paper
=============================================================================
Draws each AGV trajectory straight from agv_controller.py (no SITL run needed): the
commanded (v, w) is sampled at the node's 20 Hz tick and integrated as an ideal unicycle
from the spawn pose run_batch.sh uses, so the paths are exactly what the car is told to
drive, over as much of the trajectory as you choose to show.

    trajectory_designs.pdf          top-down panels (DESIGN_GRID): one example path per
                                    trajectory, corners coloured by speed, UAV waiting point,
                                    car spawn and cruise line marked
    trajectory_designs_legend.pdf   its legend (LEGEND_SEPARATE, as batch_analysis.py)

Style, figure widths and legend handling come from batch_analysis.py (keep this file in
the same folder), so the figure matches the results figures.

Frame: origin at the UAV waiting point, x along the car's starting heading (world +y),
y to its left. Swept trajectories are drawn at DESIGN_SPEED (their radius / weave amplitude
scales with speed, so give it in the caption); corners has one fixed profile, drawn up to
its first CORNERS_SHOWN corners and coloured by speed, since its speed varies.

Usage:  python plot_trajectory_designs.py [--agv path/to/agv_controller.py] [--out DIR]
"""
import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import batch_analysis as ba
plt = ba.plt

# ── Config ─────────────────────────────────────────────────────────────────

# agv_controller.py (the source copy, not build/ or install/); --agv overrides it.
# None = search next to this script and in the folders above it.
AGV_CONTROLLER = Path('/home/luke/ros2_ws/src/auto_lander/auto_lander/agv_controller.py')
OUT_DIR = None                      # None -> ROOT/batch_results/all_trajectories (batch_analysis)

DESIGN_GRID   = ('straight', 'sin', 'circle', 'corners')   # panels, row by row: the two
                                    # flat paths share the short top row (see ROW_FIT)
# Size. 'single' + 1 column: one IEEE column, panels stacked (3.5 x ~5.8 in, \begin{figure}).
# 'double' + 2 columns: page width 2x2 (7.16 x ~3.3 in, \begin{figure*}). A 2x2 at single
# width does not fit (titles collide, ticks crowd, ~4 in wide).
DESIGN_WIDTH  = 'single'
DESIGN_NCOLS  = 1
DESIGN_SPEED  = 45.0                # km/h, the one example drawn for the swept trajectories
                                    # (state it in the caption: their size scales with speed)
TURN_START_M  = 35.0                # run_batch.sh TURN_START_DISTANCE_M (turn-in abeam the UAV)
SIN_PHASE_DEG = 0.0                 # weave start phase drawn (run_batch.sh staggers it)
# How much to draw, in seconds after the car passes the UAV
SPAN_S = {'straight': 15.0, 'circle': 36.0, 'sin': 18.0}   # sine: 1.5 periods
CORNERS_SHOWN = 2                   # corners: the first N corners (landings happen in this
                                    # stretch), then CORNERS_TAIL_S of the following straight
CORNERS_TAIL_S = 2.0
# Equal x / y scale per panel. The weave is ~16x longer than it is wide (amplitude ~0.73 v,
# wavelength 12 v), so at equal scale it reads as a straight line: drawn stretched, both
# axes still in metres (say so in the caption).
EQUAL_ASPECT = {}                   # every panel to scale; {'sin': False} stretches the weave
# Row heights fitted to the paths at equal scale, so flat rows (straight, sine) take less
# room than tall ones (circle, corners). False: every row PANEL_ASPECT high.
ROW_FIT = True
MIN_PANEL_ASPECT = 0.22             # height / width floor for a fitted row (a straight line)
MAX_PANEL_ASPECT = 0.55             # and ceiling (taller paths get wider x limits instead)
PANEL_ASPECT = 0.55                 # rows not fitted (stretched panels, ROW_FIT False)
PATH_COLOUR  = '0.15'               # constant-speed paths: near-black, so they can't be read
                                    # as a speed on the corners panel's colour scale
SPEED_COLOURED = ('corners',)       # panels coloured by car speed (variable-speed profiles)
CORNERS_CBAR_BOX = [0.06, 0.80, 0.36, 0.045]   # its colour bar, axes fractions (x, y, w, h):
                                               # top left, clear of the first two corners
SHOW_CRUISE_LINE = True             # dotted line where the car reaches cruise speed
CMAP_RANGE = ba.TOPDOWN_CMAP_RANGE  # same speed colours as the batch top-down figure
# Axis labels. The panel frame is local, not the simulator's: origin at the UAV waiting
# point, x along the car's starting heading (world +y), y to its left. Define it in the
# caption rather than calling these world axes.
X_LABEL = 'x (m)'
Y_LABEL = 'y (m)'
DT = 0.05                           # s, the AGV node's tick


# ── Trajectories ───────────────────────────────────────────────────────────

def find_agv(path=None):
    """ Locate agv_controller.py. The source copy (src/) is preferred over the colcon
        build/ and install/ copies, which go stale without a rebuild. """
    if path is not None:
        path = Path(path).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"agv_controller.py not found at {path}")
    here = Path(__file__).resolve().parent
    candidates = [here / 'agv_controller.py']
    for d in [here, *here.parents]:
        candidates += [d / 'auto_lander' / 'auto_lander' / 'agv_controller.py',
                       d / 'src' / 'auto_lander' / 'auto_lander' / 'agv_controller.py']
    for c in candidates:
        if c.exists():
            return c
    src = Path.home() / 'ros2_ws' / 'src'
    found = sorted(p for p in src.rglob('agv_controller.py')
                   if not {'build', 'install', 'log'} & set(p.parts)) if src.exists() else []
    if found:
        return found[0]
    raise FileNotFoundError("agv_controller.py not found; pass --agv path/to/agv_controller.py "
                            "or set AGV_CONTROLLER at the top of this script")


def load_agv(path):
    """ Import agv_controller.py without ROS: rclpy / geometry_msgs are only needed by
        the node class, so stub them when they are not installed. """
    for name in ('rclpy', 'rclpy.node', 'geometry_msgs', 'geometry_msgs.msg'):
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)
    sys.modules['rclpy.node'].__dict__.setdefault('Node', object)
    sys.modules['geometry_msgs.msg'].__dict__.setdefault('Twist', object)
    path = find_agv(path)
    spec = importlib.util.spec_from_file_location('agv_controller', path)
    agv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agv)
    print(f"Trajectories from {path}")
    return agv


def drive(traj, x0, t_max):
    """ Sample the trajectory as the node does (elapsed = running sum of DT) and integrate
        an ideal unicycle exactly over each held tick, from (x0, 0) heading +x.
        Returns t, x, y, v (m/s), w (rad/s). """
    n = int(np.ceil(t_max / DT)) + 1
    t, x, y, v, w = (np.zeros(n) for _ in range(5))
    xi, yi, th, el = x0, 0.0, 0.0, 0.0
    for k in range(n):
        vk, wk = traj.sample(el)
        t[k], x[k], y[k], v[k], w[k] = el, xi, yi, vk, wk
        if abs(wk) > 1e-12:
            xi += vk / wk * (np.sin(th + wk * DT) - np.sin(th))
            yi -= vk / wk * (np.cos(th + wk * DT) - np.cos(th))
        else:
            xi += vk * DT * np.cos(th)
            yi += vk * DT * np.sin(th)
        th += wk * DT
        el += DT
    return t, x, y, v, w


def design_paths(agv, name):
    """ (t, x, y, v km/h) of one trajectory in the UAV frame: DESIGN_SPEED for the swept
        ones, the fixed profile for corners. """
    kmh = agv.KMH_TO_MS
    if name == 'corners':
        sp, (sx, sy, _) = agv.CORNERS_TOP_SPEED_KMH, agv.corners_world_pose()
    else:
        sp, (sx, sy, _) = DESIGN_SPEED, agv.swept_world_poses(DESIGN_SPEED * kmh)
    traj = agv.make_trajectory(name, sp * kmh, d_turn=TURN_START_M,
                               sin_phase_deg=SIN_PHASE_DEG)
    ux, uy = agv.UAV_WAIT_WORLD
    x0 = -np.hypot(ux - sx, uy - sy)                    # spawn distance behind the UAV
    # Swept scripts end with a held command, so drive past their last segment
    t, x, y, v, w = drive(traj, x0, t_max=min(max(traj.duration, 200.0), 400.0) + 5.0)
    if name == 'corners':   # first CORNERS_SHOWN corners, then a bit of straight
        ends = np.flatnonzero(np.diff((np.abs(w) > 1e-3).astype(int)) == -1)
        t_cut = t[ends[min(CORNERS_SHOWN, len(ends)) - 1]] + CORNERS_TAIL_S \
            if len(ends) else t[-1]
    else:
        t_cut = t[np.argmax(x >= 0.0)] + SPAN_S.get(name, 20.0)
    keep = t <= t_cut
    return t[keep], x[keep], y[keep], v[keep] / kmh


# ── Figure ─────────────────────────────────────────────────────────────────

def coloured_path(ax, x, y, v, cmap, norm):
    seg = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
    lc = LineCollection(seg, cmap=cmap, norm=norm, linewidths=ba.lw(1.4),
                        capstyle='round', zorder=3)
    lc.set_array(0.5 * (v[:-1] + v[1:]))
    ax.add_collection(lc)


def plot_designs(agv, out_dir, name='trajectory_designs'):
    panels = [(t, design_paths(agv, t)) for t in DESIGN_GRID]
    n = len(panels)
    ncols = min(DESIGN_NCOLS, n)
    nrows = int(np.ceil(n / ncols))
    width = ba.FIG_WIDTH_DOUBLE if DESIGN_WIDTH == 'double' else ba.FIG_WIDTH_SINGLE
    panel_w = (width - 0.6) / ncols

    def fitted(traj, path):        # height / width the path needs at equal scale
        if not (ROW_FIT and EQUAL_ASPECT.get(traj, True)):
            return PANEL_ASPECT
        x, y = path[1], path[2]
        return float(np.clip(1.15 * np.ptp(y) / (1.05 * np.ptp(x)),
                             MIN_PANEL_ASPECT, MAX_PANEL_ASPECT))

    rows = [max(fitted(*panels[i]) for i in range(r * ncols, min(n, (r + 1) * ncols)))
            for r in range(nrows)]
    fig, axes = ba.new_figure(nrows, ncols, (width, panel_w * sum(rows) + 0.3 * nrows + 0.15),
                              dict(w_pad=0.03, h_pad=0.03, wspace=0.04, hspace=0.04),
                              squeeze=False, gridspec_kw=dict(height_ratios=rows))
    cmap = mcolors.ListedColormap(
        plt.get_cmap(ba.COLOURMAP)(np.linspace(*CMAP_RANGE, 256)))
    cruise_x = -agv.CRUISE_LINE_BEFORE_UAV

    for k, (traj, (t, x, y, v)) in enumerate(panels):
        ax = axes[k // ncols][k % ncols]
        if traj in SPEED_COLOURED:
            # Colour scale spans this profile's own speeds, to the nearest 10 km/h
            norm = mcolors.Normalize(10.0 * np.floor(v[x >= cruise_x].min() / 10.0),
                                     10.0 * np.ceil(v.max() / 10.0))
            coloured_path(ax, x, y, v, cmap, norm)
            cax = ax.inset_axes(CORNERS_CBAR_BOX)
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                              orientation='horizontal')
            cb.set_label(f'Car speed ({ba.SPEED_UNIT})', fontsize=ba.fs(0.85), labelpad=1.5)
            cb.ax.tick_params(labelsize=ba.fs(0.8), width=ba.axlw(), length=1.5, pad=1.0)
            cb.outline.set_linewidth(ba.axlw())
            cb.ax.xaxis.set_major_locator(ba.MultipleLocator(10))
        else:
            ax.plot(x, y, color=PATH_COLOUR, lw=ba.lw(1.4), solid_capstyle='round', zorder=3)
        ax.plot(x[0], y[0], 'o', ms=ba.ms(1.3), mfc='white', mec='0.15', mew=ba.lw(), zorder=5)
        ax.plot(0.0, 0.0, '^', ms=ba.ms(1.8), mfc='black', mec='white', mew=ba.axlw(), zorder=6)
        if SHOW_CRUISE_LINE:
            ax.axvline(cruise_x, color='0.55', ls=':', lw=ba.lw(0.9), zorder=1)
        ax.autoscale_view()
        if EQUAL_ASPECT.get(traj, True):
            ax.set_aspect('equal', adjustable='datalim')
        ax.grid(ba.SHOW_GRID)
        ax.set_title(f"({'abcdefghij'[k]}) {ba.traj_label(traj)}", loc='left',
                     fontweight='bold')
        if k + ncols >= n:
            ax.set_xlabel(X_LABEL)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.supylabel(Y_LABEL, fontsize=plt.rcParams['axes.labelsize'])   # one, shared by every row

    handles = [
        Line2D([], [], ls='none', marker='^', ms=ba.ms(1.8), mfc='black', mec='white',
               mew=ba.axlw(), label='UAV wait pt.'),
        Line2D([], [], ls='none', marker='o', ms=ba.ms(1.3), mfc='white', mec='0.15',
               mew=ba.lw(), label='Car start')]
    if SHOW_CRUISE_LINE:
        handles.append(Line2D([], [], color='0.55', ls=':', lw=ba.lw(0.9),
                              label='Cruise speed met'))
    ba.top_legend(fig, handles)
    ba.finish(fig, out_dir, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--agv', type=Path, default=AGV_CONTROLLER)
    ap.add_argument('--out', type=Path, default=OUT_DIR)
    args = ap.parse_args()
    agv = load_agv(args.agv)
    out = (args.out or ba.ROOT / 'batch_results' / 'all_trajectories').expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(ba.PAPER_RC):
        plot_designs(agv, out)


if __name__ == '__main__':
    main()