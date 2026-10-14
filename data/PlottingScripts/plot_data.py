"""
plot_data.py – Quadcopter flight data visualiser
================================================
Usage:
    python plot_data.py                                  # newest controller_*.csv here
    python plot_data.py path/to/controller_data.csv
    python plot_data.py path/to/controller_data.csv --offset-x 0.1 --offset-y -0.05
    python plot_data.py --offset-y 0.1                   # 0.1 m to the car's left
    python plot_data.py --offset-x -0.2 --offset-frame global
    python plot_data.py --settle-time 5                  # resting error 5 s after the cut
    python plot_data.py --no-show                        # save plots + metrics, don't open windows

Options
-------
--offset-x / --offset-y / --offset-z   Offset (m) added to the TRUE landing-pad (roof)
                                       position to define the landing target.
--offset-frame {car,global}            Frame the X/Y offset is expressed in.
                                       "car" (default): rotated by the true car yaw at
                                       the evaluation time (+x = car forward, +y = car
                                       left). "global": added directly to global X/Y.
                                       (Z offset is always applied in global Z.)
--settle-time                          Seconds after the throttle cut at which the
                                       resting landing error is evaluated (default 3).
--no-show                              Do not open interactive figures.

Outputs
-------
Saves every figure as a PNG into ./plots, writes ./plots/mission_summary.txt,
and opens all figures interactively.

Plots
-----
1-3. Position X / Y / Z vs time
4-6. Velocity Vx / Vy / Vz vs time
7.   Yaw vs time
8.   3-D trajectory
9.   2-D X-Y trajectory
10.  Estimated longitudinal acceleration / 11. turn rate

Everything is plotted in the simulator's (ground-truth) world frame: the quad uses its
TRUE pose, and the landing-pad estimate / raw measurements are the logged relative
values placed on the quad's true position. Trajectories are shifted so the quad starts
at (0, 0).

Mission metrics (printed + saved to mission_summary.txt)
--------------------------------------------------------
* Landing error from target (quad TRUE position minus [true roof position + offset]),
  in the car frame (longitudinal / lateral) and global frame (dX / dY / dZ), at:
    - throttle cut      (first row in LAND_END_STATE)
    - touchdown         (first row after the cut where the TRUE vertical gap to the
                         roof settles to within CONTACT_GAP_TOL of its resting value -
                         the cut can happen up to landing_cut_height_max above the roof)
    - rest              (cut + --settle-time, or the last row if the log ends first)
* Time to acquisition  (ACQ_START_STATE -> ACQ_END_STATE,   default 2100 -> 4000)
* Time to landing      (LAND_START_STATE -> LAND_END_STATE, default 4000 -> 6000)
* Attempts: number of distinct entries into the 4xxx (landing descent) state group,
  with the outcome of each (aborted / landed / ...).

Raw measurement streams
-----------------------
    AprilTag:   landing_pad_april_raw_stamp,     landing_pad_april_rel_{x,y,z}_raw,
                landing_pad_april_glob_yaw_raw
    YOLO car:   landing_pad_yolo_car_raw_stamp,  landing_pad_yolo_car_rel_{x,y,z}_raw
                (fitted position - NaN while the UKF uses the pixel-space box update,
                 see yolo_box_mode)
    (YOLO landing-pad columns from older logs are plotted if present.)

Raw measurements are plotted at their own image timestamp, and placed in the world
frame using the quad's true position AT THAT TIMESTAMP (the CSV's *_glob_*_raw columns
use the quad position at the later logging time, which smears them along the flight
path by the pipeline latency).
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Colours ────────────────────────────────────────────────────────────────
C_QUAD      = "#2196F3"
C_PAD       = "#FF9800"      # filtered landing pad
C_TRUE      = "#4CAF50"
C_APRIL     = "#E91E63"
C_YOLO_LP   = "#9C27B0"
C_YOLO_CAR  = "#00ACC1"
ALPHA_LINE = 0.90
LW = 1.6
OUTLIER_SPEED_THRESH = 20.0

# ── Mission-metric state definitions ───────────────────────────────────────
# Edit these if your state numbering changes.
ACQ_START_STATE  = 2100     # start of "time to acquisition"
ACQ_END_STATE    = 4000     # end of "time to acquisition"
LAND_START_STATE = 4000     # start of "time to landing"
LAND_END_STATE   = 6000     # end of "time to landing" / throttle cut
ATTEMPT_GROUP    = 4        # state group (code // 1000) counted as a landing attempt
LP_ABOVE_GND     = 1.5      # roof (pad) height above the car's odometry origin (m)
CONTACT_GAP_TOL  = 0.05     # m, vertical gap within this of the resting gap = touchdown

# Raw measurement streams: label, colour, CSV column prefix, marker size
RAW_STREAMS = [
    ("AprilTag raw",          C_APRIL,    "landing_pad_april",    18),
    ("YOLO landing-pad raw",  C_YOLO_LP,  "landing_pad_yolo_LP",  22),   # legacy logs
    ("YOLO car fit",          C_YOLO_CAR, "landing_pad_yolo_car", 22),
]

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
    A row is flagged when the implied speed entering that row AND leaving
    that row both exceed `thresh`.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    bad_global = np.zeros(len(x), dtype=bool)

    for _ in range(passes):
        speed = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
        bad = np.zeros(len(x), dtype=bool)
        bad[1:-1] = (speed[:-1] > thresh) & (speed[1:] > thresh)
        bad_global |= bad
        idx = np.arange(len(x))
        good = ~bad_global
        x = np.interp(idx, idx[good], x[good])
        y = np.interp(idx, idx[good], y[good])
    return x, y, bad_global


def raw_events(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    One row per genuinely new raw measurement of a stream (the CSV holds the last
    measurement on every row until the next arrives), with columns:
        t            image timestamp, relative to the start of the log (s)
        x, y, z      world position: logged relative measurement + the quad's TRUE
                     position interpolated to the image timestamp
        yaw          measured yaw (AprilTag only), else absent
    Rows whose position is NaN (e.g. YOLO in pixel-space box mode) are dropped.
    """
    stamp_col = f"{prefix}_raw_stamp"
    rel_cols = [f"{prefix}_rel_{a}_raw" for a in "xyz"]
    if stamp_col not in df.columns or rel_cols[0] not in df.columns:
        return pd.DataFrame(columns=["t", "x", "y", "z"])

    stamp = df[stamp_col].fillna(0)
    new = (stamp > 0) & stamp.ne(stamp.shift())
    ev = df.loc[new].copy()
    ev["t"] = ev[stamp_col] - df["timestamp"].iloc[0]

    for a, rel_col in zip("xyz", rel_cols):
        quad_at_stamp = np.interp(ev["t"], df["t"], df[f"quad_true_{a}"])
        rel = ev[rel_col] if rel_col in ev.columns else np.nan
        ev[a] = rel + quad_at_stamp
    out = ["t", "x", "y", "z"]
    if f"{prefix}_glob_yaw_raw" in ev.columns:
        ev["yaw"] = ev[f"{prefix}_glob_yaw_raw"]
        out.append("yaw")
    return ev.loc[np.isfinite(ev["x"]) & np.isfinite(ev["y"]), out].reset_index(drop=True)


def add_state_transitions(ax, df: pd.DataFrame):
    """Mark controller-state changes on time-series plots."""
    state_col = "controller_state"
    if state_col not in df.columns:
        return

    states = pd.to_numeric(df[state_col], errors="coerce")
    changed = states.notna() & states.ne(states.shift())

    for row_idx in np.flatnonzero(changed.to_numpy()):
        state_code = int(states.iloc[row_idx])
        state_name = STATE_LABELS.get(state_code // 1000, "Controller state")
        transition_t = float(df["t"].iloc[row_idx])
        ax.axvline(transition_t, color="#555555", linewidth=0.9,
                   linestyle=":", alpha=0.65, zorder=0)
        ax.annotate(
            f"{state_name} ({state_code})",
            xy=(transition_t, 0.99),
            xycoords=("data", "axes fraction"),
            xytext=(3, -2),
            textcoords="offset points",
            rotation=90, ha="left", va="top",
            fontsize=7, color="#333333",
            annotation_clip=False,
        )


def style_ax(ax, ylabel: str, title: str, df: pd.DataFrame | None = None):
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if df is not None:
        add_state_transitions(ax, df)
    ax.legend(fontsize=8, framealpha=0.7)


def save(fig, name: str, out_dir: str):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")


def time_series(df, quad_col, pad_col, true_col, ylabel, title, name, out_dir,
                scale=1.0, true_add=0.0, raw=None, raw_field=None):
    """ One quad / UKF pad / true pad time-series figure, plus optional raw streams. """
    t = df["t"]
    fig, ax = plt.subplots(figsize=(10, 4))
    if quad_col is not None:
        ax.plot(t, scale * df[quad_col], color=C_QUAD, lw=LW, alpha=ALPHA_LINE,
                label="Quadcopter")
    ax.plot(t, scale * df[pad_col], color=C_PAD, lw=LW, alpha=ALPHA_LINE,
            label="Landing pad (UKF)", linestyle="--")
    ax.plot(t, scale * (df[true_col] + true_add), color=C_TRUE, lw=LW, alpha=ALPHA_LINE,
            label="Landing pad (true)", linestyle=":")
    for label, colour, ev, size in (raw or []):
        if raw_field in ev.columns and len(ev):
            ax.scatter(ev["t"], scale * ev[raw_field], color=colour, s=size,
                       alpha=0.75, zorder=4, label=label)
    style_ax(ax, ylabel=ylabel, title=title, df=df)
    fig.tight_layout()
    save(fig, name, out_dir)


# ── Loading ────────────────────────────────────────────────────────────────
def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()   # copy() consolidates the ~140 columns before adding more
    # Relative controller-log time
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


# ── Mission metrics ────────────────────────────────────────────────────────
def _first_row_in_state(states: pd.Series, code: int, start: int = 0):
    """Index of the first row (at or after `start`) whose state == code, else None."""
    arr = states.to_numpy()
    hits = np.flatnonzero((arr == code) & (np.arange(len(arr)) >= start))
    return int(hits[0]) if hits.size else None


def _find_attempts(states: pd.Series):
    """
    Find every contiguous run of the landing-attempt state group (4xxx).
    Returns a list of (start_row, end_row_or_None) where end_row is the first
    row AFTER the run (i.e. the row that ended the attempt).
    """
    groups = np.floor(states.to_numpy() / 1000.0)
    attempts = []
    start = None
    for i, grp in enumerate(groups):
        if np.isnan(grp):
            continue
        if grp == ATTEMPT_GROUP:
            if start is None:
                start = i
        elif start is not None:
            attempts.append((start, i))
            start = None
    if start is not None:
        attempts.append((start, None))
    return attempts


def _vertical_gap(df: pd.DataFrame) -> np.ndarray:
    """ TRUE height of the quad's body origin above the roof (m). """
    return (df["quad_true_z"] - (df["landing_pad_glob_true_z"] + LP_ABOVE_GND)).to_numpy()


def _landing_error_lines(df, row, heading, offset, offset_frame):
    """ Landing error (quad TRUE position - target) at one row, as summary lines. """
    yaw = float(df["landing_pad_glob_true_yaw"].iloc[row])
    c, s = np.cos(yaw), np.sin(yaw)
    ox, oy, oz = offset
    if offset_frame == "car":
        ox, oy = c * offset[0] - s * offset[1], s * offset[0] + c * offset[1]

    target = np.array([
        df["landing_pad_glob_true_x"].iloc[row] + ox,
        df["landing_pad_glob_true_y"].iloc[row] + oy,
        df["landing_pad_glob_true_z"].iloc[row] + LP_ABOVE_GND + oz,
    ])
    quad = df[["quad_true_x", "quad_true_y", "quad_true_z"]].iloc[row].to_numpy(float)
    ex, ey, ez = quad - target
    e_long = c * ex + s * ey      # + = quad ahead of target
    e_lat = -s * ex + c * ey      # + = quad left of target

    return [
        f"  {heading}  [t = {df['t'].iloc[row]:.2f} s]",
        f"    car frame:  long {e_long:+7.3f} m   lat {e_lat:+7.3f} m",
        f"    global:     dX {ex:+7.3f} m   dY {ey:+7.3f} m   dZ {ez:+7.3f} m",
        f"    magnitude:  XY {np.hypot(ex, ey):.3f} m   3-D {np.sqrt(ex**2 + ey**2 + ez**2):.3f} m",
    ]


def compute_mission_metrics(df: pd.DataFrame, offset=(0.0, 0.0, 0.0),
                            offset_frame: str = "car", settle_time: float = 3.0):
    """Build the mission-summary text (list of lines) for the log in `df`."""
    L = ["=" * 64, "MISSION METRICS", "=" * 64]

    if "controller_state" not in df.columns:
        L.append("  'controller_state' column not found – cannot compute metrics.")
        return L

    states = pd.to_numeric(df["controller_state"], errors="coerce")
    t = df["t"].to_numpy()

    # ── Landing error ─────────────────────────────────────────────────
    L += ["", "Landing error", "-" * 64]
    L.append(
        f"  Target = true roof position + offset "
        f"({offset[0]:+.3f}, {offset[1]:+.3f}, {offset[2]:+.3f}) m "
        f"[{'car frame: +x forward, +y left' if offset_frame == 'car' else 'global frame'}]"
    )
    L.append("  Error = quad TRUE position - target (car frame: +long = quad ahead, "
             "+lat = quad left). dZ is the body origin's height above the target.")

    needed = ["quad_true_x", "quad_true_y", "quad_true_z", "landing_pad_glob_true_x",
              "landing_pad_glob_true_y", "landing_pad_glob_true_z", "landing_pad_glob_true_yaw"]
    missing = [col for col in needed if col not in df.columns]
    cut_row = _first_row_in_state(states, LAND_END_STATE)

    if missing:
        L.append(f"  Cannot compute: missing column(s) {missing}")
    elif not np.isfinite(df[needed].to_numpy(float)).any():
        L.append("  Cannot compute: no ground truth in this log.")
    elif cut_row is None:
        L.append(f"  State {LAND_END_STATE} never reached – error at the FINAL log row:")
        L += _landing_error_lines(df, len(df) - 1, "End of log", offset, offset_frame)
    else:
        # Rest: settle_time after the cut, or the last row if the log ends first
        rest_row = int(np.searchsorted(t, t[cut_row] + settle_time))
        rest_note = ""
        if rest_row >= len(df):
            rest_row = len(df) - 1
            rest_note = f"  [log ends {t[-1] - t[cut_row]:.2f} s after the cut]"

        # Touchdown: first row after the cut whose true vertical gap is within
        # CONTACT_GAP_TOL of the resting gap (the cut can happen above the roof)
        gap = _vertical_gap(df)
        settled = np.flatnonzero(gap[cut_row:rest_row + 1] <= gap[rest_row] + CONTACT_GAP_TOL)
        contact_row = cut_row + int(settled[0]) if settled.size else rest_row

        L += _landing_error_lines(df, cut_row, f"At throttle cut (state {LAND_END_STATE})",
                                  offset, offset_frame)
        L += _landing_error_lines(df, contact_row,
                                  f"At touchdown ({t[contact_row] - t[cut_row]:.2f} s after cut, "
                                  f"cut height {gap[cut_row] - gap[rest_row]:.2f} m above rest)",
                                  offset, offset_frame)
        L += _landing_error_lines(df, rest_row,
                                  f"At rest ({t[rest_row] - t[cut_row]:.2f} s after cut){rest_note}",
                                  offset, offset_frame)

    # ── Time to acquisition ───────────────────────────────────────────
    L += ["", f"Time to acquisition (state {ACQ_START_STATE} -> {ACQ_END_STATE})", "-" * 64]
    a0 = _first_row_in_state(states, ACQ_START_STATE)
    a1 = (_first_row_in_state(states, ACQ_END_STATE, start=a0)
          if a0 is not None else None)
    if a0 is None:
        L.append(f"  n/a – state {ACQ_START_STATE} never reached.")
    elif a1 is None:
        L.append(f"  n/a – state {ACQ_START_STATE} reached at t = {t[a0]:.2f} s, "
                 f"but state {ACQ_END_STATE} never reached afterwards.")
    else:
        L.append(f"  {t[a1] - t[a0]:.2f} s   (t = {t[a0]:.2f} s -> {t[a1]:.2f} s)")

    # ── Attempts + time to landing ────────────────────────────────────
    attempts = _find_attempts(states)

    L += ["", f"Time to landing (state {LAND_START_STATE} -> {LAND_END_STATE})", "-" * 64]
    if cut_row is None:
        L.append(f"  n/a – state {LAND_END_STATE} never reached.")
    else:
        # Start of the attempt that actually ended in the throttle cut.
        l0 = None
        for s, _e in attempts:
            if s <= cut_row:
                l0 = s
        if l0 is None:
            l0 = _first_row_in_state(states, LAND_START_STATE)
        if l0 is None:
            L.append(f"  n/a – state {LAND_START_STATE} never reached before "
                     f"landing confirmed.")
        else:
            L.append(f"  {t[cut_row] - t[l0]:.2f} s   (t = {t[l0]:.2f} s -> {t[cut_row]:.2f} s)")
            if len(attempts) > 1:
                first_start = attempts[0][0]
                L.append(
                    f"  (successful attempt only; from FIRST descent start "
                    f"incl. aborted attempts: {t[cut_row] - t[first_start]:.2f} s)"
                )

    L += ["", "Attempts", "-" * 64]
    L.append(f"  Landing attempts (entries into {ATTEMPT_GROUP}xxx): {len(attempts)}")
    n_aborted = 0
    for k, (s, e) in enumerate(attempts, start=1):
        if e is None:
            outcome = "still in descent at end of log"
            t_end = t[-1]
        else:
            code = int(states.iloc[e])
            t_end = t[e]
            if code // 1000 == 5:
                outcome = f"aborted (state {code})"
                n_aborted += 1
            elif code // 1000 == 6:
                outcome = f"landing confirmed (state {code})"
            else:
                outcome = f"ended (state {code})"
        L.append(f"    #{k}: t = {t[s]:.2f} s -> {t_end:.2f} s "
                 f"({t_end - t[s]:.2f} s)  {outcome}")
    L.append(f"  Aborted attempts: {n_aborted}")
    L.append("=" * 64)
    return L


def print_mission_metrics(df, out_dir, offset, offset_frame, settle_time):
    lines = compute_mission_metrics(df, offset, offset_frame, settle_time)
    print()
    for line in lines:
        print(line)
    print()
    path = os.path.join(out_dir, "mission_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved → {path}")


# ── Time-series plots ──────────────────────────────────────────────────────
def plot_position(df, raw, out_dir):
    for a in "xyz":
        time_series(
            df, f"quad_true_{a}", f"landing_pad_glob_{a}", f"landing_pad_glob_true_{a}",
            ylabel=f"Position {a.upper()} (m)", title=f"Position {a.upper()} vs Time",
            name=f"position_{a}.png", out_dir=out_dir,
            true_add=LP_ABOVE_GND if a == "z" else 0.0,   # pad sits on the roof
            raw=raw, raw_field=a,
        )
    print("  Position plots done.")


def plot_velocity(df, out_dir):
    for a in "xyz":
        time_series(
            df, f"quad_true_v{a}", f"landing_pad_glob_v{a}", f"landing_pad_glob_true_v{a}",
            ylabel=f"Velocity V{a} (m/s)", title=f"Velocity V{a.upper()} vs Time",
            name=f"velocity_v{a}.png", out_dir=out_dir,
        )
    print("  Velocity plots done.")


def plot_yaw(df, raw, out_dir):
    time_series(
        df, "quad_true_yaw", "landing_pad_glob_yaw", "landing_pad_glob_true_yaw",
        ylabel="Yaw (°)", title="Yaw vs Time", name="yaw.png", out_dir=out_dir,
        scale=np.degrees(1.0), raw=raw, raw_field="yaw",   # only AprilTag measures yaw
    )
    print("  Yaw plot done.")


def plot_ukf_only(df, col, scale, ylabel, title, name, out_dir):
    """ Single-series UKF state plot (acceleration, turn rate). """
    if col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["t"], scale * df[col], color=C_PAD, lw=LW, alpha=ALPHA_LINE,
            label="Landing pad (UKF)")
    style_ax(ax, ylabel=ylabel, title=title, df=df)
    fig.tight_layout()
    save(fig, name, out_dir)
    print(f"  {title} done.")


# ── Trajectories ───────────────────────────────────────────────────────────
def plot_trajectories(df, raw, out_dir):
    """ 3-D and top-down trajectories, all in the true world frame, shifted so the
        quad starts at the origin. """
    origin = df[["quad_true_x", "quad_true_y"]].iloc[0].to_numpy(float)
    # copy=True: pandas >= 3 returns read-only views, and these are shifted in place below
    quad = df[["quad_true_x", "quad_true_y", "quad_true_z"]].to_numpy(float, copy=True)
    pad = df[["landing_pad_glob_x", "landing_pad_glob_y", "landing_pad_glob_z"]].to_numpy(float, copy=True)
    true = df[["landing_pad_glob_true_x", "landing_pad_glob_true_y",
               "landing_pad_glob_true_z"]].to_numpy(float, copy=True)
    true[:, 2] += LP_ABOVE_GND
    for arr in (quad, pad, true):
        arr[:, :2] -= origin

    lines = [
        (quad, dict(color=C_QUAD, label="Quadcopter")),
        (pad,  dict(color=C_PAD, label="Landing pad (UKF)", linestyle="--")),
        (true, dict(color=C_TRUE, label="Landing pad (true)", linestyle=":")),
    ]

    # 3-D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for arr, kw in lines:
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], lw=1.5, alpha=0.9, **kw)
    for label, colour, ev, size in raw:
        if len(ev):
            ax.scatter(ev["x"] - origin[0], ev["y"] - origin[1], ev["z"],
                       color=colour, s=size * 0.6, alpha=0.6, label=label)
    ax.scatter(*quad[0], color=C_QUAD, s=60, marker="o", label="Quad start")
    ax.scatter(*quad[-1], color=C_QUAD, s=80, marker="*", label="Quad end")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3-D Trajectory", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, framealpha=0.7, loc="upper left")
    all_vals = np.concatenate([arr[np.isfinite(arr)] for arr, _ in lines])
    mid = (all_vals.max() + all_vals.min()) / 2
    half = (all_vals.max() - all_vals.min()) / 2
    ax.set_xlim(mid - half, mid + half)
    ax.set_ylim(mid - half, mid + half)
    ax.set_zlim(mid - half, mid + half)
    fig.tight_layout()
    save(fig, "trajectory_3d.png", out_dir)

    # Top-down
    fig, ax = plt.subplots(figsize=(8, 8))
    for arr, kw in lines:
        ax.plot(arr[:, 0], arr[:, 1], lw=LW, alpha=ALPHA_LINE, **kw)
    for label, colour, ev, size in raw:
        if len(ev):
            ax.scatter(ev["x"] - origin[0], ev["y"] - origin[1], color=colour, s=size,
                       alpha=0.7, label=label, zorder=4)
    ax.scatter(*quad[0, :2], color=C_QUAD, marker="o", s=70, label="Quad start", zorder=5)
    ax.scatter(*quad[-1, :2], color=C_QUAD, marker="*", s=100, label="Quad end", zorder=5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Top-Down Trajectory")
    ax.grid(True, alpha=0.5)
    ax.legend(framealpha=0.7)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save(fig, "trajectory_2d.png", out_dir)
    print("  Trajectory plots done.")


# ── Main ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Quadcopter flight data visualiser + mission metrics."
    )
    p.add_argument("csv", nargs="?", default=None,
                   help="Path to controller CSV (default: newest controller_*.csv "
                        "next to this script).")
    p.add_argument("--offset-x", type=float, default=0.0,
                   help="X offset (m) added to true roof position to form the landing target.")
    p.add_argument("--offset-y", type=float, default=0.0,
                   help="Y offset (m) added to true roof position to form the landing target.")
    p.add_argument("--offset-z", type=float, default=0.0,
                   help="Z offset (m) added to true roof position to form the landing target.")
    p.add_argument("--offset-frame", choices=["car", "global"], default="car",
                   help="Frame the X/Y offset is expressed in (default: car). "
                        "'car' rotates the X/Y offset by the true car/pad yaw "
                        "(+x forward, +y left); 'global' adds it directly.")
    p.add_argument("--settle-time", type=float, default=3.0,
                   help="Seconds after the throttle cut to evaluate the resting error "
                        "(default: 3).")
    p.add_argument("--no-show", action="store_true",
                   help="Save plots/metrics but don't open interactive windows.")
    return p.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.csv is not None:
        csv_path = args.csv
    else:
        csvs = sorted(glob.glob(os.path.join(script_dir, "controller_*.csv")))
        csv_path = csvs[-1] if csvs else ""
    if not os.path.isfile(csv_path):
        print(f"ERROR: could not find CSV at '{csv_path}'")
        print("Usage: python plot_data.py [path/to/data.csv] "
              "[--offset-x X] [--offset-y Y] [--offset-z Z] "
              "[--offset-frame car|global] [--settle-time S] [--no-show]")
        sys.exit(1)
    out_dir = os.path.join(script_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoading data from: {csv_path}")
    df = load(csv_path)
    print(f"  {len(df)} rows  |  duration {df['t'].iloc[-1]:.1f} s\n")

    # ── Clean true X/Y ────────────────────────────────────────────────
    true_x, true_y, outlier_mask = clean_true_position(
        df["landing_pad_glob_true_x"].values,
        df["landing_pad_glob_true_y"].values,
        df["timestamp"].values,
    )
    df["landing_pad_glob_true_x"] = true_x
    df["landing_pad_glob_true_y"] = true_y
    if outlier_mask.sum():
        print(f"Removed {outlier_mask.sum()} true-position outlier row(s) at t = "
              f"{np.round(df['t'].values[outlier_mask], 2).tolist()}")

    # ── Raw measurement streams (at their own timestamps) ─────────────
    raw = []
    print("\nRaw measurement counts:")
    for label, colour, prefix, size in RAW_STREAMS:
        stamp_col = f"{prefix}_raw_stamp"
        if stamp_col not in df.columns:
            continue
        stamp = df[stamp_col].fillna(0)
        new = (stamp > 0) & stamp.ne(stamp.shift())
        ev = raw_events(df, prefix)
        raw.append((label, colour, ev, size))
        line = f"  {label:22s}: {int(new.sum())} detections"
        if prefix == "landing_pad_yolo_car" and "yolo_box_mode" in df.columns:
            n_box = int((new & (df["yolo_box_mode"] == 1)).sum())
            line += f"  ({n_box} pixel-space box updates, {int(new.sum()) - n_box} fallback fits)"
        print(line)

    # ── Mission metrics (landing error, acquisition/landing time, attempts)
    print_mission_metrics(
        df, out_dir,
        offset=(args.offset_x, args.offset_y, args.offset_z),
        offset_frame=args.offset_frame,
        settle_time=args.settle_time,
    )

    print("\nGenerating position plots …")
    plot_position(df, raw, out_dir)
    print("\nGenerating velocity plots …")
    plot_velocity(df, out_dir)
    print("\nGenerating yaw plot …")
    plot_yaw(df, raw, out_dir)
    print("\nGenerating trajectory plots …")
    plot_trajectories(df, raw, out_dir)
    print("\nGenerating UKF state plots …")
    plot_ukf_only(df, "landing_pad_glob_a", 1.0, "Longitudinal Acceleration (m/s²)",
                  "Estimated Longitudinal Acceleration vs Time", "acceleration.png", out_dir)
    plot_ukf_only(df, "landing_pad_glob_yaw_rate", np.degrees(1.0), "Turn Rate (°/s)",
                  "Estimated Turn Rate vs Time", "turn_rate.png", out_dir)
    if args.no_show:
        print("\nAll done!")
    else:
        print("\nAll done! Opening figures …")
        plt.show()


if __name__ == "__main__":
    main()