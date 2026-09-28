"""
plot_data.py – Quadcopter flight data visualiser
================================================

Usage:
    python plot_data.py
    python plot_data.py path/to/controller_data.csv

Outputs
-------
Saves every figure as a PNG into ./plots and opens all figures interactively.

Plots
-----
1. Position X vs time
2. Position Y vs time
3. Position Z vs time
4. Velocity Vx vs time
5. Velocity Vy vs time
6. Velocity Vz vs time
7. Yaw vs time
8. 3-D trajectory
9. 2-D X-Y trajectory

Raw measurement streams
-----------------------
The current CSV contains three raw measurement streams:

    AprilTag:
        landing_pad_april_raw_stamp
        landing_pad_april_rel_x_raw
        landing_pad_april_rel_y_raw
        landing_pad_glob_x_raw
        landing_pad_glob_y_raw

    YOLO landing pad:
        landing_pad_yolo_LP_raw_stamp
        landing_pad_yolo_LP_rel_x_raw
        landing_pad_yolo_LP_rel_y_raw
        landing_pad_yolo_LP_glob_x_raw
        landing_pad_yolo_LP_glob_y_raw

    YOLO car:
        landing_pad_yolo_car_raw_stamp
        landing_pad_yolo_car_rel_x_raw
        landing_pad_yolo_car_rel_y_raw
        landing_pad_yolo_car_glob_x_raw
        landing_pad_yolo_car_glob_y_raw

Raw measurements are plotted using the actual measurement timestamp rather
than the controller CSV row timestamp.

Note
----
YOLO raw measurements currently contain only X/Y in the CSV, so they are
included in the 2-D trajectory and X/Y time plots, but not the 3-D plot.
"""


import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)

        speed = np.sqrt((dx / dt) ** 2 + (dy / dt) ** 2)

        bad = np.zeros(len(x), dtype=bool)

        for i in range(1, len(x) - 1):
            if speed[i - 1] > thresh and speed[i] > thresh:
                bad[i] = True

        bad_global |= bad

        idx = np.arange(len(x))
        good = ~bad_global

        x = np.interp(idx, idx[good], x[good])
        y = np.interp(idx, idx[good], y[good])

    return x, y, bad_global


def get_measurement_events(
    df: pd.DataFrame,
    stamp_col: str,
    x_col: str,
    y_col: str,
):
    """
    Extract one row per genuinely new raw measurement.

    The raw measurement is held constant in the controller CSV between
    detections, so simply plotting every CSV row would duplicate the same
    measurement many times.

    Returns a DataFrame containing:
        measurement_t
        x
        y
    """

    required = [stamp_col, x_col, y_col]

    if not all(col in df.columns for col in required):
        return pd.DataFrame(columns=["measurement_t", "x", "y"])

    mask = df[stamp_col].fillna(0) > 0

    events = df.loc[mask, [stamp_col, x_col, y_col]].copy()

    if events.empty:
        return pd.DataFrame(columns=["measurement_t", "x", "y"])

    # Keep only the first controller row corresponding to each raw measurement.
    events = events.loc[events[stamp_col].ne(events[stamp_col].shift())].copy()

    log_start_stamp = df["timestamp"].iloc[0]

    events["measurement_t"] = events[stamp_col] - log_start_stamp
    events["x"] = events[x_col]
    events["y"] = events[y_col]

    return events[["measurement_t", "x", "y"]].reset_index(drop=True)


def style_ax(ax, ylabel: str, title: str):
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name: str, out_dir: str):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")


# ── Loading ────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Relative controller-log time
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]

    return df


# ── Position plots ─────────────────────────────────────────────────────────

def plot_position(df: pd.DataFrame, out_dir: str):
    t = df["t"]

    # Raw measurement events
    april = get_measurement_events(
        df,
        "landing_pad_april_raw_stamp",
        "landing_pad_april_rel_x_raw",
        "landing_pad_april_rel_y_raw",
    )

    yolo_lp = get_measurement_events(
        df,
        "landing_pad_yolo_LP_raw_stamp",
        "landing_pad_yolo_LP_rel_x_raw",
        "landing_pad_yolo_LP_rel_y_raw",
    )

    yolo_car = get_measurement_events(
        df,
        "landing_pad_yolo_car_raw_stamp",
        "landing_pad_yolo_car_rel_x_raw",
        "landing_pad_yolo_car_rel_y_raw",
    )

    # X and Y have all three raw streams.
    axes_xy = [
        (
            "x",
            "quad_true_x",
            "landing_pad_glob_x",
            "landing_pad_glob_true_x",
            "landing_pad_april_glob_x_raw",
            "landing_pad_yolo_LP_glob_x_raw",
            "landing_pad_yolo_car_glob_x_raw",
        ),
        (
            "y",
            "quad_true_y",
            "landing_pad_glob_y",
            "landing_pad_glob_true_y",
            "landing_pad_april_glob_y_raw",
            "landing_pad_yolo_LP_glob_y_raw",
            "landing_pad_yolo_car_glob_y_raw",
        ),
    ]

    for (
        axis,
        q_col,
        p_col,
        true_col,
        april_col,
        yolo_lp_col,
        yolo_car_col,
    ) in axes_xy:

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            t,
            df[q_col],
            color=C_QUAD,
            lw=LW,
            alpha=ALPHA_LINE,
            label="Quadcopter",
        )

        ax.plot(
            t,
            df[p_col],
            color=C_PAD,
            lw=LW,
            alpha=ALPHA_LINE,
            label="Landing pad (UKF)",
            linestyle="--",
        )

        ax.plot(
            t,
            df[true_col],
            color=C_TRUE,
            lw=LW,
            alpha=ALPHA_LINE,
            label="Landing pad (true)",
            linestyle=":",
        )

        if april_col in df.columns:
            mask = df["landing_pad_april_raw_stamp"].fillna(0) > 0
            mask &= df["landing_pad_april_raw_stamp"].ne(
                df["landing_pad_april_raw_stamp"].shift()
            )

            ax.scatter(
                df.loc[mask, "t"],
                df.loc[mask, april_col],
                color=C_APRIL,
                s=18,
                alpha=0.75,
                zorder=4,
                label="AprilTag raw",
            )

        if yolo_lp_col in df.columns:
            mask = df["landing_pad_yolo_LP_raw_stamp"].fillna(0) > 0
            mask &= df["landing_pad_yolo_LP_raw_stamp"].ne(
                df["landing_pad_yolo_LP_raw_stamp"].shift()
            )

            ax.scatter(
                df.loc[mask, "t"],
                df.loc[mask, yolo_lp_col],
                color=C_YOLO_LP,
                s=22,
                alpha=0.75,
                zorder=4,
                label="YOLO landing-pad raw",
            )

        if yolo_car_col in df.columns:
            mask = df["landing_pad_yolo_car_raw_stamp"].fillna(0) > 0
            mask &= df["landing_pad_yolo_car_raw_stamp"].ne(
                df["landing_pad_yolo_car_raw_stamp"].shift()
            )

            ax.scatter(
                df.loc[mask, "t"],
                df.loc[mask, yolo_car_col],
                color=C_YOLO_CAR,
                s=22,
                alpha=0.75,
                zorder=4,
                label="YOLO car raw",
            )

        style_ax(
            ax,
            ylabel=f"Position {axis.upper()} (m)",
            title=f"Position {axis.upper()} vs Time",
        )

        fig.tight_layout()
        save(fig, f"position_{axis}.png", out_dir)

    # Z has the same raw measurement streams as X/Y.
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        t,
        df["quad_z"],
        color=C_QUAD,
        lw=LW,
        alpha=ALPHA_LINE,
        label="Quadcopter",
    )

    ax.plot(
        t,
        df["landing_pad_glob_z"],
        color=C_PAD,
        lw=LW,
        alpha=ALPHA_LINE,
        label="Landing pad (UKF)",
        linestyle="--",
    )

    ax.plot(
        t,
        df["landing_pad_glob_true_z"],
        color=C_TRUE,
        lw=LW,
        alpha=ALPHA_LINE,
        label="Landing pad (true)",
        linestyle=":",
    )

    if {
        "landing_pad_april_raw_stamp",
        "landing_pad_april_glob_z_raw",
    }.issubset(df.columns):

        mask = df["landing_pad_april_raw_stamp"].fillna(0) > 0
        mask &= df["landing_pad_april_raw_stamp"].ne(
            df["landing_pad_april_raw_stamp"].shift()
        )

        ax.scatter(
            df.loc[mask, "t"],
            df.loc[mask, "landing_pad_april_glob_z_raw"],
            color=C_APRIL,
            s=18,
            alpha=0.75,
            zorder=4,
            label="AprilTag raw",
        )

    if {
        "landing_pad_yolo_LP_raw_stamp",
        "landing_pad_yolo_LP_glob_z_raw",
    }.issubset(df.columns):

        mask = df["landing_pad_yolo_LP_raw_stamp"].fillna(0) > 0
        mask &= df["landing_pad_yolo_LP_raw_stamp"].ne(
            df["landing_pad_yolo_LP_raw_stamp"].shift()
        )

        ax.scatter(
            df.loc[mask, "t"],
            df.loc[mask, "landing_pad_yolo_LP_glob_z_raw"],
            color=C_YOLO_LP,
            s=22,
            alpha=0.75,
            zorder=4,
            label="YOLO landing-pad raw",
        )

    if {
        "landing_pad_yolo_car_raw_stamp",
        "landing_pad_yolo_car_glob_z_raw",
    }.issubset(df.columns):

        mask = df["landing_pad_yolo_car_raw_stamp"].fillna(0) > 0
        mask &= df["landing_pad_yolo_car_raw_stamp"].ne(
            df["landing_pad_yolo_car_raw_stamp"].shift()
        )

        ax.scatter(
            df.loc[mask, "t"],
            df.loc[mask, "landing_pad_yolo_car_glob_z_raw"],
            color=C_YOLO_CAR,
            s=22,
            alpha=0.75,
            zorder=4,
            label="YOLO car raw",
        )

    style_ax(
        ax,
        ylabel="Position Z (m)",
        title="Position Z vs Time",
    )

    fig.tight_layout()
    save(fig, "position_z.png", out_dir)

    print("  Position plots done.")


# ── Velocity plots ─────────────────────────────────────────────────────────

def plot_velocity(df: pd.DataFrame, out_dir: str):
    axes_info = [
        ("x", "quad_vx", "landing_pad_glob_vx", "landing_pad_glob_true_vx"),
        ("y", "quad_vy", "landing_pad_glob_vy", "landing_pad_glob_true_vy"),
        ("z", "quad_vz", "landing_pad_glob_vz", "landing_pad_glob_true_vz"),
    ]

    t = df["t"]

    for axis, q_col, p_col, tp_col in axes_info:

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            t,
            df[q_col],
            color=C_QUAD,
            lw=LW,
            alpha=ALPHA_LINE,
            label="Quadcopter",
        )

        ax.plot(
            t,
            df[p_col],
            color=C_PAD,
            lw=LW,
            alpha=ALPHA_LINE,
            label="Landing pad (UKF)",
            linestyle="--",
        )

        ax.plot(
            t,
            df[tp_col],
            color=C_TRUE,
            lw=LW,
            alpha=ALPHA_LINE,
            label="Landing pad (true)",
            linestyle=":",
        )

        style_ax(
            ax,
            ylabel=f"Velocity V{axis} (m/s)",
            title=f"Velocity V{axis.upper()} vs Time",
        )

        fig.tight_layout()
        save(fig, f"velocity_v{axis}.png", out_dir)

    print("  Velocity plots done.")


# ── Yaw plot ───────────────────────────────────────────────────────────────

def plot_yaw(df: pd.DataFrame, out_dir: str):
    t = df["t"]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        t,
        np.degrees(df["quad_yaw"]),
        color=C_QUAD,
        lw=LW,
        alpha=ALPHA_LINE,
        label="Quadcopter",
    )

    ax.plot(
        t,
        np.degrees(df["landing_pad_glob_yaw"]),
        color=C_PAD,
        lw=LW,
        alpha=ALPHA_LINE,
        label="Landing pad (UKF)",
        linestyle="--",
    )

    ax.plot(
        t,
        np.degrees(df["landing_pad_glob_true_yaw"]),
        color=C_TRUE,
        lw=LW,
        alpha=ALPHA_LINE,
        label="Landing pad (true)",
        linestyle=":",
    )

    # Only AprilTag has a meaningful raw yaw measurement.
    if "landing_pad_april_raw_stamp" in df.columns:
        mask = df["landing_pad_april_raw_stamp"].fillna(0) > 0
        mask &= df["landing_pad_april_raw_stamp"].ne(
            df["landing_pad_april_raw_stamp"].shift()
        )

        ax.scatter(
            t[mask],
            np.degrees(df.loc[mask, "landing_pad_april_glob_yaw_raw"]),
            color=C_APRIL,
            s=18,
            alpha=0.75,
            zorder=4,
            label="AprilTag raw",
        )

    style_ax(
        ax,
        ylabel="Yaw (°)",
        title="Yaw vs Time",
    )

    fig.tight_layout()
    save(fig, "yaw.png", out_dir)

    print("  Yaw plot done.")


# ── Acceleration / turn rate ───────────────────────────────────────────────

def plot_acceleration(df: pd.DataFrame, out_dir: str):
    t = df["t"]
    accel_col = "landing_pad_glob_a"

    if accel_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        t,
        df[accel_col],
        color=C_PAD,
        lw=LW,
        alpha=ALPHA_LINE,
        label="Landing pad (UKF)",
    )

    style_ax(
        ax,
        ylabel="Longitudinal Acceleration (m/s²)",
        title="Estimated Longitudinal Acceleration vs Time",
    )

    fig.tight_layout()
    save(fig, "acceleration.png", out_dir)

    print("  Acceleration plot done.")


def plot_turn_rate(df: pd.DataFrame, out_dir: str):
    t = df["t"]
    turn_col = "landing_pad_glob_yaw_rate"

    if turn_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        t,
        np.degrees(df[turn_col]),
        color=C_PAD,
        lw=LW,
        alpha=ALPHA_LINE,
        label="Landing pad (UKF)",
    )

    style_ax(
        ax,
        ylabel="Turn Rate (°/s)",
        title="Estimated Turn Rate vs Time",
    )

    fig.tight_layout()
    save(fig, "turn_rate.png", out_dir)

    print("  Turn rate plot done.")


# ── 3-D trajectory ─────────────────────────────────────────────────────────

def plot_3d(df: pd.DataFrame, out_dir: str):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    STARTX = df["landing_pad_glob_x"].iloc[0]
    STARTY = df["landing_pad_glob_y"].iloc[0]

    ax.plot(
        df["quad_x"],
        df["quad_y"],
        df["quad_z"],
        color=C_QUAD,
        lw=1.5,
        alpha=0.9,
        label="Quadcopter",
    )

    ax.plot(
        df["landing_pad_glob_x"] - STARTX,
        df["landing_pad_glob_y"] - STARTY,
        df["landing_pad_glob_z"],
        color=C_PAD,
        lw=1.5,
        alpha=0.9,
        label="Landing pad (UKF)",
        linestyle="--",
    )

    ax.plot(
        df["landing_pad_glob_true_x"] - STARTX,
        df["landing_pad_glob_true_y"] - STARTY,
        df["landing_pad_glob_true_z"],
        color=C_TRUE,
        lw=1.5,
        alpha=0.9,
        label="Landing pad (true)",
        linestyle=":",
    )

    ax.scatter(
        df[["quad_x", "quad_y", "quad_z"]].iloc[0, 0],
        df[["quad_x", "quad_y", "quad_z"]].iloc[0, 1],
        df[["quad_x", "quad_y", "quad_z"]].iloc[0, 2],
        color=C_QUAD,
        s=60,
        marker="o",
        zorder=5,
        label="Quad start",
    )

    ax.scatter(
        df[["quad_x", "quad_y", "quad_z"]].iloc[-1, 0],
        df[["quad_x", "quad_y", "quad_z"]].iloc[-1, 1],
        df[["quad_x", "quad_y", "quad_z"]].iloc[-1, 2],
        color=C_QUAD,
        s=80,
        marker="*",
        zorder=5,
        label="Quad end",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3-D Trajectory", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, framealpha=0.7, loc="upper left")

    all_vals = np.concatenate([
        df["quad_x"].values,
        df["quad_y"].values,
        df["quad_z"].values,
        (df["landing_pad_glob_x"] - STARTX).values,
        (df["landing_pad_glob_y"] - STARTY).values,
        df["landing_pad_glob_z"].values,
        (df["landing_pad_glob_true_x"] - STARTX).values,
        (df["landing_pad_glob_true_y"] - STARTY).values,
        df["landing_pad_glob_true_z"].values,
    ])

    mid = (all_vals.max() + all_vals.min()) / 2
    half = (all_vals.max() - all_vals.min()) / 2

    ax.set_xlim(mid - half, mid + half)
    ax.set_ylim(mid - half, mid + half)
    ax.set_zlim(mid - half, mid + half)

    fig.tight_layout()
    save(fig, "trajectory_3d.png", out_dir)

    print("  3-D trajectory plot done.")


# ── 2-D trajectory ─────────────────────────────────────────────────────────

def plot_2d(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 8))

    STARTX = df["landing_pad_glob_x"].iloc[0]
    STARTY = df["landing_pad_glob_y"].iloc[0]

    # Main trajectories
    ax.plot(
        df["quad_x"],
        df["quad_y"],
        color=C_QUAD,
        lw=LW,
        alpha=ALPHA_LINE,
        label="Quadcopter",
    )

    ax.plot(
        df["landing_pad_glob_x"] - STARTX,
        df["landing_pad_glob_y"] - STARTY,
        color=C_PAD,
        lw=LW,
        alpha=ALPHA_LINE,
        linestyle="--",
        label="Landing pad (UKF)",
    )

    ax.plot(
        df["landing_pad_glob_true_x"] - STARTX,
        df["landing_pad_glob_true_y"] - STARTY,
        color=C_TRUE,
        lw=LW,
        alpha=ALPHA_LINE,
        linestyle=":",
        label="Landing pad (true)",
    )

    # AprilTag raw
    if {
        "landing_pad_april_raw_stamp",
        "landing_pad_april_glob_x_raw",
        "landing_pad_april_glob_y_raw",
    }.issubset(df.columns):

        mask = df["landing_pad_april_raw_stamp"].fillna(0) > 0
        mask &= df["landing_pad_april_raw_stamp"].ne(
            df["landing_pad_april_raw_stamp"].shift()
        )

        ax.scatter(
            df.loc[mask, "landing_pad_april_glob_x_raw"] - STARTX,
            df.loc[mask, "landing_pad_april_glob_y_raw"] - STARTY,
            color=C_APRIL,
            s=20,
            alpha=0.7,
            label="AprilTag raw",
            zorder=4,
        )

    # YOLO landing-pad raw
    if {
        "landing_pad_yolo_LP_raw_stamp",
        "landing_pad_yolo_LP_glob_x_raw",
        "landing_pad_yolo_LP_glob_y_raw",
    }.issubset(df.columns):

        mask = df["landing_pad_yolo_LP_raw_stamp"].fillna(0) > 0
        mask &= df["landing_pad_yolo_LP_raw_stamp"].ne(
            df["landing_pad_yolo_LP_raw_stamp"].shift()
        )

        ax.scatter(
            df.loc[mask, "landing_pad_yolo_LP_glob_x_raw"] - STARTX,
            df.loc[mask, "landing_pad_yolo_LP_glob_y_raw"] - STARTY,
            color=C_YOLO_LP,
            s=22,
            alpha=0.7,
            label="YOLO landing-pad raw",
            zorder=4,
        )

    # YOLO car raw
    if {
        "landing_pad_yolo_car_raw_stamp",
        "landing_pad_yolo_car_glob_x_raw",
        "landing_pad_yolo_car_glob_y_raw",
    }.issubset(df.columns):

        mask = df["landing_pad_yolo_car_raw_stamp"].fillna(0) > 0
        mask &= df["landing_pad_yolo_car_raw_stamp"].ne(
            df["landing_pad_yolo_car_raw_stamp"].shift()
        )

        ax.scatter(
            df.loc[mask, "landing_pad_yolo_car_glob_x_raw"] - STARTX,
            df.loc[mask, "landing_pad_yolo_car_glob_y_raw"] - STARTY,
            color=C_YOLO_CAR,
            s=22,
            alpha=0.7,
            label="YOLO car raw",
            zorder=4,
        )

    # Start/end
    ax.scatter(
        df["quad_x"].iloc[0],
        df["quad_y"].iloc[0],
        color=C_QUAD,
        marker="o",
        s=70,
        label="Quad start",
        zorder=5,
    )

    ax.scatter(
        df["quad_x"].iloc[-1],
        df["quad_y"].iloc[-1],
        color=C_QUAD,
        marker="*",
        s=100,
        label="Quad end",
        zorder=5,
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Top-Down Trajectory")
    ax.grid(True, alpha=0.5)
    ax.legend(framealpha=0.7)

    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    save(fig, "trajectory_2d.png", out_dir)

    print("  2-D trajectory plot done.")


# ── Main ───────────────────────────────────────────────────────────────────

def main():

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(
            script_dir,
            "controller_20260928_214030.csv",
        )

    if not os.path.isfile(csv_path):
        print(f"ERROR: could not find CSV at '{csv_path}'")
        print("Usage: python plot_data.py [path/to/data.csv]")
        sys.exit(1)

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "plots",
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoading data from: {csv_path}")

    df = load(csv_path)

    print(
        f"  {len(df)} rows  |  "
        f"duration {df['t'].iloc[-1]:.1f} s\n"
    )

    # ── Clean true X/Y ────────────────────────────────────────────────

    t_arr = df["timestamp"].values

    true_x, true_y, outlier_mask = clean_true_position(
        df["landing_pad_glob_true_x"].values,
        df["landing_pad_glob_true_y"].values,
        t_arr,
    )

    df["landing_pad_glob_true_x"] = true_x
    df["landing_pad_glob_true_y"] = true_y

    n_outliers = outlier_mask.sum()

    if n_outliers:
        print(
            f"Removed {n_outliers} true-position outlier row(s) "
            f"at t = "
            f"{np.round(df['t'].values[outlier_mask], 2).tolist()}"
        )

    # Count raw detections
    measurement_specs = [
        (
            "AprilTag",
            "landing_pad_april_raw_stamp",
        ),
        (
            "YOLO landing pad",
            "landing_pad_yolo_LP_raw_stamp",
        ),
        (
            "YOLO car",
            "landing_pad_yolo_car_raw_stamp",
        ),
    ]

    print("\nRaw measurement counts:")

    for name, stamp_col in measurement_specs:
        if stamp_col in df.columns:
            events = df.loc[df[stamp_col].fillna(0) > 0, stamp_col]
            count = events.ne(events.shift()).sum()
            print(f"  {name:18s}: {count}")

    print("\nGenerating position plots …")
    plot_position(df, out_dir)

    print("\nGenerating velocity plots …")
    plot_velocity(df, out_dir)

    print("\nGenerating yaw plot …")
    plot_yaw(df, out_dir)

    print("\nGenerating 3-D trajectory …")
    plot_3d(df, out_dir)

    print("\nGenerating 2-D X-Y trajectory …")
    plot_2d(df, out_dir)

    print("\nGenerating acceleration plot …")
    plot_acceleration(df, out_dir)

    print("\nGenerating turn rate plot …")
    plot_turn_rate(df, out_dir)

    print("\nAll done! Opening figures …")
    plt.show()


if __name__ == "__main__":
    main()