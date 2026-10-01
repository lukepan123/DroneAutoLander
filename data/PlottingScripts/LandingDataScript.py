import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

ROOT_DIR = Path("/home/luke/ros2_ws/TurnRunData/10ms_11degs")  # folder containing the 15ms_0_T1, etc. folders

# A trial is considered to have landed when true_height drops to
# this value or below.
LANDING_HEIGHT_THRESHOLD = 1.9  # m

# Maximum horizontal landing error for a "successful" landing.
SUCCESS_ERROR_THRESHOLD = 0.2  # m

# Output files
TRIAL_OUTPUT = ROOT_DIR / "trial_results.csv"
SUMMARY_OUTPUT = ROOT_DIR / "trial_summary.csv"


# ============================================================
# HELPERS
# ============================================================

TRIAL_RE = re.compile(
    r"^(?P<speed>\d+(?:\.\d+)?)ms_"
    r"(?:(?P<turn_rate>-?\d+(?:\.\d+)?)degs_)?"
    r"(?P<angle>-?\d+(?:\.\d+)?)_T(?P<trial>\d+)$"
)


def first_timestamp_for_state(df, state, after_time=None):
    """Return the first timestamp where controller_state == state."""
    mask = np.isclose(df["controller_state"], state, equal_nan=False)

    if after_time is not None:
        mask &= df["timestamp"] >= after_time

    rows = df.loc[mask, "timestamp"]

    if rows.empty:
        return np.nan

    return rows.iloc[0]


def find_landing_row(df, acquired_time):
    if pd.isna(acquired_time):
        return None

    mask = (
        (df["timestamp"] >= acquired_time)
        & np.isclose(df["controller_state"], 6000)
    )

    rows = df.loc[mask]

    if rows.empty:
        return None

    return rows.iloc[0]


def calculate_landing_error(row):
    if row is None:
        return np.nan

    x = row["landing_pad_rel_true_x"]
    y = row["landing_pad_rel_true_y"]
    yaw = row["landing_pad_rel_true_yaw"]

    if pd.isna(x) or pd.isna(y) or pd.isna(yaw):
        return np.nan

    # 0.4 m behind the reported true position in the pad/body frame
    offset_body_x = 0.0
    offset_body_y = -0.4

    # Rotate body-frame offset into world/ENU frame
    offset_x = offset_body_x * np.cos(yaw) - offset_body_y * np.sin(yaw)
    offset_y = offset_body_x * np.sin(yaw) + offset_body_y * np.cos(yaw)

    x += offset_x
    y += offset_y

    return float(np.hypot(x, y))


# ============================================================
# PROCESS ONE TRIAL
# ============================================================

def process_trial(folder):
    match = TRIAL_RE.match(folder.name)

    if not match:
        return None

    speed = float(match.group("speed"))
    turn_rate = (
        float(match.group("turn_rate"))
        if match.group("turn_rate") is not None
        else 0.0
    )
    angle = float(match.group("angle"))
    trial = int(match.group("trial"))

    csv_path = folder / f"{folder.name}.csv"

    if not csv_path.exists():
        print(f"WARNING: missing CSV: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"WARNING: could not read {csv_path}: {e}")
        return None

    required_columns = [
        "timestamp",
        "controller_state",
        "true_height",
        "landing_pad_rel_true_x",
        "landing_pad_rel_true_y",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        print(f"WARNING: {csv_path} missing columns: {missing}")
        return None

    # Convert relevant columns to numeric
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "controller_state"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # State 2000 -> State 3000
    # --------------------------------------------------------

    first_detection_time = first_timestamp_for_state(
        df, 2000
    )

    pad_acquired_time = first_timestamp_for_state(
        df, 3000, after_time=first_detection_time
    )

    if pd.isna(first_detection_time) or pd.isna(pad_acquired_time):
        detection_to_acquired = np.nan
    else:
        detection_to_acquired = (
            pad_acquired_time - first_detection_time
        )

    # --------------------------------------------------------
    # Landing
    # --------------------------------------------------------

    landing_row = find_landing_row(df, pad_acquired_time)

    if landing_row is None:
        landing_time = np.nan
        landing_error = np.nan
        landing_success = False
        landed = False
    else:
        landing_time = float(landing_row["timestamp"])
        landing_error = calculate_landing_error(landing_row)

        landed = True
        landing_success = (
            not pd.isna(landing_error)
            and landing_error <= SUCCESS_ERROR_THRESHOLD
        )

    return {
        "speed_mps": speed,
        "turn_rate_dps": turn_rate,
        "angle_deg": angle,
        "trial": trial,

        "first_detection_time": first_detection_time,
        "pad_acquired_time": pad_acquired_time,
        "detection_to_acquired_s": detection_to_acquired,

        "landing_time": landing_time,
        "landing_error_xy_m": landing_error,

        "landed": landed,
        "landing_success": landing_success,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    results = []

    # Find folders matching:
    #   15ms_0_T1
    #   20ms_-10_T2
    #   7.5ms_15_T3
    # etc.
    for folder in sorted(ROOT_DIR.iterdir()):
        if not folder.is_dir():
            continue

        if TRIAL_RE.match(folder.name):
            result = process_trial(folder)

            if result is not None:
                results.append(result)

    if not results:
        print("No trial folders found.")
        return

    # --------------------------------------------------------
    # Per-trial results
    # --------------------------------------------------------

    trial_df = pd.DataFrame(results)

    trial_df = trial_df.sort_values(
        ["speed_mps", "angle_deg", "trial"]
    ).reset_index(drop=True)

    trial_df.to_csv(TRIAL_OUTPUT, index=False)

    # --------------------------------------------------------
    # Per-condition summary
    # --------------------------------------------------------

    summary = (
        trial_df
        .groupby(["speed_mps", "turn_rate_dps", "angle_deg"], dropna=False)
        .agg(
            trials=("trial", "count"),
            avg_landing_error_xy_m=("landing_error_xy_m", "mean"),
            landing_success_rate=("landing_success", "mean"),
            avg_detection_to_acquired_s=("detection_to_acquired_s", "mean"),
            landed_trials=("landed", "sum"),
        )
        .reset_index()
    )

    summary["landing_success_rate"] *= 100.0

    summary = summary.rename(
        columns={
            "landing_success_rate": "landing_success_rate_percent"
        }
    )

    summary.to_csv(SUMMARY_OUTPUT, index=False)

    # --------------------------------------------------------
    # Overall averages
    # --------------------------------------------------------

    overall = {
        "trials": len(trial_df),
        "avg_landing_error_xy_m": trial_df[
            "landing_error_xy_m"
        ].mean(),
        "landing_success_rate_percent":
            trial_df["landing_success"].mean() * 100.0,
        "avg_detection_to_acquired_s":
            trial_df["detection_to_acquired_s"].mean(),
        "landed_trials":
            trial_df["landed"].sum(),
    }

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    print("\n================ PER-TRIAL RESULTS ================\n")
    print(trial_df.to_string(index=False))

    print("\n================ CONDITION AVERAGES ================\n")
    print(summary.to_string(index=False))

    print("\n================ OVERALL ================\n")
    print(f"Trials:                         {overall['trials']}")
    print(
        f"Avg landing error:              "
        f"{overall['avg_landing_error_xy_m']:.3f} m"
    )
    print(
        f"Landing success rate:           "
        f"{overall['landing_success_rate_percent']:.1f}%"
    )
    print(
        f"Avg detection -> acquired time: "
        f"{overall['avg_detection_to_acquired_s']:.3f} s"
    )
    print(
        f"Landed trials:                  "
        f"{int(overall['landed_trials'])}/{overall['trials']}"
    )

    print(f"\nPer-trial results saved to: {TRIAL_OUTPUT}")
    print(f"Summary saved to:            {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()