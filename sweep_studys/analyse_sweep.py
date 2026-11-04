#!/usr/bin/env python3
"""
Raw YOLO box-fit error from a static sweep (the filter is ignored).

Inputs : <SWEEP_OUT>/manifest.csv (from sweep_runner.py) and, for each run_dir in it,
         the controller_*.csv your controller node wrote there (AUTO_LANDER_SWEEP=1).
Outputs: <SWEEP_OUT>/metrics_case.csv   one row per (R, elev, aspect)
         <SWEEP_OUT>/metrics_geom.csv   pooled over aspect, for the fan map
SWEEP_OUT defaults to sweep_out/ next to this script; run_sweep.sh sets it per condition.

One sample per new YOLO image: the box-fit pad position (fallback fit, or the shadow
fit if the UKF happens to be in box mode). Errors are in the car body frame
(lon = along the car, forward +; lat = across the car, left +; error = fit - truth),
reported as bias, std and RMSE, plus detection and fit-success rates.

Wrong-object gate (SWEEP-TEMP change): a detection is "wrong object" when its BEARING
from the drone is more than GATE_BEARING_DEG away from the true pad's bearing. A wrong
assumed pad height slides the estimate along the camera ray, which changes its range
but not its bearing, so a bearing gate never mistakes a large height-induced error for
a wrong object. (The old 5 m distance gate would have silently discarded exactly the
ray-cast errors the wrong-height condition is meant to measure: at 15 deg elevation a
1.5 m height error is a ~5.5 m range error.) Near nadir the bearing is undefined, so
within NADIR_D_M horizontal distance the distance gate GATE_M is used instead.

Rates during the hold:
    det_rate    frames with any detection / frames expected at YOLO_RATE
    valid_rate  frames with a correct detection / frames expected
    false_rate  frames with a wrong-object detection / frames expected
Missed frames never reach the error statistics (the CSV repeats the last raw value,
so only new image stamps are used).

Ray-cast pad offset (SWEEP-TEMP): the ray-cast returns where the bounding-box CENTRE ray
meets the pad-height plane, i.e. roughly the car's centre, not the pad. For a fair
comparison it is given the same heading assumption as the mesh fallback fit: the car
points straight away from the camera, along the drone's heading (pitch-only gimbal). The
pad is then CENTRE_TO_PAD behind that point along the drone heading. No ground truth is
used. For ray-cast folders (condition.txt says AUTO_LANDER_RAYCAST=1, or "raycast" in the
folder name) two sets of metrics are written:
    metrics_case.csv / metrics_geom.csv              offset applied (the fair baseline)
    metrics_case_naive.csv / metrics_geom_naive.csv  raw box-centre estimate
Mesh folders get only the first pair (the fit already places the pad on the car).
With AUTO_LANDER_TRUE_YAW=1 (condition.txt) the controller hands the true heading to
ukf.py, which offsets the ray-cast to the pad itself, so ray-cast folders also get only
the first pair.

Conventions follow plot_errors.py: relative X/Y are world-aligned, the car heading is
landing_pad_glob_true_yaw, and the truth is shifted to the pad by PAD_OFFSET.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ config
OUT_DIR    = Path(os.environ.get("SWEEP_OUT",                                  # SWEEP-TEMP
                                 Path(__file__).resolve().parent / "sweep_out"))
PAD_OFFSET = (-0.5, 0.0)   # pad in car frame (fwd, left); = yolo_car_pad_position[:2]
YOLO_RATE  = 10.0          # yolo_processing_rate (Hz), for the detection rate
GATE_BEARING_DEG = 30.0    # SWEEP-TEMP: bearing off the true pad by more = wrong object
NADIR_D_M  = 2.0           # SWEEP-TEMP: below this true horizontal distance, bearing is
                           # meaningless, so fall back to the distance gate
GATE_M     = 5.0           # distance gate, used only near nadir
CENTRE_TO_PAD = (-0.465, 0.0)  # SWEEP-TEMP: pad minus car-extents centre, car frame (fwd, left);
                               # centre x = (-2.196 + 2.126) / 2 = -0.035 from yolo_car_boxes
# -------------------------------------------------------------------------


def stats(lat, lon):
    n = len(lat)
    out = {"n": n}
    for name, v in (("lat", lat), ("lon", lon)):
        out[f"bias_{name}"] = v.mean() if n else np.nan
        out[f"std_{name}"] = v.std(ddof=1) if n > 1 else np.nan
        out[f"rmse_{name}"] = np.sqrt(np.mean(v ** 2)) if n else np.nan
    return out


def load_run(run_dir):
    files = [f for f in sorted(Path(run_dir).glob("controller_*.csv")) if f.stat().st_size > 0]
    if not files:
        raise FileNotFoundError(
            f"No non-empty controller_*.csv in {run_dir}. If the stack is still running, "
            f"stop it first: the controller only flushes its CSV when it shuts down.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True).sort_values("timestamp")


def is_raycast(run_dir):
    """SWEEP-TEMP: read condition.txt written by run_sweep.sh, else guess from the path."""
    cond = Path(run_dir) / "condition.txt"
    if cond.exists():
        return "AUTO_LANDER_RAYCAST=1" in cond.read_text()
    return "raycast" in str(run_dir).lower()


def is_true_yaw(run_dir):
    """SWEEP-TEMP: the controller supplied the true car heading (AUTO_LANDER_TRUE_YAW=1).
    Then the ray-cast already applies the pad offset in ukf.py, so no offline offset."""
    cond = Path(run_dir) / "condition.txt"
    return cond.exists() and "AUTO_LANDER_TRUE_YAW=1" in cond.read_text()


def wrong_object(raw, truth, err):
    """SWEEP-TEMP: bearing gate away from nadir, distance gate near it."""
    with np.errstate(invalid="ignore"):
        b_raw = np.arctan2(raw[:, 1], raw[:, 0])
        b_true = np.arctan2(truth[:, 1], truth[:, 0])
        b_err = np.degrees(np.abs((b_raw - b_true + np.pi) % (2 * np.pi) - np.pi))
        d_true = np.hypot(truth[:, 0], truth[:, 1])
        return np.where(d_true > NADIR_D_M, b_err > GATE_BEARING_DEG, err > GATE_M)


def main():
    print(f"Analysing {OUT_DIR}")
    man = pd.read_csv(OUT_DIR / "manifest.csv")
    man = man[man.status == "ok"].drop_duplicates("case_id", keep="last")
    runs = {}
    for rd in man.run_dir.unique():
        try:
            runs[rd] = load_run(rd)
        except FileNotFoundError as e:
            n = int((man.run_dir == rd).sum())
            print(f"WARNING: skipping {n} cases from {Path(rd).name}: {e}")
    man = man[man.run_dir.isin(list(runs))]
    if man.empty:
        raise SystemExit("No cases with a controller CSV to analyse.")

    rows, errs = {}, {}            # SWEEP-TEMP: keyed by metrics suffix ("" or "_naive")
    for _, c in man.iterrows():
        df = runs[c.run_dir]
        w = df[(df.timestamp >= c.t_start) & (df.timestamp <= c.t_end)]
        if w.empty:
            continue

        # One row per new YOLO image (the CSV repeats the last raw value until the next)
        stamp = w.landing_pad_yolo_car_raw_stamp.fillna(0)
        ev = w[((stamp > 0) & stamp.ne(stamp.shift())).to_numpy()]
        n_img = len(ev)
        raw = ev[["landing_pad_yolo_car_rel_x_raw", "landing_pad_yolo_car_rel_y_raw"]].to_numpy()
        ok = np.isfinite(raw).all(axis=1)

        # Truth at those rows: pad relative to the quad, world-aligned
        yaw = ev.landing_pad_glob_true_yaw.to_numpy()
        fwd = np.column_stack([np.cos(yaw), np.sin(yaw)])
        left = np.column_stack([-np.sin(yaw), np.cos(yaw)])
        truth = (ev[["landing_pad_rel_true_x", "landing_pad_rel_true_y"]].to_numpy()
                 + PAD_OFFSET[0] * fwd + PAD_OFFSET[1] * left)

        # SWEEP-TEMP: ray-cast gets the mesh's heading assumption (car along drone heading)
        variants = {"": raw}
        if is_raycast(c.run_dir) and not is_true_yaw(c.run_dir):
            qy = ev.quad_yaw.to_numpy()
            a_fwd = np.column_stack([np.cos(qy), np.sin(qy)])
            a_left = np.column_stack([-np.sin(qy), np.cos(qy)])
            variants = {"": raw + CENTRE_TO_PAD[0] * a_fwd + CENTRE_TO_PAD[1] * a_left,
                        "_naive": raw}

        for suffix, est in variants.items():
            e = est - truth
            err = np.hypot(e[:, 0], e[:, 1])
            wo = wrong_object(est, truth, err)                      # SWEEP-TEMP
            good = ok & ~wo
            wrong = ok & wo
            lat = np.sum(e[good] * left[good], axis=1)
            lon = np.sum(e[good] * fwd[good], axis=1)

            dur = c.t_end - c.t_start
            expected = max(dur * YOLO_RATE, 1.0)
            rows.setdefault(suffix, []).append(dict(
                case_id=c.case_id, R=c.R, elev=c.elev, aspect=c.aspect, dur=dur,
                det_rate=min(1.0, n_img / expected),
                valid_rate=min(1.0, good.sum() / expected),
                false_rate=min(1.0, wrong.sum() / expected),
                fit_rate=ok.mean() if n_img else np.nan,
                wrong_err_med=np.median(err[wrong]) if wrong.any() else np.nan,
                **stats(lat, lon)))
            errs.setdefault(suffix, []).append(
                pd.DataFrame(dict(R=c.R, elev=c.elev, lat=lat, lon=lon)))

    if not rows:
        raise SystemExit("No logged frames inside any case window.")
    for suffix in rows:
        summarise(pd.DataFrame(rows[suffix]), pd.concat(errs[suffix], ignore_index=True), suffix)


def summarise(case_df, E, suffix):
    """Pool over aspect, write metrics{suffix}.csv and print a summary."""
    label = "naive box centre" if suffix == "_naive" else "pad estimate"
    geom = []
    for (R, el), cd in case_df.groupby(["R", "elev"]):
        g = E[(E.R == R) & (E.elev == el)]
        geom.append(dict(R=R, elev=el,
                         d=R * np.cos(np.radians(el)), h=R * np.sin(np.radians(el)),
                         det_rate=np.average(cd.det_rate, weights=cd.dur),
                         valid_rate=np.average(cd.valid_rate, weights=cd.dur),
                         false_rate=np.average(cd.false_rate, weights=cd.dur),
                         **stats(g.lat.to_numpy(), g.lon.to_numpy())))
    geom_df = pd.DataFrame(geom)

    case_df.to_csv(OUT_DIR / f"metrics_case{suffix}.csv", index=False)
    geom_df.to_csv(OUT_DIR / f"metrics_geom{suffix}.csv", index=False)
    print(f"[{label}] {len(case_df)} cases -> metrics_case{suffix}.csv, "
          f"{len(geom_df)} geometries -> metrics_geom{suffix}.csv")
    w = case_df.dur
    print(f"Hold-period rates (all cases): detected {np.average(case_df.det_rate, weights=w):.0%}, "
          f"correct {np.average(case_df.valid_rate, weights=w):.0%}, "
          f"wrong object {np.average(case_df.false_rate, weights=w):.0%}")
    hit = case_df[case_df.false_rate > 0.05].sort_values("false_rate", ascending=False)
    if len(hit):
        print(f"{len(hit)} cases with >5% wrong-object detections, worst first:")
        print(hit[["case_id", "det_rate", "valid_rate", "false_rate", "wrong_err_med"]]
              .head(15).round(2).to_string(index=False))

    # Sanity check: the easiest case (steepest, closest) should be near-unbiased.
    # Expected to fire for the wrong-height conditions only if nadir is badly affected.
    ref = case_df.sort_values(["elev", "R"], ascending=[False, True]).iloc[0]
    if ref.n == 0 or np.hypot(ref.bias_lat, ref.bias_lon) > 1.0:
        print(f"WARNING: {ref.case_id} has no fits or >1 m bias. "
              f"Check CAR_MODEL, PAD_OFFSET and the time windows before trusting the rest.")


if __name__ == "__main__":
    main()
