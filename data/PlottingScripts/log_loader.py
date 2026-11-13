"""
log_loader.py – shared loader for SITL and real-flight controller logs
======================================================================
Used by plot_state.py and plot_errors.py (keep it in the same folder).

    from log_loader import load_log
    df = load_log("controller_20261113_155759.csv")          # fleet file found automatically
    df = load_log(path, fleet="logs/fleet.csv", rtk_bias=(2.1, -1.0))

What it does
------------
* SITL logs (ground_truth_available = True) are returned as logged, with platform 'sim'.
* Real logs (all *_true_* columns NaN) get platform 'real'. If an RTK telemetry log
  (fleet.csv: one row per vehicle message, columns name, lat, lon, rel_alt_m, vx_ms,
  vy_ms, vz_ms, hdg_deg, fc_boot_ms, utc_est, h_acc_m, fix_name, ...) overlaps the flight,
  the *_true_* columns are synthesised from it IN THE SAME CONVENTIONS AS SITL, so the
  plotting scripts need no special cases:

      quad_true_{x,y,z,vx,vy,vz,yaw}     quad (local ENU frame of the controller log)
      landing_pad_glob_true_{x,y,z,...}  CAR ORIGIN (ground level, car odometry origin),
                                         exactly like /landing_pad/odom in Gazebo
      landing_pad_rel_true_*             car origin - quad
      true_height, nees                  as the controller would have logged them
      truth_sigma_xy                     reported 1-sigma of the truth (car and quad h_acc)

  Fleet logs are found automatically: *fleet*.csv / *rtk*.csv next to the log, or inside
  the ground station's rec_<YYYYMMDD>_<HHMMSS>Z folders (found at any depth under the log's
  folder, its parent and FLEET_SEARCH_ROOTS, e.g. pi_7inch/recordings/rec_*Z/). The recording that overlaps the flight most is used.

  Alignment, all estimated from the data:
    - Pi clock vs GPS time: the quad appears in both logs (controller quad_x/y from
      /mavros/global_position/local, fleet quad from GLOBAL_POSITION_INT), so the clock
      offset is the time shift that best overlays the two tracks.
    - Local frame origin (FCU home): median offset between the two quad tracks.
    - Each FCU's boot clock is mapped to UTC with a linear fit (the car FCU's clock was
      measured to run ~0.23 % slow).
  Car heading: GPS course over ground when moving (> COURSE_MIN_SPEED), otherwise the
  car's compass heading corrected by its measured bias (the car compass read ~24 deg
  off its course in the Nov-13 logs, so it is never used raw).

Every result also gets quad_ref_{x,y,z,vx,vy,vz,yaw}: truth when available, else the FCU
estimate, so plots always have a quad trajectory. df.attrs carries the metadata
(truth_source, platform, pad geometry, alignment numbers).

Truth limitations (real flights)
--------------------------------
The quad flew on a standalone 3-D GPS fix (h_acc 0.6-1.0 m) while the car was RTK FLOAT
against the base, so their frames need not agree: a near-constant offset of ~2-3 m was
found between them in both Nov-13 RTK flights. `rtk_bias` removes a constant ENU offset
(car minus quad frame): pass measured numbers (e.g. from a static co-location before
take-off), or 'auto' to estimate it from the vision data itself. 'auto' makes absolute
errors meaningless (they are referenced to the vision mean) but keeps dynamic errors,
lag and noise valid. Heights use the flat-field assumption (car origin at the quad's
home height), so height "truth" is only as good as the baro.
"""
import glob
import os

import numpy as np
import pandas as pd

# ── Platform geometry ──────────────────────────────────────────────────────
# pad_in_car: landing-pad top centre in the car frame (x fwd, y left, z up, origin on
# the ground at the car's odometry origin). Keep in step with the launch files.
PLATFORMS = {
    "sim": dict(
        name="SITL Grand Vitara",
        # main_launch_sim.py yolo_car_pad_position. SITL car frame: origin on the ground
        # midway between the axles (the real car's frame below is centred on the body, so
        # the x offsets differ legitimately)
        pad_in_car=(-0.80, 0.0, 1.705),
        pad_size=(1.20, 1.20),             # landing_pad_size (length, width)
        # Body footprint x_min, x_max, y_min, y_max (m), mirrors excluded, from the mesh
        car_footprint=(-2.41, 2.06, -0.88, 0.88),
    ),
    "real": dict(
        name="2008 Grand Vitara",
        pad_in_car=(-0.79, 0.0, 1.75),     # main_launch.py CAR_PAD_POSITION (height ESTIMATE)
        # TODO measure: car GPS antenna position in the car frame (x, y). Default assumes
        # the car's GPS sits on the pad centre.
        car_antenna=(-0.79, 0.0),
    ),
}

# ── RTK telemetry log ──────────────────────────────────────────────────────
FLEET_PATTERNS = ("*fleet*.csv", "*rtk*.csv")
# Field-day layout: controller logs in .../pi_7inch, ground-station recordings in
# .../pi_7inch/recordings/rec_<YYYYMMDD>_<HHMMSS>Z/ (UTC start time), fleet log inside.
# rec_*Z folders are found at ANY depth under these roots, the controller CSV's folder and
# its parent, so the layout also works when copied elsewhere. Add more field days here.
FLEET_SEARCH_ROOTS = [
    "/home/luke/ros2_ws/field_2026-11-13_novideo/field_2026-11-13/pi_7inch/recordings",
]
REC_DIR_GLOB = "rec_*Z"
RTK_CAR_NAME = "car"
RTK_BASE_NAME = "base"
RTK_QUAD_NAME = None            # None: the one vehicle that is neither car nor base
GROUND_Z = 0.0                  # car origin height in the local frame (flat field, m)
COURSE_MIN_SPEED = 1.0          # m/s, below this the car heading comes from its compass
MAX_FLEET_GAP = 1.0             # s, truth is NaN further than this from a fleet sample
CLOCK_SEARCH = 5.0              # s, +- range searched for the Pi -> GPS clock offset
ALIGN_RMS_WARN = 0.5            # m, warn if the quad tracks overlay worse than this
EARTH_R = 6378137.0

TRUTH_COLS = [
    "quad_true_x", "quad_true_y", "quad_true_z", "quad_true_vx", "quad_true_vy",
    "quad_true_vz", "quad_true_yaw",
    "landing_pad_glob_true_x", "landing_pad_glob_true_y", "landing_pad_glob_true_z",
    "landing_pad_glob_true_vx", "landing_pad_glob_true_vy", "landing_pad_glob_true_vz",
    "landing_pad_glob_true_yaw",
    "landing_pad_rel_true_x", "landing_pad_rel_true_y", "landing_pad_rel_true_z",
    "landing_pad_rel_true_vx", "landing_pad_rel_true_vy", "landing_pad_rel_true_vz",
    "landing_pad_rel_true_yaw", "true_height",
]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _finite_any(df, col):
    return col in df.columns and np.isfinite(pd.to_numeric(df[col], errors="coerce")).any()


def _log(verbose, *a):
    if verbose:
        print(*a)


# ── Public entry point ─────────────────────────────────────────────────────
def load_log(csv_path, fleet="auto", rtk_bias=None, verbose=True):
    """ Load a controller CSV (SITL or real) and attach truth where it exists.

    :param csv_path: controller_*.csv
    :param fleet:    RTK telemetry CSV, 'auto' (search next to the log, the cwd, this file's
                     folder and rec_*Z recording folders, see candidate_fleets), or None
    :param rtk_bias: None / (east, north) in m / 'auto'. Constant offset of the car's RTK
                     frame relative to the quad's GPS frame, removed from the car track.
    :return: DataFrame with df.attrs metadata (see module docstring)
    """
    df = pd.read_csv(csv_path)
    # A log cut by a flat battery / killed process can end in a partial row
    df = df[np.isfinite(pd.to_numeric(df["timestamp"], errors="coerce"))]
    df = df.reset_index(drop=True).copy()     # copy() also consolidates ~160 columns

    sim_truth = _finite_any(df, "quad_true_x") and _finite_any(df, "landing_pad_glob_true_x")
    platform = "sim" if sim_truth else "real"
    attrs = dict(platform=platform, platform_name=PLATFORMS[platform]["name"],
                 pad_in_car=PLATFORMS[platform]["pad_in_car"],
                 lp_above_gnd=PLATFORMS[platform]["pad_in_car"][2],
                 truth_source="sim" if sim_truth else "none", rtk={})

    if not sim_truth:
        for col in TRUTH_COLS:
            if col not in df.columns:
                df[col] = np.nan
        path = find_fleet(csv_path, df) if fleet == "auto" else fleet
        if path:
            try:
                info = attach_rtk_truth(df, path, rtk_bias=rtk_bias, verbose=verbose)
                if info is not None:
                    attrs["truth_source"] = "rtk"
                    attrs["rtk"] = info
            except Exception as exc:          # never lose the plots over the truth
                _log(verbose, f"RTK truth: failed on {path} ({exc}) - continuing without truth.")
        else:
            if verbose:
                explain_no_fleet(csv_path, df)

    _add_quad_ref(df, attrs["truth_source"] != "none")
    df.attrs.update(attrs)
    return df


def resolve_csv(csv_path=None, log_dir=None, csv_file=None, script_dir=None):
    """ Which controller CSV to analyse, in order of precedence:
        1. csv_path (command line or caller), as given, or looked up in log_dir if it
           doesn't exist as given (so a bare file name works)
        2. csv_file set in the script: a file name in log_dir, or a full path
        3. the newest controller_*.csv in log_dir, then next to the script
    """
    dirs = [d for d in (log_dir, script_dir) if d]

    def lookup(name):
        if os.path.isfile(name):
            return os.path.abspath(name)
        for d in dirs:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        return None

    for name, origin in ((csv_path, "given"), (csv_file, "set in the script")):
        if name:
            p = lookup(str(name))
            if p is None:
                raise FileNotFoundError(f"controller CSV {origin} not found: '{name}' "
                                        f"(looked in: {', '.join(map(str, dirs)) or 'cwd'})")
            return p
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, "controller_*.csv")), key=os.path.getmtime)
        if found:
            return found[-1]
    raise FileNotFoundError(f"no controller_*.csv found in: {', '.join(map(str, dirs))}")


def describe_truth(df):
    """ One-line description of where the truth came from, for summaries and titles. """
    a = df.attrs
    src = a.get("truth_source", "none")
    if src == "sim":
        return f"SITL ground truth ({a.get('platform_name')})"
    if src == "none":
        return f"no ground truth - real flight ({a.get('platform_name')}); FCU estimates only"
    r = a.get("rtk", {})
    bias = r.get("bias", (0.0, 0.0))
    return (f"RTK truth from {_short(r.get('fleet', '?'))} "
            f"(clock {r.get('clock_offset', np.nan):+.3f} s, overlay RMS "
            f"{r.get('align_rms', np.nan):.2f} m, bias removed ({bias[0]:+.2f}, {bias[1]:+.2f}) m"
            f"{' AUTO' if r.get('bias_auto') else ''}, median truth 1-sigma "
            f"{r.get('truth_sigma_med', np.nan):.2f} m)")


# ── Fleet discovery ────────────────────────────────────────────────────────
def _short(path):
    """ 'rec_.../fleet.csv' (from the rec_*Z folder down) or just the file name, for printing. """
    parts = os.path.normpath(path).split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith("rec_"):
            return "/".join(parts[i:])
    return parts[-1]


def _fleet_span(path):
    try:
        f = pd.read_csv(path, usecols=["name", "utc_est"])
    except (ValueError, OSError, pd.errors.ParserError):
        return None
    if not len(f):
        return None
    return float(f["utc_est"].min()), float(f["utc_est"].max())


def _rec_start(path):
    """ UTC start (epoch s) from a rec_YYYYMMDD_HHMMSSZ folder anywhere in the path. """
    from datetime import datetime, timezone
    for part in reversed(os.path.normpath(path).split(os.sep)):
        if part.startswith("rec_") and part.endswith("Z"):
            try:
                return datetime.strptime(part, "rec_%Y%m%d_%H%M%SZ").replace(
                    tzinfo=timezone.utc).timestamp()
            except ValueError:
                return None
    return None


def _search_roots(csv_path):
    log_dir = os.path.dirname(os.path.abspath(csv_path))
    flat = [log_dir, os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    rec_roots = [log_dir, os.path.dirname(log_dir)] + \
        [os.path.abspath(r) for r in FLEET_SEARCH_ROOTS if os.path.isdir(r)]
    return list(dict.fromkeys(flat)), list(dict.fromkeys(rec_roots))


def candidate_fleets(csv_path):
    """ Every fleet/RTK CSV that could belong to this log: next to it, in the cwd or this
        file's folder, and anywhere inside rec_*Z folders found (at any depth) under the
        log's folder, its parent and FLEET_SEARCH_ROOTS. """
    flat, rec_roots = _search_roots(csv_path)
    found = set()
    for d in flat:
        for pat in FLEET_PATTERNS:
            found.update(glob.glob(os.path.join(d, pat)))
    rec_dirs = set()
    for root in rec_roots:
        if os.path.basename(root.rstrip(os.sep)).startswith("rec_"):
            rec_dirs.add(root)
        rec_dirs.update(glob.glob(os.path.join(root, "**", REC_DIR_GLOB), recursive=True))
    for d in rec_dirs:
        for pat in FLEET_PATTERNS:
            found.update(glob.glob(os.path.join(d, "**", pat), recursive=True))
    return sorted(os.path.abspath(p) for p in found)


def explain_no_fleet(csv_path, df):
    """ Print why no fleet log matched: where it looked and what it found. """
    t0, t1 = float(df["timestamp"].min()), float(df["timestamp"].max())
    from datetime import datetime, timezone
    fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%SZ")
    flat, rec_roots = _search_roots(csv_path)
    print(f"RTK truth: no fleet log overlaps this flight ({fmt(t0)} - {fmt(t1)} UTC).")
    print(f"  searched for {' / '.join(FLEET_PATTERNS)} in: {', '.join(flat)}")
    print(f"  and in rec_*Z folders under: {', '.join(rec_roots)}")
    cands = candidate_fleets(csv_path)
    if not cands:
        print("  -> no candidate files found at all: check FLEET_SEARCH_ROOTS / file names.")
    for p in cands[:20]:
        span = _fleet_span(p)
        txt = "unreadable / not a fleet log" if span is None \
            else f"covers {fmt(span[0])} - {fmt(span[1])} UTC"
        print(f"  candidate {_short(p)}: {txt}")
    if len(cands) > 20:
        print(f"  ... and {len(cands) - 20} more")


def find_fleet(csv_path, df=None):
    """ The fleet/RTK CSV whose time span overlaps this flight the most (None if none). """
    if df is None:
        df = pd.read_csv(csv_path, usecols=["timestamp"])
    t0, t1 = float(df["timestamp"].min()), float(df["timestamp"].max())
    best, best_ov = None, 5.0          # need > 5 s overlap
    for p in candidate_fleets(csv_path):
        start = _rec_start(p)
        if start is not None and start > t1 + 60.0:   # recording began after the flight
            continue
        span = _fleet_span(p)
        if span is None:
            continue
        ov = min(t1, span[1]) - max(t0, span[0])
        if ov > best_ov:
            best, best_ov = p, ov
    return best


# ── RTK truth ──────────────────────────────────────────────────────────────
def _enu(lat, lon, lat0, lon0):
    e = np.radians(np.asarray(lon) - lon0) * EARTH_R * np.cos(np.radians(lat0))
    n = np.radians(np.asarray(lat) - lat0) * EARTH_R
    return e, n


def _track(f, name, lat0, lon0):
    """ One vehicle's track: unique FCU samples, UTC time from a robust boot-clock fit. """
    d = f[f["name"] == name].drop_duplicates("fc_boot_ms").copy()
    d = d[np.isfinite(d["fc_boot_ms"]) & np.isfinite(d["lat"])]
    if len(d) < 10:
        return None
    tb, tu = d["fc_boot_ms"].to_numpy(float) / 1e3, d["utc_est"].to_numpy(float)
    keep = np.ones(len(d), bool)
    for _ in range(3):                         # drop telemetry-delayed rows, refit
        p = np.polyfit(tb[keep], tu[keep], 1)
        r = tu - np.polyval(p, tb)
        keep = np.abs(r - np.median(r[keep])) < max(3 * np.std(r[keep]), 0.05)
    d["t"] = np.polyval(p, tb)
    d["clock_slope"] = p[0]
    d["e"], d["n"] = _enu(d["lat"], d["lon"], lat0, lon0)
    d["ve"], d["vn"] = d["vy_ms"].astype(float), d["vx_ms"].astype(float)   # NED -> ENU
    d["vu"] = -d["vz_ms"].astype(float)
    d = d.sort_values("t")
    d = d[np.r_[True, np.diff(d["t"].to_numpy()) > 1e-3]]
    return d.reset_index(drop=True)


def _hermite(tk, x, v, t):
    """ Position at times t from samples (tk, x, v): cubic Hermite, NaN outside. """
    from scipy.interpolate import CubicHermiteSpline
    out = CubicHermiteSpline(tk, x, v, extrapolate=False)(t)
    return out


def _gap_mask(tk, t):
    """ True where t is within MAX_FLEET_GAP of a fleet sample and inside the span. """
    i = np.clip(np.searchsorted(tk, t), 1, len(tk) - 1)
    near = np.minimum(np.abs(t - tk[i - 1]), np.abs(tk[i] - t))
    return (t >= tk[0]) & (t <= tk[-1]) & (near <= MAX_FLEET_GAP)


def _car_yaw_samples(c):
    """ Car heading (ENU, rad) per fleet sample + the measured compass bias (rad). """
    speed = np.hypot(c["ve"], c["vn"]).to_numpy()
    course = np.arctan2(c["vn"], c["ve"]).to_numpy()
    compass = np.radians(90.0 - c["hdg_deg"].to_numpy(float))
    moving = speed > COURSE_MIN_SPEED
    if moving.sum() >= 5:
        d = _wrap(compass[moving] - course[moving])
        bias = float(np.arctan2(np.median(np.sin(d)), np.median(np.cos(d))))
    else:
        bias = 0.0
    yaw = np.where(moving, course, _wrap(compass - bias))
    yaw = np.where(np.isfinite(yaw), yaw, _wrap(compass - bias))
    return np.unwrap(yaw), bias, moving


def attach_rtk_truth(df, fleet_path, rtk_bias=None, verbose=True):
    """ Fill the *_true_* columns of a real-flight log from an RTK fleet log (in place).
        Returns the alignment info dict, or None if the logs can't be aligned. """
    f = pd.read_csv(fleet_path)
    names = set(f["name"].astype(str).unique())
    quad_name = RTK_QUAD_NAME or next(
        (n for n in sorted(names) if n not in (RTK_CAR_NAME, RTK_BASE_NAME)), None)
    if RTK_CAR_NAME not in names or quad_name is None:
        _log(verbose, f"RTK truth: {fleet_path} lacks '{RTK_CAR_NAME}' or a quad track.")
        return None

    base = f[f["name"] == RTK_BASE_NAME]
    ref = base.iloc[0] if len(base) else f[f["name"] == RTK_CAR_NAME].iloc[0]
    lat0, lon0 = float(ref["lat"]), float(ref["lon"])
    q = _track(f, quad_name, lat0, lon0)
    c = _track(f, RTK_CAR_NAME, lat0, lon0)
    if q is None or c is None:
        _log(verbose, "RTK truth: too few fleet samples.")
        return None

    tc = df["timestamp"].to_numpy(float)
    qx, qy = df["quad_x"].to_numpy(float), df["quad_y"].to_numpy(float)
    qtk = q["t"].to_numpy()

    # ── Pi clock -> GPS time: shift that best overlays the two quad tracks ──
    def overlay(dt):
        tt = tc + dt
        ok = (tt > qtk[0]) & (tt < qtk[-1])
        if ok.sum() < 50:
            return np.inf, ok, 0.0, 0.0
        ee = np.interp(tt[ok], qtk, q["e"]) - qx[ok]
        nn = np.interp(tt[ok], qtk, q["n"]) - qy[ok]
        oe, on = np.median(ee), np.median(nn)
        return float(np.sqrt(np.mean((ee - oe) ** 2 + (nn - on) ** 2))), ok, oe, on

    grid = np.arange(-CLOCK_SEARCH, CLOCK_SEARCH, 0.02)
    costs = np.array([overlay(x)[0] for x in grid])
    if not np.isfinite(costs).any():
        _log(verbose, f"RTK truth: {_short(fleet_path)} does not overlap this log.")
        return None
    from scipy.optimize import minimize_scalar
    x0 = grid[int(np.argmin(costs))]
    dt = float(minimize_scalar(lambda x: overlay(x)[0], bounds=(x0 - 0.03, x0 + 0.03),
                               method="bounded").x)
    rms, ok, home_e, home_n = overlay(dt)
    excursion = np.hypot(np.ptp(qx[ok]), np.ptp(qy[ok]))
    if excursion < 2.0:
        _log(verbose, "RTK truth: WARNING the quad barely moved - clock offset unobservable.")
    if rms > ALIGN_RMS_WARN:
        _log(verbose, f"RTK truth: WARNING quad tracks overlay with RMS {rms:.2f} m "
                      f"(> {ALIGN_RMS_WARN} m) - check the fleet file matches this flight.")

    tt = tc + dt
    valid = _gap_mask(c["t"].to_numpy(), tt) & _gap_mask(qtk, tt)

    # ── Quad (in the controller's local frame) ──
    q_e = _hermite(qtk, q["e"].to_numpy(), q["ve"].to_numpy(), tt) - home_e
    q_n = _hermite(qtk, q["n"].to_numpy(), q["vn"].to_numpy(), tt) - home_n
    q_z = np.interp(tt, qtk, q["rel_alt_m"])
    q_v = [np.interp(tt, qtk, q[k]) for k in ("ve", "vn", "vu")]

    # ── Car heading, then the car origin from its antenna ──
    ctk = c["t"].to_numpy()
    yaw_k, compass_bias, moving = _car_yaw_samples(c)
    car_yaw = _wrap(np.interp(tt, ctk, yaw_k))
    ant = PLATFORMS["real"]["car_antenna"]
    cy, sy = np.cos(car_yaw), np.sin(car_yaw)
    a_e = _hermite(ctk, c["e"].to_numpy(), c["ve"].to_numpy(), tt) - home_e
    a_n = _hermite(ctk, c["n"].to_numpy(), c["vn"].to_numpy(), tt) - home_n
    c_v = [np.interp(tt, ctk, c[k]) for k in ("ve", "vn")]
    sig = np.sqrt(np.interp(tt, ctk, c["h_acc_m"]) ** 2 + np.interp(tt, qtk, q["h_acc_m"]) ** 2)

    def fill(bias):
        org_e = a_e - bias[0] - (cy * ant[0] - sy * ant[1])
        org_n = a_n - bias[1] - (sy * ant[0] + cy * ant[1])
        cols = {
            "quad_true_x": q_e, "quad_true_y": q_n, "quad_true_z": q_z,
            "quad_true_vx": q_v[0], "quad_true_vy": q_v[1], "quad_true_vz": q_v[2],
            "quad_true_yaw": df["quad_yaw"].to_numpy(float),   # FCU yaw: no better source
            "landing_pad_glob_true_x": org_e, "landing_pad_glob_true_y": org_n,
            "landing_pad_glob_true_z": np.full(len(df), GROUND_Z),
            "landing_pad_glob_true_vx": c_v[0], "landing_pad_glob_true_vy": c_v[1],
            "landing_pad_glob_true_vz": np.zeros(len(df)),
            "landing_pad_glob_true_yaw": car_yaw,
        }
        cols["landing_pad_rel_true_x"] = org_e - q_e
        cols["landing_pad_rel_true_y"] = org_n - q_n
        cols["landing_pad_rel_true_z"] = GROUND_Z - q_z
        cols["landing_pad_rel_true_vx"] = c_v[0] - q_v[0]
        cols["landing_pad_rel_true_vy"] = c_v[1] - q_v[1]
        cols["landing_pad_rel_true_vz"] = -q_v[2]
        cols["landing_pad_rel_true_yaw"] = _wrap(car_yaw - cols["quad_true_yaw"])
        cols["true_height"] = GROUND_Z - q_z
        cols["truth_sigma_xy"] = sig
        for k, v in cols.items():
            df[k] = np.where(valid, v, np.nan)

    bias_auto = isinstance(rtk_bias, str) and rtk_bias.lower() == "auto"
    bias = (0.0, 0.0) if (rtk_bias is None or bias_auto) else tuple(map(float, rtk_bias))
    fill(bias)
    if bias_auto:
        est = estimate_vision_bias(df)
        if est is None:
            _log(verbose, "RTK truth: 'auto' bias needs YOLO frames in pad lock - none, bias 0.")
        else:
            bias = est
            fill(bias)

    _offline_nees(df)

    cov = c[(c["t"] >= tt[valid].min()) & (c["t"] <= tt[valid].max())] if valid.any() else c
    info = dict(
        fleet=os.path.abspath(fleet_path), quad_name=quad_name,
        clock_offset=dt, align_rms=rms, home_offset=(home_e, home_n),
        bias=bias, bias_auto=bias_auto and bias != (0.0, 0.0),
        compass_bias_deg=float(np.degrees(compass_bias)),
        coverage=(float(tc[valid].min() - tc[0]), float(tc[valid].max() - tc[0]))
        if valid.any() else (np.nan, np.nan),
        car_fix=cov["fix_name"].value_counts().to_dict() if "fix_name" in cov else {},
        quad_fix=q["fix_name"].value_counts().to_dict() if "fix_name" in q else {},
        truth_sigma_med=float(np.nanmedian(sig[valid])) if valid.any() else np.nan,
        car_clock_slope=float(c["clock_slope"].iloc[0]),
    )
    if verbose:
        print(f"RTK truth: {_short(fleet_path)}  quad='{quad_name}'")
        print(f"  Pi clock -> GPS time offset {dt:+.3f} s, quad-track overlay RMS {rms:.3f} m, "
              f"FCU home at ENU ({home_e:+.2f}, {home_n:+.2f}) m from the base")
        print(f"  truth covers t = {info['coverage'][0]:.1f} - {info['coverage'][1]:.1f} s "
              f"of the log; car fix {info['car_fix']}, quad fix {info['quad_fix']}")
        print(f"  car compass bias {info['compass_bias_deg']:+.1f} deg vs course (corrected), "
              f"car FCU clock rate {info['car_clock_slope']:.5f} s/s")
        print(f"  reported truth 1-sigma (car + quad h_acc) median {info['truth_sigma_med']:.2f} m")
        tag = " (AUTO: vision-referenced, absolute error no longer meaningful)" \
            if info["bias_auto"] else ""
        print(f"  RTK frame bias removed: E {bias[0]:+.2f} m, N {bias[1]:+.2f} m{tag}")
    return info


def estimate_vision_bias(df):
    """ Median (filter at YOLO image stamp - RTK pad truth) over pad lock / descent, in
        ENU (m). Assumes the vision is unbiased on average: diagnostic use only. """
    need = ["yolo_post_rel_x", "yolo_post_rel_y", "landing_pad_yolo_car_raw_stamp"]
    if not all(k in df.columns for k in need):
        return None
    s = np.nan_to_num(df["landing_pad_yolo_car_raw_stamp"].to_numpy(float))
    new = (s > 0) & np.r_[True, s[1:] != s[:-1]]
    grp = pd.to_numeric(df["controller_state"], errors="coerce").to_numpy() // 1000
    ok = new & (grp >= 3) & (grp <= 4) & np.isfinite(df["yolo_post_rel_x"].to_numpy(float))
    if ok.sum() < 10:
        return None
    t = df["timestamp"].to_numpy(float)
    px, py, _ = PLATFORMS["real"]["pad_in_car"]
    yaw = np.interp(s[ok], t, np.unwrap(np.nan_to_num(df["landing_pad_glob_true_yaw"])))
    tx = np.interp(s[ok], t, df["landing_pad_rel_true_x"]) + np.cos(yaw) * px - np.sin(yaw) * py
    ty = np.interp(s[ok], t, df["landing_pad_rel_true_y"]) + np.sin(yaw) * px + np.cos(yaw) * py
    ex = df["yolo_post_rel_x"].to_numpy(float)[ok] - tx
    ey = df["yolo_post_rel_y"].to_numpy(float)[ok] - ty
    good = np.isfinite(ex) & np.isfinite(ey)
    if good.sum() < 10:
        return None
    # The filter sees pad = car + bias_err, so the car track moves by -(median error)
    return (-float(np.median(ex[good])), -float(np.median(ey[good])))


def _offline_nees(df):
    """ NEES over (px, py, yaw) as the controller computes it in SITL, from the logged
        covariance (sigmas + off-diagonals). Only filled when the column is all NaN. """
    need = ["sigma_px", "sigma_py", "sigma_yaw", "P_px_py", "P_px_yaw", "P_py_yaw",
            "landing_pad_rel_x", "landing_pad_rel_y", "landing_pad_glob_yaw"]
    if _finite_any(df, "nees") or not all(k in df.columns for k in need):
        return
    px, py, _ = PLATFORMS["real"]["pad_in_car"]
    yaw_t = df["landing_pad_glob_true_yaw"].to_numpy(float)
    c, s = np.cos(yaw_t), np.sin(yaw_t)
    e = np.stack([
        df["landing_pad_rel_true_x"].to_numpy(float) + c * px - s * py
        - df["landing_pad_rel_x"].to_numpy(float),
        df["landing_pad_rel_true_y"].to_numpy(float) + s * px + c * py
        - df["landing_pad_rel_y"].to_numpy(float),
        _wrap(yaw_t - df["landing_pad_glob_yaw"].to_numpy(float)),
    ], axis=1)
    sx, sy_, sw = (df[k].to_numpy(float) for k in ("sigma_px", "sigma_py", "sigma_yaw"))
    P = np.zeros((len(df), 3, 3))
    P[:, 0, 0], P[:, 1, 1], P[:, 2, 2] = sx ** 2, sy_ ** 2, sw ** 2
    P[:, 0, 1] = P[:, 1, 0] = df["P_px_py"].to_numpy(float)
    P[:, 0, 2] = P[:, 2, 0] = df["P_px_yaw"].to_numpy(float)
    P[:, 1, 2] = P[:, 2, 1] = df["P_py_yaw"].to_numpy(float)
    nees = np.full(len(df), np.nan)
    ok = np.isfinite(e).all(axis=1) & np.isfinite(P).all(axis=(1, 2))
    seeded = ok & (sx < 0.999)              # skip the un-seeded placeholder state (P0)
    for i in np.flatnonzero(seeded):
        try:
            nees[i] = float(e[i] @ np.linalg.solve(P[i], e[i]))
        except np.linalg.LinAlgError:
            pass
    df["nees"] = nees


def _add_quad_ref(df, has_truth):
    """ quad_ref_*: truth where present, otherwise the FCU estimate. """
    for a in ("x", "y", "z", "vx", "vy", "vz", "yaw"):
        est = df[f"quad_{a}"].to_numpy(float)
        tru = df[f"quad_true_{a}"].to_numpy(float) if has_truth else np.full(len(df), np.nan)
        df[f"quad_ref_{a}"] = np.where(np.isfinite(tru), tru, est)
    df.attrs["quad_ref_label"] = "Quadcopter" if has_truth else "Quadcopter (FCU estimate)"