#!/usr/bin/env python3
"""
Landing-pad drift check.

The drone hovers over the car's landing pad, matched to the car's velocity at
t=0, then holds that velocity for a window DT. Worst case, the car brakes /
accelerates and/or turns hard during that window, so the pad ends up displaced
from the drone. We draw the pad (a rectangle) at each extreme displacement; the
region common to ALL of them is the part of the pad guaranteed to be under the
drone. DT is valid if the landing-gear footprint fits inside that common region.

Edit the constants below, then run:  python pad_drift.py

Frame: origin = drone / quad centre, x = along the car's heading at t=0,
y = lateral (left +).  (The diagram is drawn rotated so the heading points up.) Pad displacement relative to the drone:

    dx = 0.5 * a_max * DT^2                    (brake / accelerate)
    dy = R * (1 - cos(omega_max * DT))         (full-lock turn, R = v / omega_max)

with the max turn rate a function of speed:

    a_lat(v)   = min( v^2 * tan(delta_max) / L ,  A_LAT_MAX )
    omega_max  = a_lat(v) / v
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# ==========================================================================
# CONFIG - edit these
# ==========================================================================
# --- Calculator inputs ---
A_MAX = 9.81/2           # car max acceleration / braking, m/s^2
SPEED = 10.0             # current car speed, m/s
DT = 0.25                # window to test, s

# --- Pad / landing gear ---
BOX_SIZE = (2.0, 1.2)   # landing pad dimensions (along heading, lateral), m
GEAR_SIZE = (0.3, 0.3)  # landing-gear footprint (along heading, lateral), m

# --- Extras ---
DV_ERROR = 0.0          # velocity estimate error, m/s (adds dv*DT to each axis)
INCLUDE_YAW = True     # also rotate the pad by omega_max*DT in the turning cases
DESCENT_SPEED = 1.0     # drone descent speed, m/s (None to skip max-height calc)

# --- Height / reverse-thrust model ---
MAX_DOWN_THRUST = -30.0   # N, max downward (reverse) thrust. Negative = downward.
DRONE_MASS = 1.5          # kg
INITIAL_VELOCITY = 1.0    # m/s, downward speed at the start of the window (positive = down)
INCLUDE_GRAVITY = False   # False: a_down = -F/m  (F treated as the net vertical force)
                          # True:  a_down = g - F/m
VEHICLE_SIZE = (4.5, 1.8) # vehicle (length, width) that must fit in frame, m
FRAME_BUFFER = 0.25       # extra room around the vehicle (0.25 -> 25% bigger in each dimension)
CAMERA_RES = (1080, 1920) # px; only the 16:9 aspect ratio matters. Camera looks straight down,
                          # image long side along the vehicle's long side.

# --- Car model ---
WHEELBASE = 2.7         # L, m
MAX_STEER_DEG = 30.0    # max steering angle, deg
A_LAT_MAX = 5.0         # tyre-grip lateral accel limit (~mu*g), m/s^2

# --- Output ---
script_dir = Path(__file__).parent
out_dir = script_dir / 'plots'
SAVE_PATH = out_dir / 'pad_diagram.png'       # e.g. "pad_diagram.png" to save the pad-extremes diagram, else None
SAVE_PATH_PLOTS = out_dir / 'pad_plots.png'   # e.g. "pad_plots.png" to save the 2x2 summary plots, else None
SAVE_PATH_HEIGHT = out_dir / 'pad_height.png' # e.g. "pad_height.png" to save the height/FOV plots, else None

SHOW_PLOT = True        # open a plot window
# ==========================================================================

# --------------------------------------------------------------------------
# Vehicle model
# --------------------------------------------------------------------------
@dataclass
class CarModel:
    wheelbase: float = WHEELBASE
    max_steer_deg: float = MAX_STEER_DEG
    a_lat_max: float = A_LAT_MAX

    def lateral_accel_limit(self, v):
        """Max lateral acceleration available at speed v (m/s^2)."""
        v = np.asarray(v, dtype=float)
        steer_limited = v**2 * np.tan(np.radians(self.max_steer_deg)) / self.wheelbase
        return np.minimum(steer_limited, self.a_lat_max)

    def omega_max(self, v):
        """Max turn rate at speed v (rad/s). Zero at standstill."""
        v = np.asarray(v, dtype=float)
        a_lat = self.lateral_accel_limit(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(v > 1e-6, a_lat / v, 0.0)
        return w


# --------------------------------------------------------------------------
# Drift model
# --------------------------------------------------------------------------
def drift_components(t, a_max, v, car, dv=0.0):
    """Worst-case pad displacement (dx, dy), both >= 0, relative to the drone."""
    t = np.asarray(t, dtype=float)
    omega = float(car.omega_max(v))

    dx = 0.5 * a_max * t**2 + dv * t
    if omega > 1e-9:
        R = v / omega
        dy = R * (1.0 - np.cos(omega * t)) + dv * t
    else:
        dy = dv * t
    return dx, dy


def max_dt(a_max, v, car, box_size=(1.0, 1.0), dv=0.0, t_horizon=10.0, n=200001):
    """
    Largest dt for which the pad CENTRE stays within the half-dimensions of the drone
    (i.e. the point-sized-gear limit). Returns (dt_max, limiting_axis).
    """
    half_x, half_y = np.asarray(box_size, dtype=float) / 2

    t = np.linspace(0.0, t_horizon, n)
    dx, dy = drift_components(t, a_max, v, car, dv)

    def first_cross(d, limit):
        idx = np.argmax(d > limit)
        return np.inf if d[idx] <= limit else t[idx]

    tx, ty = first_cross(dx, half_x), first_cross(dy, half_y)
    return (tx, "longitudinal") if tx <= ty else (ty, "lateral")


# --------------------------------------------------------------------------
# Small convex-polygon helpers (no external geometry dependency)
# --------------------------------------------------------------------------
def _rect(cx, cy, hx, hy, angle=0.0):
    """Rectangle (CCW vertices) centred at (cx, cy), rotated by angle about its centre."""
    pts = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    return pts @ np.array([[c, s], [-s, c]]) + [cx, cy]


def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _clip(subject, clip):
    """Sutherland-Hodgman: intersection of two convex CCW polygons."""
    out = [np.asarray(p) for p in subject]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, out = out, []
        if not inp:
            break

        def inside(p):
            return _cross(b - a, p - a) >= -1e-12

        def intersect(p, q):
            d1, d2 = q - p, b - a
            return p + (_cross(a - p, d2) / _cross(d1, d2)) * d1

        s = inp[-1]
        for e in inp:
            if inside(e):
                if not inside(s):
                    out.append(intersect(s, e))
                out.append(e)
            elif inside(s):
                out.append(intersect(s, e))
            s = e
    return np.array(out) if len(out) >= 3 else np.empty((0, 2))


def _contains(poly, pt, tol=1e-9):
    """Point-in-convex-CCW-polygon test."""
    if len(poly) < 3:
        return False
    return all(_cross(poly[(i + 1) % len(poly)] - poly[i], pt - poly[i]) >= -tol
               for i in range(len(poly)))


# --------------------------------------------------------------------------
# Pad-at-extremes geometry
# --------------------------------------------------------------------------
def pad_cases(dt, a_max, v, car, box_size, dv=0.0, include_yaw=False):
    """
    The pad placed at each extreme displacement (relative to the drone at origin).
    Returns list of (label, (cx, cy), polygon).
    """
    dx, dy = (float(d) for d in drift_components(dt, a_max, v, car, dv))
    psi = float(car.omega_max(v)) * dt if include_yaw else 0.0

    spec = [
        ("accelerate",         +dx, 0.0, 0.0),
        ("brake",              -dx, 0.0, 0.0),
        ("accelerate + left",  +dx, +dy, +psi),
        ("accelerate + right", +dx, -dy, -psi),
        ("brake + left",       -dx, +dy, +psi),
        ("brake + right",      -dx, -dy, -psi),
    ]
    half_x, half_y = np.asarray(box_size, dtype=float) / 2
    return [(lbl, (cx, cy), _rect(cx, cy, half_x, half_y, ang)) for lbl, cx, cy, ang in spec]


def common_region(cases):
    """Intersection of all the pad polygons (empty (0,2) array if none)."""
    poly = cases[0][2]
    for _, _, p in cases[1:]:
        poly = _clip(poly, p)
        if len(poly) < 3:
            return np.empty((0, 2))
    return poly


def gear_fits(dt, a_max, v, car, box_size, dv, gear_size, include_yaw=False):
    """True if the gear footprint (centred on the drone) lies inside the common region."""
    cases = pad_cases(dt, a_max, v, car, box_size, dv, include_yaw)
    common = common_region(cases)
    gx, gy = gear_size[0] / 2, gear_size[1] / 2
    corners = _rect(0, 0, gx, gy)
    return all(_contains(common, c) for c in corners), common


def max_dt_for_gear(a_max, v, car, box_size, dv, gear_size, include_yaw=False,
                    t_horizon=5.0, n=500):
    """Largest DT for which the gear still fits (coarse scan, then bisection)."""
    prev = 0.0
    for t in np.linspace(0, t_horizon, n + 1)[1:]:
        if not gear_fits(t, a_max, v, car, box_size, dv, gear_size, include_yaw)[0]:
            lo, hi = prev, t
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if gear_fits(mid, a_max, v, car, box_size, dv, gear_size, include_yaw)[0]:
                    lo = mid
                else:
                    hi = mid
            return lo
        prev = t
    return np.inf


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def make_plots(a_max, v, car, box_size, dv, save=None):
    """The original 2x2 summary: drift vs time, turn-rate model, and Δt_max sweeps."""
    dt_star, limiter = max_dt(a_max, v, car, box_size, dv)

    half_x, half_y = np.asarray(box_size, dtype=float) / 2

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Landing pad drift  |  a_max={a_max:g} m/s²,  v={v:g} m/s,  "
        f"box={box_size[0]:g}×{box_size[1]:g} m,  dv={dv:g} m/s",
        fontsize=12,
    )

    # (1) drift vs time at the operating point
    ax = axs[0, 0]
    t_plot = np.linspace(0, max(1.5, 1.6 * min(dt_star, 5)), 600)
    dx, dy = drift_components(t_plot, a_max, v, car, dv)
    ax.plot(t_plot, dx, label="longitudinal (brake/accel)")
    ax.plot(t_plot, dy, label="lateral (full turn)")


    ax.axhline(half_x, color="r", ls="--", label=f"longitudinal edge ±{half_x:g} m")
    ax.axhline(half_y, color="r", ls="--", label=f"lateral edge ±{half_y:g} m")
    if np.isfinite(dt_star):
        ax.axvline(dt_star, color="k", ls=":", label=f"Δt_max = {dt_star:.2f} s")
    ax.set_ylim(0, 3 * max(half_x, half_y))
    ax.set_xlabel("Δt (s)")
    ax.set_ylabel("worst-case pad drift (m)")
    ax.set_title("Drift vs time at operating point")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (2) turn-rate model vs speed
    ax = axs[0, 1]
    vs = np.linspace(0.01, max(30, 1.5 * v), 400)
    ax.plot(vs, car.omega_max(vs), color="tab:green")
    ax.plot([v], [car.omega_max(v)], "ro", label=f"ω_max({v:g}) = {float(car.omega_max(v)):.2f} rad/s")
    ax.set_xlabel("speed v (m/s)")
    ax.set_ylabel("max turn rate ω_max (rad/s)")
    ax.set_title("Turn-rate model: min(steering lock, tyre grip) / v")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (3) Δt_max vs a_max (at current speed)
    ax = axs[1, 0]
    a_vals = np.linspace(0.5, 10, 60)
    dts = np.array([max_dt(a, v, car, box_size, dv, t_horizon=5.0, n=20001)[0] for a in a_vals])
    ax.plot(a_vals, dts, label="Δt_max (exact, incl. lateral limit)")
    ax.plot(a_vals, np.sqrt(2 * half_x / a_vals), "--", alpha=0.6,
            label="longitudinal-only  √(2h/a)")
    a_lat = float(car.lateral_accel_limit(v))
    if a_lat > 0:
        ax.axhline(np.sqrt(2 * half_y / a_lat), color="tab:red", ls="--", alpha=0.6,
                   label=f"lateral cap (a_lat={a_lat:.1f} m/s²)")
    ax.plot([a_max], [dt_star], "ro")
    ax.set_xlabel("max acceleration a_max (m/s²)")
    ax.set_ylabel("Δt_max (s)")
    ax.set_title(f"Δt_max vs acceleration  (v = {v:g} m/s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (4) Δt_max vs speed (at current a_max)
    ax = axs[1, 1]
    v_vals = np.linspace(0.5, max(30, 1.5 * v), 60)
    dts_v = np.array([max_dt(a_max, vv, car, box_size, dv, t_horizon=5.0, n=20001)[0] for vv in v_vals])
    ax.plot(v_vals, dts_v)
    ax.plot([v], [dt_star], "ro", label=f"operating point ({limiter}-limited)")
    ax.set_xlabel("speed v (m/s)")
    ax.set_ylabel("Δt_max (s)")
    ax.set_title(f"Δt_max vs speed  (a_max = {a_max:g} m/s²)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        fig.savefig(save, dpi=150)
        print(f"Saved plot to {save}")
    return fig


def _view(pts):
    """Rotate model coords (x = heading, y = left) so the car heading points UP on screen.
    (x, y) -> (-y, x): screen-x is +right of the car, screen-y is along the heading."""
    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        return np.array([-pts[1], pts[0]])
    return np.column_stack([-pts[:, 1], pts[:, 0]])


def make_diagrams(a_max, v, car, box_size, dv, dt, gear_size, include_yaw=False,
                  save=None, show=True):
    dt_star, limiter = max_dt(a_max, v, car, box_size, dv)
    dt_gear = max_dt_for_gear(a_max, v, car, box_size, dv, gear_size, include_yaw)

    half_x, half_y = np.asarray(box_size, dtype=float) / 2

    cases = pad_cases(dt, a_max, v, car, box_size, dv, include_yaw)
    fits, common = gear_fits(dt, a_max, v, car, box_size, dv, gear_size, include_yaw)
    omega = float(car.omega_max(v))

    fig, ax = plt.subplots(figsize=(11, 8.5))
    verdict = "GEAR FITS  ✓  (Δt valid)" if fits else "GEAR DOES NOT FIT  ✗  (Δt too large)"
    fig.suptitle(
        f"Landing pad at extreme displacements  |  Δt={dt:g} s,  a_max={a_max:g} m/s²,  "
        f"v={v:g} m/s,  ω_max={omega:.2f} rad/s,  dv={dv:g} m/s"
        f"{',  yaw included' if include_yaw else ''}\n"
        f"pad {box_size[0]:g}×{box_size[1]:g} m,  gear {gear_size[0]:g}×{gear_size[1]:g} m  →  {verdict}",
        fontsize=11, color="tab:green" if fits else "tab:red",
    )

    # nominal pad (no displacement) for reference
    ax.add_patch(Polygon(_view(_rect(0, 0, half_x, half_y)), closed=True, fill=False,
                         ec="gray", ls=":", lw=1.2, label="nominal pad (Δt = 0)"))

    # pad at each extreme
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for (label, (cx, cy), poly), col in zip(cases, colors):
        ax.add_patch(Polygon(_view(poly), closed=True, fc=col, alpha=0.07, ec="none"))
        ax.add_patch(Polygon(_view(poly), closed=True, fill=False, ec=col, lw=1.6, label=label))
        sx, sy = _view([cx, cy])
        ax.plot(sx, sy, "x", color=col, ms=7)

    # region common to every case
    if len(common) >= 3:
        ax.add_patch(Polygon(_view(common), closed=True, fc="tab:green", alpha=0.35,
                             ec="tab:green", lw=1.5, label="common region (all cases)"))

    # landing gear footprint, centred on the drone
    gx, gy = gear_size[0] / 2, gear_size[1] / 2
    ax.add_patch(Polygon(_view(_rect(0, 0, gx, gy)), closed=True, fill=False, lw=2.5,
                         ec="darkgreen" if fits else "red", hatch="//",
                         label="landing gear footprint"))
    ax.plot(0, 0, "k+", ms=12, mew=2)
    ax.annotate("drone / quad centre", (0, 0), xytext=(6, -14), textcoords="offset points", fontsize=8)

    # car heading arrow (screen up), placed in axes coordinates
    ax.annotate("", xy=(0.06, 0.20), xytext=(0.06, 0.08), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.075, 0.14, "car heading", transform=ax.transAxes, fontsize=8, va="center")

    # axis limits from all geometry
    allp = _view(np.vstack([c[2] for c in cases]))
    m = 0.2 * max(half_x, half_y)
    ax.set_xlim(allp[:, 0].min() - m, allp[:, 0].max() + m)
    ax.set_ylim(allp[:, 1].min() - m, allp[:, 1].max() + m)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral offset relative to drone, right + (m)")
    ax.set_ylabel("along car heading, relative to drone (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    ax.text(1.01, 0.02, (
        f"pad shift at Δt:\n  |dx| = {abs(cases[0][1][0]):.3f} m  (along heading)\n"
        f"  |dy| = {abs(cases[2][1][1]):.3f} m  (lateral)\n\n"
        f"common region (bbox):\n  {'%.3f (lat) × %.3f (long) m' % tuple(np.ptp(common, axis=0)[::-1]) if len(common) else 'empty'}\n\n"
        f"limits on Δt:\n  gear fits:   {dt_gear:.2f} s\n  point gear:  {dt_star:.2f} s ({limiter})"),
        transform=ax.transAxes, fontsize=8, va="bottom", family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save}")
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------
# Height / reverse-thrust model
# --------------------------------------------------------------------------
G = 9.81  # m/s^2


def height_model(force, dt, mass, v0, vehicle_size, buffer, cam_res, include_gravity=False):
    """
    How high above the pad could the drone start and still reach it within dt,
    and what camera FOV is then needed to see the whole vehicle from that height?

    force : vertical thrust, N (negative = downward). Scalar or array.
    dt    : time available (s) - the max window from the pad-extremes diagram.
    v0    : initial DOWNWARD speed (m/s, positive = down).

        a_down = -F/m  (+ g if include_gravity)
        H      = v0*dt + 0.5*a_down*dt^2          (height above the pad)
        v_td   = v0 + a_down*dt                   (speed on reaching the pad)

    FOV: a downward-looking camera at height H sees a footprint of width
    2*H*tan(HFOV/2). The vehicle (x (1+buffer)) must fit inside the 16:9 frame,
    long side along the image width:
        w_req = (1+buffer) * max(L, W*aspect),   h_req = w_req / aspect
    Returns a dict of arrays: a_down, H, v_touch, w_req, h_req, hfov, vfov, dfov (deg).
    """
    F = np.asarray(force, dtype=float)
    a_down = -F / mass + (G if include_gravity else 0.0)
    H = v0 * dt + 0.5 * a_down * dt**2
    v_touch = v0 + a_down * dt

    aspect = cam_res[0] / cam_res[1]
    L, W = vehicle_size
    w_req = (1.0 + buffer) * max(L, W * aspect)
    h_req = w_req / aspect
    d_req = np.hypot(w_req, h_req)

    Hs = np.where(H > 0, H, np.nan)   # no meaningful FOV if we can't be above the pad
    fov = lambda size: 2.0 * np.degrees(np.arctan(size / (2.0 * Hs)))
    return dict(a_down=a_down, H=H, v_touch=v_touch, w_req=w_req, h_req=h_req,
                hfov=fov(w_req), vfov=fov(h_req), dfov=fov(d_req))


def make_height_plots(dt, mass, v0, f_max, vehicle_size, buffer, cam_res,
                      include_gravity=False, save=None):
    """Two plots: thrust force vs max start height + impact speed, and thrust force vs required camera FOV."""
    end = 1.5 * f_max if f_max != 0 else -1.0
    forces = np.linspace(0.0, end, 200)
    res = height_model(forces, dt, mass, v0, vehicle_size, buffer, cam_res, include_gravity)
    op = height_model(f_max, dt, mass, v0, vehicle_size, buffer, cam_res, include_gravity)
    ref = height_model(0.0, dt, mass, v0, vehicle_size, buffer, cam_res, include_gravity)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f"Reverse-thrust height model  |  Δt = {dt:.3f} s (max from diagram),  m = {mass:g} kg,  "
        f"v0 = {v0:g} m/s down,  gravity {'included' if include_gravity else 'excluded'}\n"
        f"vehicle {vehicle_size[0]:g}×{vehicle_size[1]:g} m,  buffer {buffer*100:.0f}%,  "
        f"camera {cam_res[0]}×{cam_res[1]}",
        fontsize=10)

    # (1) force vs max start height (left axis) and impact speed (right axis)
    l1, = ax1.plot(forces, res["H"], color="tab:blue", label="max start height")
    l2 = ax1.axhline(float(ref["H"]), color="tab:blue", ls="--", lw=1, alpha=0.6,
                     label=f"height, no thrust (F = 0): {float(ref['H']):.2f} m")
    l3, = ax1.plot([f_max], [float(op["H"])], "o", color="tab:blue",
                   label=f"height at F = {f_max:g} N: {float(op['H']):.2f} m")
    ax1.set_xlabel("vertical thrust force F (N)   (negative = downward)")
    ax1.set_ylabel("max start height above pad (m)", color="tab:blue")
    ax1.tick_params(axis="y", colors="tab:blue")
    ax1.set_title("Thrust force vs max start height and impact speed")
    ax1.invert_xaxis()
    ax1.grid(alpha=0.3)

    ax1b = ax1.twinx()
    l4, = ax1b.plot(forces, res["v_touch"], color="tab:red", label="impact speed")
    l5 = ax1b.axhline(float(ref["v_touch"]), color="tab:red", ls="--", lw=1, alpha=0.6,
                      label=f"impact, no thrust (F = 0): {float(ref['v_touch']):.2f} m/s")
    l6, = ax1b.plot([f_max], [float(op["v_touch"])], "o", color="tab:red",
                    label=f"impact at F = {f_max:g} N: {float(op['v_touch']):.2f} m/s")
    ax1b.set_ylabel("impact speed at pad (m/s)", color="tab:red")
    ax1b.tick_params(axis="y", colors="tab:red")
    # both quantities are linear in F, so start each axis at zero to keep the two lines distinct
    ax1.set_ylim(bottom=0, top=1.08 * float(np.nanmax(res["H"])))
    ax1b.set_ylim(bottom=0, top=1.08 * float(np.nanmax(res["v_touch"])))
    ax1.legend(handles=[l1, l2, l3, l4, l5, l6], fontsize=7, loc="upper left")

    # (2) force vs required FOV
    ax2.plot(forces, res["hfov"], label="horizontal FOV")
    ax2.plot(forces, res["vfov"], label="vertical FOV")
    ax2.plot(forces, res["dfov"], label="diagonal FOV")
    ax2.plot([f_max] * 3, [float(op["hfov"]), float(op["vfov"]), float(op["dfov"])], "ro")
    ax2.annotate(f"H {float(op['hfov']):.0f}°  /  V {float(op['vfov']):.0f}°  /  D {float(op['dfov']):.0f}°",
                 (f_max, float(op["hfov"])), xytext=(-10, 10), textcoords="offset points",
                 fontsize=8, ha="right")
    ax2.set_xlabel("vertical thrust force F (N)   (negative = downward)")
    ax2.set_ylabel("required camera FOV at start height (deg)")
    ax2.set_title("Thrust force vs required FOV (vehicle + buffer fills frame)")
    ax2.invert_xaxis()
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save}")
    return fig


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    car = CarModel()
    omega = float(car.omega_max(SPEED))
    dx, dy = (float(d) for d in drift_components(DT, A_MAX, SPEED, car, DV_ERROR))
    fits, common = gear_fits(DT, A_MAX, SPEED, car, BOX_SIZE, DV_ERROR, GEAR_SIZE, INCLUDE_YAW)
    dt_star, limiter = max_dt(A_MAX, SPEED, car, BOX_SIZE, DV_ERROR)
    dt_gear = max_dt_for_gear(A_MAX, SPEED, car, BOX_SIZE, DV_ERROR, GEAR_SIZE, INCLUDE_YAW)

    print("=" * 60)
    print(f" a_max = {A_MAX:g} m/s^2    speed = {SPEED:g} m/s    Δt = {DT:g} s")
    print(f" ω_max(v) = {omega:.3f} rad/s ({np.degrees(omega):.1f} deg/s)"
          f"   lateral accel = {SPEED * omega:.2f} m/s^2")
    print(f" pad shift at Δt:  dx = {dx:.3f} m,  dy = {dy:.3f} m"
          f"{'  (+ yaw %.1f deg)' % np.degrees(omega * DT) if INCLUDE_YAW else ''}")
    print(f" pad {BOX_SIZE[0]:g}x{BOX_SIZE[1]:g} m, gear {GEAR_SIZE[0]:g}x{GEAR_SIZE[1]:g} m")
    if len(common):
        print(f" common region (bbox): {np.ptp(common[:, 0]):.3f} x {np.ptp(common[:, 1]):.3f} m")
    else:
        print(" common region: empty")
    print(f" gear fits at Δt = {DT:g} s?  {'YES' if fits else 'NO'}")
    print("-" * 60)
    print(f" max Δt, gear fits : {dt_gear:.3f} s")
    print(f" max Δt, point gear: {dt_star:.3f} s   [{limiter}-limited]")
    if DESCENT_SPEED:
        print(f" H_max (open-loop, {DESCENT_SPEED:g} m/s descent, gear-fit Δt) = {DESCENT_SPEED * dt_gear:.2f} m")
    print("=" * 60)

    # ---- height / reverse-thrust model, using the gear-fit Δt from the diagram ----
    if np.isfinite(dt_gear) and dt_gear > 0:
        args = (dt_gear, DRONE_MASS, INITIAL_VELOCITY, VEHICLE_SIZE, FRAME_BUFFER, CAMERA_RES, INCLUDE_GRAVITY)
        hm = height_model(MAX_DOWN_THRUST, *args)
        h0 = height_model(0.0, *args)
        print(f" HEIGHT MODEL   (Δt = {dt_gear:.3f} s, the gear-fit max from the diagram)")
        print(f" thrust F = {MAX_DOWN_THRUST:g} N, mass {DRONE_MASS:g} kg -> a_down = {float(hm['a_down']):.2f} m/s^2"
              f"  [gravity {'included' if INCLUDE_GRAVITY else 'excluded'}]")
        print(f" initial downward speed = {INITIAL_VELOCITY:g} m/s")
        print(f" max start height above pad: {float(hm['H']):.3f} m   (F = 0: {float(h0['H']):.3f} m)")
        print(f" speed on reaching the pad : {float(hm['v_touch']):.2f} m/s")
        print(f" vehicle {VEHICLE_SIZE[0]:g}x{VEHICLE_SIZE[1]:g} m + {FRAME_BUFFER*100:.0f}% buffer"
              f" -> frame must cover {float(hm['w_req']):.2f} x {float(hm['h_req']):.2f} m")
        print(f" required FOV at start height: H {float(hm['hfov']):.1f}°, V {float(hm['vfov']):.1f}°,"
              f" D {float(hm['dfov']):.1f}°   (F = 0: H {float(h0['hfov']):.1f}°)")
        print("=" * 60)
        make_height_plots(dt_gear, DRONE_MASS, INITIAL_VELOCITY, MAX_DOWN_THRUST, VEHICLE_SIZE,
                          FRAME_BUFFER, CAMERA_RES, INCLUDE_GRAVITY, save=SAVE_PATH_HEIGHT)
    else:
        print(" Height model skipped: gear-fit Δt is zero or unbounded.")

    make_plots(A_MAX, SPEED, car, BOX_SIZE, DV_ERROR, save=SAVE_PATH_PLOTS)
    make_diagrams(A_MAX, SPEED, car, BOX_SIZE, DV_ERROR, DT, GEAR_SIZE, INCLUDE_YAW,
                  save=SAVE_PATH, show=False)
    if SHOW_PLOT:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()