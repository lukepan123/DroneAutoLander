import numpy as np

from collections import deque
from collections import namedtuple
from scipy.optimize import least_squares

from .state_definitions import LP_State
from .state_definitions import LP_Meas

""" UKF Class definitions to implement a target/chaser kinematic model. Consists of all
    higher level UKF logic for prediction and updating of kinematic model, as well as
    UKF rollback for update measurements.

    Two measurement models are supported:
      - "pose" (cam is None): z = (px, py, pz, yaw), measured directly. Used by AprilTag
        and by the YOLO fallback (fitted position, plus the box-derived yaw once it
        can be resolved against the filter's heading, see fit_car_box).
      - "box"  (cam given):   z = (u_min, u_max, v_min, v_max), the YOLO bounding box
        edges in pixels. h(x) projects the car's shape (a union of boxes, at the sigma
        point's pad position AND yaw) through the image-time camera pose. The UKF therefore uses the current
        yaw estimate and propagates the yaw/position/height coupling through the
        covariance properly, instead of a yaw-agnostic fit upstream.
"""

class UKF:
    """ Defines the UKF class for the landing platform. Every predict() and every
        accepted update() call pushes one of these, in chronological order, allowing the
        filter to rewind to any point in the buffer window and *replay* history exactly
        rather than re-simulate over it. This is what lets multiple, independent
        measurement streams (e.g. AprilTag + YOLO) each perform out-of-sequence-
        measurement (OOSM) corrections without one stream's correction silently erasing
        the other's.

        kind == "predict": data = quad_vel used for that process step
        kind == "update":  data = _Meas(z, R, cam) used for that measurement correction.
                           For box measurements, cam holds the camera pose (attitude +
                           gimbal) at the image stamp, so a replay projects through the
                           exact geometry the image was taken with.

        x / P / X_prop are the UKF state, covariance, and propagated sigma
        points immediately AFTER this event was applied.
    """

    _Event = namedtuple("_Event", ["t", "kind", "x", "P", "X_prop", "data"])

    # cam = (R_level_cam (3x3), t_level_cam (3,), K (3x3), altitude) or None
    _Meas = namedtuple("_Meas", ["z", "R", "cam"])


    def __init__(self, P_diag_init, Q_diag_init,
                 alpha, beta, kappa,
                 mahalanobis_threshold,
                 init_pos, init_vel, init_yaw,
                 seed_window_n, seed_window_min_dt,
                 car_boxes, pad_position,
                 car_fit_max_rel_rms, car_roof_gain, rear_axle=None,
                 yaw_seed_var_max=None, buffer_window=0.5) -> None:

        """ Initialise the UKF. Preprocesses and computes the required sigma weights and
            stores them to reduce run-time computation.

        :param car_boxes:           Car shape as a union of boxes in the car model frame
                                    (x forward, y left, z up), flat list of 6 per box:
                                    x_min, x_max, y_min, y_max, z_min, z_max (m)
        :param pad_position:        Landing pad top-centre in the same frame (m). This is
                                    the point the filter tracks; its z is also the initial
                                    pad height above home altitude.
        :param car_fit_max_rel_rms: Max fallback box-fit residual, as a fraction of box size
        :param car_roof_gain:       Low-pass gain for the learned pad height (0 = fixed)
        :param yaw_seed_var_max:    Cap on the seeded yaw variance (rad^2), also used as
                                    the yaw variance when seeding can't find a heading.
                                    Keeps the yaw sigma points inside +/- pi (1.0 is
                                    safe here). None = original behaviour.
        :param buffer_window:       OOSM history length (s). Must comfortably exceed the
                                    worst-case image -> UKF latency, or late measurements
                                    are dropped as older than the buffer.
        """
        # ---- UKF PARAMETERS ----
        # Dimensions
        self.dim_x = len(LP_State)
        self.dim_z = len(LP_Meas)   # both models are 4-dimensional

        # UKF scaling parameters (Merwe)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Mahalanobis_threshold (prevent bad measurements going into filter)
        self._mahalanobis_threshold = mahalanobis_threshold
        self._rejected_measurements = 0

        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        self.gamma = np.sqrt(self.dim_x + self.lambda_)

        # Length of UKF Historical Buffer. Must comfortably exceed the WORST
        # CASE end-to-end latency of the slowest measurement stream feeding
        # this filter
        self._buffer_window = float(buffer_window)

        # OOSM rewind/replay. False = late measurements applied to the current state as
        # if fresh (the uncompensated baseline)
        self._oosm_enabled = True

        # Number of sigma points
        self.num_sigma = 2 * self.dim_x + 1

        # Preallocate sigma arrays
        self.X = np.zeros((self.num_sigma, self.dim_x))  # sigma points
        self.X_prop = np.zeros((self.num_sigma, self.dim_x))  # fx result
        self.Z = np.zeros((self.num_sigma, self.dim_z))  # hx result

        # Precompute weights
        self.Wm = np.full(self.num_sigma, 0.5 / (self.dim_x + self.lambda_))
        self.Wc = np.full(self.num_sigma, 0.5 / (self.dim_x + self.lambda_))

        self.Wm[0] = self.lambda_ / (self.dim_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.dim_x + self.lambda_) + (
            1 - self.alpha**2 + self.beta
        )

        # ---- CAR SHAPE MODEL ----
        # Union of boxes in the car frame. The state tracks the pad, so every corner is
        # stored relative to the pad: a corner in the level frame is pad + R(yaw) @ corner.
        boxes = np.asarray(car_boxes, dtype=float).ravel()
        pad = np.asarray(pad_position, dtype=float).ravel()
        if boxes.size == 0 or boxes.size % 6 != 0:
            raise ValueError("car_boxes must hold 6 values per box "
                             "(x_min, x_max, y_min, y_max, z_min, z_max)")
        if pad.size != 3:
            raise ValueError("pad_position must hold 3 values (x, y, z)")
        boxes = boxes.reshape(-1, 6)
        if np.any(boxes[:, 0::2] > boxes[:, 1::2]):
            raise ValueError("car_boxes: each min must be <= its max")
        corners = np.array([
            [x, y, z]
            for x0, x1, y0, y1, z0, z1 in boxes
            for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)
        ])
        self._car_corners = corners - pad                       # (8 * n_boxes, 3)

        # Centre of the car's overall extents, relative to the pad (fit initial guess)
        self._centre_from_pad = 0.5 * (corners.min(axis=0) + corners.max(axis=0)) - pad

        self._roof_agl = pad[2]   # pad height above home altitude, learned online,
                                  # persists across reset()
        self._car_fit_max_rel_rms = car_fit_max_rel_rms
        self._car_roof_gain = car_roof_gain
        self._yaw_seed_var_max = yaw_seed_var_max

        # Ackermann kinematics: only the rear axle moves along the heading. CTRA is run
        # at the axle; this is the pad's offset from it in the car frame (x fwd, y left).
        self.axle_to_pad = (np.zeros(2) if rear_axle is None
                            else pad[:2] - np.asarray(rear_axle, dtype=float)[:2])

        # ---- UKF INITIALISATION ----
        # Initial state
        self.x = np.zeros(self.dim_x)
        self.x[LP_State.PX]       = init_pos[0]
        self.x[LP_State.PY]       = init_pos[1]
        self.x[LP_State.PZ]       = init_pos[2]
        self.x[LP_State.V]        = np.hypot(init_vel[0], init_vel[1])
        self.x[LP_State.A]        = 0.0
        self.x[LP_State.YAW]      = init_yaw
        self.x[LP_State.YAW_RATE] = 0.0

        # Initial covariance
        self.P_init = np.diag(P_diag_init)
        self.P = self.P_init.copy()

        # Process noise
        self.Q = np.diag(Q_diag_init)
        self._q_pos_long  = 1e-4   # along-track noise density (m²/s) - small, v/a handle this
        self._q_pos_lat   = 1e-7   # cross-track noise density - make this very small
        self._q_pos_floor = 1e-6   # isotropic floor for quad_vel error (σ_quad² · dt)

        # Timestamp of the last predict/update cycle, initialised to None
        self._last_update_time: float | None = None

        # Seeding buffers/variables
        self._seed_buffers = {"apriltag": deque(), "yolo": deque()}
        self._seed_window_n = seed_window_n
        self._seed_window_min_dt = seed_window_min_dt
        self._seed_pending = False
        self._measurement_last_stamps = {k: None for k in self._seed_buffers}
        self._measurement_diagnostics = {k: {} for k in self._seed_buffers}
        self._reset_measurement_diagnostics()
        self._last_measurement_type = "yolo"

        # Most recent process input (quad_vel) seen by ANY predict() call.
        # Used as a fallback when an OOSM anchor happens to be an "update"
        # event (which carries no process input of its own) and we need
        # something to bridge the small residual gap to the OOSM's own
        # timestamp.
        self._last_quad_vel = np.zeros(3)

        # OOSM event-log buffer, ordered oldest -> newest by timestamp.
        self._UKF_buffer: deque["UKF._Event"] = deque()

        # Diagnostic variables
        self._last_nis = np.nan
        self.consecutive_rejections = 0   # live measurements gated out in a row (divergence check)


    def predict(self, quad_vel, timestamp: float, buffer: bool = True) -> None:
        """ Do prediction step for UKF up to `timestamp`. Calls fx_vectorised which is
            the kinematic process function for the filter. The step is always the gap
            since the filter's own last event, so live predicts, OOSM bridges and
            replays all share one notion of time.

        :param quad_vel:   quadcopter velocity (in LOCAL frame) (m/s)
        :param timestamp:  time to propagate to (s)
        :param buffer:     update buffer (bool)
        """
        if self._last_update_time is None:
            return
        dt = timestamp - self._last_update_time
        if dt <= 1e-9:
            return  # never propagate backwards

        # Guarantee P is symmetric positive definite before proceeding, repair if needed
        try:
            S = np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            self._repair_P()
            S = np.linalg.cholesky(self.P)

        n = self.dim_x

        # Cholesky of covariance
        S = self.gamma * S

        yaw0 = self.x[LP_State.YAW]          # at the top, before sigma points

        # Generate sigma points
        self.X[0] = self.x
        for i in range(n):
            col = S[:, i]
            self.X[i + 1] = self.x + col
            self.X[n + i + 1] = self.x - col

        # Propagate through fx
        self._fx_vectorized(self.X, self.X_prop, quad_vel, dt)

        # Predicted mean
        self.x[:] = (self.Wm[:, None] * self.X_prop).sum(axis=0)
        self.x[LP_State.YAW] = self._circular_mean(
            self.X_prop[:, LP_State.YAW], self.Wm
        )

        # Predicted covariance
        dX = self.X_prop - self.x
        dX[:, LP_State.YAW] = self._wrap(dX[:, LP_State.YAW])  # wrap yaw deviations

        self.P = (dX.T * self.Wc) @ dX + self._process_noise(dt, yaw0)
        self.P = 0.5 * (self.P + self.P.T)

        # Re-seed X_prop with sigma points from the UPDATED predicted covariance
        self._reseed_sigma_points()

        # Update timestamp/last-known process input, and buffer this event
        self._last_update_time = timestamp
        self._last_quad_vel = np.asarray(quad_vel, dtype=float).copy()
        if buffer:
            self._push_predict_event(timestamp, quad_vel)


    def forward_predict(self, quad_vel, dt: float) -> np.ndarray:
        """ Propagate the current state forward by dt using the UKF process model.
        Does not modify covariance, sigma-point buffers, timestamps, or the update buffer.

        :param quad_vel:   quadcopter velocity (in LOCAL frame) (m/s)
        :param dt:         timestep (s)
        :return: predicted state vector (dim_x,)
        """
        # Guarantee P is symmetric positive definite before proceeding, repair if needed
        try:
            S = np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            self._repair_P()
            S = np.linalg.cholesky(self.P)

        n = self.dim_x
        S = self.gamma * S

        # Generate sigma points into local scratch arrays (don't touch self.X)
        X = np.empty_like(self.X)
        X[0] = self.x
        for i in range(n):
            col = S[:, i]
            X[i + 1] = self.x + col
            X[n + i + 1] = self.x - col

        # Propagate through fx
        X_prop = np.empty_like(self.X_prop)
        self._fx_vectorized(X, X_prop, quad_vel, dt)

        # Predicted mean
        x_pred = (self.Wm[:, None] * X_prop).sum(axis=0)
        x_pred[LP_State.YAW] = self._circular_mean(X_prop[:, LP_State.YAW], self.Wm)

        return x_pred


    def update(
            self,
            measurement_type: str,
            stamp: float,
            z,
            R,
            now: float,
            cam=None,
            pipeline_timing=None,
            quad_vel=None,
    ) -> bool:
        """ Do update step for UKF. Manages higher level update functions such as
            checking the measurement timestamp and performing the UKF rewind and
            rollback. Safe to call from multiple independent measurement streams (e.g.
            AprilTag and YOLO) in any interleaving/order - out-of-sequence measurements
            are spliced into the correct chronological position and every event after
            that point (predicts AND updates, from every stream) is replayed against the
            corrected timeline.

        :param measurement_type: Stream name ("apriltag" or "yolo")
        :param stamp:           Measurement (image) timestamp (s)
        :param z:               Pose (px, py, pz, yaw) if cam is None, otherwise box
                                edges (u_min, u_max, v_min, v_max) in pixels
        :param R:               Measurement covariance (4x4)
        :param now:             Current timestamp (s)
        :param cam:             (R_level_cam, t_level_cam, K, altitude) at the image
                                stamp for box measurements, else None
        :param pipeline_timing: Optional Vector3Stamped containing pipeline timestamps
        :param quad_vel:        Drone velocity (local frame), stored with seed samples so
                                the seed can recover the car's own velocity
        :return: True on success, False on failure.
        """
        last_timestamp = self._measurement_last_stamps[measurement_type]

        # Ignore duplicate measurements, but don't error.
        if last_timestamp is not None and stamp == last_timestamp:
            return True

        # Newest timestamp is tracked per stream; older OOSMs don't move it backwards.
        if last_timestamp is None or stamp > last_timestamp:
            self._measurement_last_stamps[measurement_type] = stamp #type: ignore

        meas = self._Meas(
            np.asarray(z, dtype=float).copy(), np.asarray(R, dtype=float).copy(), cam
        )

        # Record raw measurement/latency for this stream before any update path can fail.
        self._record_measurement_diagnostics(
            measurement_type, meas, stamp, now, pipeline_timing
        )
        self._last_measurement_type = measurement_type

        # If seeding is required, run the seeding procedure and exit early. Only pose
        # measurements can seed - box measurements need a yaw prior to be meaningful.
        if self._seed_pending:
            if cam is not None:
                return False
            buf = self._seed_buffers[measurement_type]
            # Restart the window after a detection gap: an isolated early detection
            # (e.g. a partial box at the image edge) shouldn't anchor the velocity fit
            if buf and stamp - buf[-1][0] > 0.3:
                buf.clear()
            qv = None if quad_vel is None else np.asarray(quad_vel, dtype=float)[:2].copy()
            buf.append((stamp, meas.z, meas.R, qv))
            span = buf[-1][0] - buf[0][0]
            if len(buf) >= self._seed_window_n or (len(buf) >= 3 and span >= self._seed_window_min_dt):
                self.seed_from_window(list(buf))
                self._seed_pending = False
                for b in self._seed_buffers.values():
                    b.clear()
            return True

        # In-order path. With OOSM off, late measurements are applied to the current
        # state as if fresh (the uncompensated baseline), without rewinding the clock.
        if (not self._oosm_enabled or self._last_update_time is None
                or stamp >= self._last_update_time):
            self.predict(self._last_quad_vel, stamp)   # no-op if stamp is in the past
            accepted = self._update_apply(meas)
            if accepted:
                self._record_post_update(measurement_type)
                if self._last_update_time is None or stamp > self._last_update_time:
                    self._last_update_time = stamp       # never move the clock backwards
                self._push_update_event(self._last_update_time, meas)
            return accepted

        # ---- OOSM path ----
        if not self._UKF_buffer:
            return False

        buffer_copy = list(self._UKF_buffer)  # ordered oldest -> newest
        anchor_idx = None
        for i, ev in enumerate(buffer_copy):
            if ev.t <= stamp:
                anchor_idx = i
            else:
                break

        if anchor_idx is None:
            return False  # OOSM older than entire buffer, skip

        anchor = buffer_copy[anchor_idx]
        future_events = buffer_copy[anchor_idx + 1:]

        # Save full current state in case we need to bail out
        x_now, P_now, X_prop_now = self.x.copy(), self.P.copy(), self.X_prop.copy()
        t_now = self._last_update_time

        # Rewind to anchor
        self.x, self.P, self.X_prop = anchor.x.copy(), anchor.P.copy(), anchor.X_prop.copy()
        self._last_update_time = anchor.t

        # Bridge the small residual gap to the OOSM's own timestamp. If the
        # anchor itself is an "update" event it has no process input of its
        # own, so fall back to the nearest preceding predict's quad_vel.
        if stamp > anchor.t:
            bridge_quad_vel = next(
                (ev.data for ev in future_events if ev.kind == "predict"),
                self._nearest_quad_vel(buffer_copy, anchor_idx),  # fallback: no later predict
            )
            self.predict(bridge_quad_vel, stamp, buffer=False)

        accepted = self._update_apply(meas)

        # If the update failed, bail back to current state - real buffer untouched
        if not accepted:
            self.x, self.P, self.X_prop = x_now, P_now, X_prop_now
            self._last_update_time = t_now
            return False

        # x is at the image stamp here, before the replay moves it on to "now"
        self._record_post_update(measurement_type)

        # Only commit to the rewind now that we know it succeeded: prune
        # entries forward of the anchor from the real buffer, then rebuild
        # it by splicing this OOSM in and replaying every event that originally
        # came after it - from BOTH streams - in the order it actually
        # happened, instead of blindly re-predicting over lost corrections.
        while self._UKF_buffer and self._UKF_buffer[-1].t > anchor.t:
            self._UKF_buffer.pop()

        self._last_update_time = stamp
        self._push_update_event(stamp, meas)

        for ev in future_events:
            if ev.kind == "predict":
                self.predict(ev.data, ev.t)  # buffer=True re-pushes it
            else:  # "update" - replay the other stream's correction too
                if self._update_apply(ev.data, replay=True):
                    self._last_update_time = ev.t
                    self._push_update_event(ev.t, ev.data)
                # A failed replay (rare - singular innovation covariance) is
                # simply skipped: _update_apply never mutates state before it
                # can fail, so this is safe and just drops that one stale
                # correction rather than corrupting the timeline.

        return True


    def fit_car_box(self, box, cam, learn: bool = True, yaw_ref=None):
        """ Fallback for a YOLO box, used while the yaw estimate is too poor for the
            pixel-space update (e.g. during seeding). Returns a drone-relative pad
            position to feed through the "pose" model, plus (optionally) a car yaw.

            Intersect the box-centre ray with the plane at the car's mid-height, then
            refine by fitting the car shape (union of boxes) so that its projected
            bounding box matches the YOLO box. This stays valid for oblique views (foreshortening, visible roof +
            sides), unlike a size-based depth. The car's yaw relative to the camera
            heading is fitted too, with a weak prior towards "aligned". The box size also
            reveals if the pad is higher or lower than the learned pad height (car on a
            slope, altitude drift); that offset is low-passed into self._roof_agl.

            Yaw: a box only reveals |car yaw - camera heading| (delta). The candidates
            psi_cam +/- delta (mirror) and the same + pi (front/back) all give the same
            box, so they are resolved against the filter's current heading (yaw_ref).
            Front/back needs only the rough direction of travel. The mirror pair is
            hardest to separate when delta is small, which is exactly when choosing
            wrong costs least: if it can't be separated, the camera axis is returned
            with delta folded into the sigma.

        :param box:     (u_min, u_max, v_min, v_max) in pixels
        :param cam:     (R_level_cam, t_level_cam, K, altitude) at the image stamp
        :param yaw_ref: Optional (yaw, sigma_yaw, px_sigma): the filter's heading, its
                        1-sigma (rad), and the YOLO edge noise (px) used for the fitted
                        yaw's sigma. None skips the yaw extraction.
        :return: (pos, yaw, sigma_yaw), or None if the camera isn't looking toward the
                 ground. pos = [px, py, pz] (pad top-centre) in the local-level frame.
                 yaw/sigma_yaw are None when not requested or not resolvable.
        """
        R_lc, t_lc, K, altitude = cam
        box = np.asarray(box, dtype=float)
        cx, cy = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
        bw, bh = box[1] - box[0], box[3] - box[2]

        ray = R_lc @ np.linalg.solve(K, [cx, cy, 1.0])
        ray /= np.linalg.norm(ray)
        if ray[2] >= -1e-3:
            return None

        # Plane at the height of the centre of the car's extents
        dz_c = self._centre_from_pad[2]
        t = (self._roof_agl + dz_c - (altitude + t_lc[2])) / ray[2]
        if t <= 0:
            return None
        centre = t_lc + t * ray

        # Heading of the horizontal axis perpendicular to the image x-axis: the car is
        # assumed (weakly) to have its length along this.
        u_w = R_lc[:2, 0]
        if np.linalg.norm(u_w) < 0.2:  # camera x-axis ~vertical: heading undefined
            return centre - np.array([0.0, 0.0, dz_c]), None, None   # xy pad offset ignored
        psi_cam = np.arctan2(u_w[0], -u_w[1])

        # Initial pad position, assuming the car points along psi_cam
        c, s = np.cos(psi_cam), np.sin(psi_cam)
        dx, dy = self._centre_from_pad[:2]
        pad0 = centre - np.array([c * dx - s * dy, s * dx + c * dy, dz_c])

        k_dz = 3.0  # height prior (px per metre): weak in pass 1, ~pinned in pass 2
        yaw_prior_deg = 5.0  # yaw offset costing 1 px of residual (prior towards aligned)

        def residual(p):  # p = [pad x, pad y, yaw offset, height offset]
            edges = self._box_edges(
                np.array([[p[0], p[1], pad0[2] + p[3]]]), np.array([psi_cam + p[2]]), cam
            )[0]
            return np.r_[edges - box, np.degrees(p[2]) / yaw_prior_deg, k_dz * p[3]]

        def fit():
            # Fit from a few yaw guesses (one start gets stuck at "aligned"). The sign of
            # the yaw is not observable from a box, so blend the fits by likelihood (~2 px
            # edge noise) instead of picking one at random: ambiguous -> midpoint.
            fits = [least_squares(residual, [pad0[0], pad0[1], yaw0, 0.0], method="lm",
                                  max_nfev=25, ftol=1e-3, xtol=1e-3)
                    for yaw0 in (-0.3, 0.3)]
            cost = np.array([f.cost for f in fits])
            best = fits[cost.argmin()]
            w = np.exp(-(cost - cost.min()) / 4.0)
            rms = np.sqrt(np.mean(best.fun[:4] ** 2))
            return w @ np.array([f.x for f in fits]) / w.sum(), rms, best

        # Pass 1: let the height float to learn where the pad really is
        (x, y, _, dz), rms, _ = fit()
        if rms > max(self._car_fit_max_rel_rms * 0.5 * (bw + bh), 3.0):
            return pad0, None, None  # box doesn't fit the car model (bad yaw / bad detection)
        if learn:
            self._roof_agl += self._car_roof_gain * dz
        pad0 = np.array([x, y, pad0[2] + self._car_roof_gain * dz])

        # Pass 2: fit x, y with that height ~fixed (height and range are coupled in
        # oblique views, so a free height adds noise to x, y). When extracting yaw, the
        # prior is weakened so the fitted |yaw| isn't shrunk towards "aligned".
        k_dz = 100.0
        if yaw_ref is not None:
            yaw_prior_deg = 30.0
        (x, y, _, _), _, best = fit()
        pos = np.array([x, y, self._roof_agl - altitude])
        if yaw_ref is None:
            return pos, None, None

        # ---- Box yaw: |yaw offset| from the best single fit (the +/- blend above
        # collapses it to ~0), folded into [0, pi/2] since the car is ~symmetric ----
        yaw_f, sigma_f, px_sigma = yaw_ref
        delta = abs(self._wrap(best.x[2]))
        delta = min(delta, np.pi - delta)

        # 1-sigma of delta from Gauss-Newton on the pixel rows (x, y, yaw columns)
        J = best.jac[:4, :3]
        try:
            sigma_delta = px_sigma * np.sqrt(np.linalg.inv(J.T @ J)[2, 2])
        except np.linalg.LinAlgError:
            return pos, None, None
        if not np.isfinite(sigma_delta):
            return pos, None, None

        # Front/back: flip the camera axis if the car faces back along it
        k = 2.0  # sigmas of separation needed to commit to a hypothesis
        e = self._wrap(yaw_f - psi_cam)
        if abs(e) > np.pi / 2:
            psi_cam, e = self._wrap(psi_cam + np.pi), self._wrap(e - np.pi)
        if np.pi - 2 * abs(e) < k * sigma_f:
            return pos, None, None  # front/back unresolved

        # Mirror pair psi_cam +/- delta sits 2 * min(delta, |e|) apart as seen from yaw_f
        if 2 * min(delta, abs(e)) > k * sigma_f:
            return pos, self._wrap(psi_cam + np.sign(e) * delta), sigma_delta
        return pos, psi_cam, np.hypot(sigma_delta, delta)  # ambiguous: axis, honest sigma


    def reset(self) -> None:
        """Reset runtime state; leaves parameters, weights, preallocated sigma buffers
        and the learned roof height untouched.
        """
        self.x = np.zeros(self.dim_x)
        self.P = self.P_init.copy()
        self._last_update_time = None
        self._last_quad_vel = np.zeros(3)
        self._UKF_buffer.clear()

        self._seed_pending = True
        for b in self._seed_buffers.values():
            b.clear()
        self._measurement_last_stamps = {k: None for k in self._seed_buffers}
        self._last_measurement_type = "yolo"
        self._reset_measurement_diagnostics()

        self.X.fill(0.0)
        self.X_prop.fill(0.0)
        self.Z.fill(0.0)

        self._rejected_measurements = 0
        self._last_nis = np.nan
        self.consecutive_rejections = 0


    @property
    def seed_pending(self) -> bool:
        """ Return True while the initial measurement window is being collected. """
        return self._seed_pending


    def _reset_measurement_diagnostics(self) -> None:
        """ Reset the per-stream raw measurement/latency diagnostics. """
        for d in self._measurement_diagnostics.values():
            d.update({
                "raw": np.zeros(4),
                "box": False,
                "raw_stamp": 0.0,
                "meas_age": 0.0,
                "cam_to_image_lag": np.nan,
                "image_to_transform_lag": np.nan,
                "transform_to_UKF_lag": np.nan,
                "total_lag": np.nan,
                "post_x": np.full(3, np.nan),
            })


    def _record_measurement_diagnostics(
            self, measurement_type, meas, measurement_timestamp, now, pipeline_timing=None
    ) -> None:
        """ Store raw measurement and latency diagnostics for one measurement stream. """
        d = self._measurement_diagnostics[measurement_type]
        d["raw"] = meas.z.copy()
        d["box"] = meas.cam is not None
        d["raw_stamp"] = measurement_timestamp
        d["meas_age"] = (now - measurement_timestamp) * 1000.0
        d["cam_to_image_lag"] = np.nan
        d["image_to_transform_lag"] = np.nan
        d["transform_to_UKF_lag"] = np.nan
        d["total_lag"] = d["meas_age"]
        d["post_x"] = np.full(3, np.nan)   # filled by _record_post_update if accepted

        if pipeline_timing is not None:
            t1 = pipeline_timing.vector.x
            t2 = pipeline_timing.vector.y
            d["cam_to_image_lag"] = (t1 - measurement_timestamp) * 1000.0
            d["image_to_transform_lag"] = (t2 - t1) * 1000.0
            d["transform_to_UKF_lag"] = (now - t2) * 1000.0
            d["total_lag"] = (
                d["cam_to_image_lag"]
                + d["image_to_transform_lag"]
                + d["transform_to_UKF_lag"]
            )


    def _record_post_update(self, measurement_type) -> None:
        """ Store the filtered position at the measurement's own stamp. Call right
            after a successful _update_apply, before any OOSM replay moves x on.
        """
        self._measurement_diagnostics[measurement_type]["post_x"] = self.x[
            [LP_State.PX, LP_State.PY, LP_State.PZ]
        ].copy()


    def get_measurement_diagnostics(self) -> dict:
        """ Retrieve diagnostics associated with all measurement streams. """
        d = {
            k: {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
            for k, v in self._measurement_diagnostics.items()
        }
        d["latest"] = d[self._last_measurement_type]
        return d


    def seed_position(
        self, z, timestamp: float, R, ignore_above: float = 1e6
    ) -> None:
        """ Directly seed the position/yaw states (and their covariance) from a
            measurement, instead of letting them converge from zero via the
            normal Kalman gain over several update() calls. Call this once,
            right after reset(), as soon as the first valid measurement of any
            stream becomes available.

            Axes whose R is >= ignore_above are treated as carrying no real
            information for this stream (e.g. YOLO's yaw, R~1e9) and are
            left untouched - neither x nor P is modified for that axis. This
            matters because seeding P from a sentinel-large R would otherwise
            blow past covar_max_eig safety thresholds on a perfectly healthy
            filter. Those axes get filled in normally (via Kalman gain) the
            next time a stream that does trust them provides an update.

            Velocity/acceleration/yaw-rate are always left at reset() defaults
            since no stream measures them directly.

        :param z:            Measurement of landing pad pose (px, py, pz, yaw)
        :param timestamp:     Timestamp of the seeding measurement (s)
        :param R:             Measurement noise for this stream. Also used to decide
                              which axes to skip.
        :param ignore_above:  R[axis,axis] at or above this is treated as "no
                            information" and that axis is left unseeded.
        """
        z = np.asarray(z, dtype=float)

        pos_states = [LP_State.PX, LP_State.PY, LP_State.PZ, LP_State.YAW]
        meas_states = [LP_Meas.PX, LP_Meas.PY, LP_Meas.PZ, LP_Meas.YAW]

        seeded_any = False
        for xs, zs in zip(pos_states, meas_states):
            r = R[zs, zs]
            if r >= ignore_above:
                continue  # no real info on this axis for this stream - leave as-is

            if xs == LP_State.YAW:
                self.x[xs] = self._wrap(z[zs])
            else:
                self.x[xs] = z[zs]

            # Pad a bit above raw R since this is a single sample, not a
            # filtered estimate - avoids being overconfident off one measurement.
            self.P[xs, xs] = max(r * 4.0, 1e-4)
            seeded_any = True

        if not seeded_any:
            return  # nothing trustworthy in this measurement, don't bother logging an event

        self._last_update_time = timestamp

        # Re-seed X_prop from the (possibly partially) seeded x/P, so an OOSM
        # rewind that later lands exactly on this event sees sigma points
        # consistent with it, not stale pre-seed ones.
        self._reseed_sigma_points()

        self._push_update_event(timestamp, self._Meas(z, np.asarray(R, dtype=float), None))


    def seed_from_window(
        self, samples: list[tuple], ignore_above: float = 1e6
    ) -> None:
        """ Seed position, yaw, velocity AND yaw-rate from a short window of raw
            measurements, instead of a single sample. Differencing several noisy
            position/yaw samples via least-squares gives a much better initial
            velocity/yaw-rate estimate than starting from zero and letting the
            Kalman gain infer it recursively over several update() calls -
            useful when you have a guaranteed hold period (e.g. ~1s / 10
            samples) before you need a usable estimate.

            Call this once, right after reset(), in place of seed_position() -
            not in addition to it - once your window of samples is full.

        :param samples: (timestamp, z, R[, quad_vel]) tuples, oldest first, ALL FROM
                        THE SAME STREAM (mixing streams with different noise/rate
                        characteristics will bias the fit). z = (px,py,pz,yaw), relative
                        to the drone. quad_vel (drone xy velocity, or None) turns the
                        fitted relative velocity back into the car's own velocity.
        :param ignore_above: as per seed_position - axes with R >= this are
                            treated as unmeasured by this stream and skipped.
        """
        n = len(samples)
        if n < 3:
            t0, z0, R0 = samples[-1][:3]
            self.seed_position(z0, t0, R0, ignore_above)
            return

        t = np.array([s[0] for s in samples])
        Z = np.array([s[1] for s in samples])
        R_last = samples[-1][2]
        dt = t - t[0]  # regress on seconds-since-window-start, not raw epoch time

        seeded_any = False
        vx = vy = 0.0
        have_xy = False

        for xs, zs in [
            (LP_State.PX, LP_Meas.PX),
            (LP_State.PY, LP_Meas.PY),
            (LP_State.PZ, LP_Meas.PZ),
        ]:
            r = R_last[zs, zs]
            if r >= ignore_above:
                continue
            slope, intercept = np.polyfit(dt, Z[:, zs], 1)
            self.x[xs] = intercept + slope * dt[-1]     # fitted value "now" (denoised)
            self.P[xs, xs] = max(r * 4.0, 1e-4)         # same padding convention as seed_position
            if xs == LP_State.PX:
                vx, have_xy = slope, True
            elif xs == LP_State.PY:
                vy = slope
            seeded_any = True

        r_yaw = R_last[LP_Meas.YAW, LP_Meas.YAW]
        if r_yaw < ignore_above:
            yaw_unwrapped = np.unwrap(Z[:, LP_Meas.YAW])   # critical: unwrap before fitting
            slope_w, intercept_w = np.polyfit(dt, yaw_unwrapped, 1)
            self.x[LP_State.YAW] = self._wrap(intercept_w + slope_w * dt[-1])
            self.x[LP_State.YAW_RATE] = slope_w
            self.P[LP_State.YAW, LP_State.YAW] = max(r_yaw * 4.0, 1e-4)
            span = dt[-1] - dt[0]
            var_omega = 12.0 * r_yaw / max(n * (n**2 - 1) * (span / (n - 1)) ** 2, 1e-9)
            self.P[LP_State.YAW_RATE, LP_State.YAW_RATE] = max(4.0 * var_omega, 1e-3)
            seeded_any = True

        if have_xy:
            # The fit is to RELATIVE positions, so its slope is car minus drone velocity.
            # Add the drone's mean velocity over the window to get the car's own motion.
            qv = [smp[3] for smp in samples if len(smp) > 3 and smp[3] is not None]
            if qv:
                quad_v = np.mean(qv, axis=0)
                vx, vy = vx + quad_v[0], vy + quad_v[1]

            speed_mag = float(np.hypot(vx, vy))
            r_pos = max(R_last[LP_Meas.PX, LP_Meas.PX], R_last[LP_Meas.PY, LP_Meas.PY])
            span = dt[-1] - dt[0]
            var_v = 12.0 * r_pos / max(n * (n**2 - 1) * (span / (n - 1)) ** 2, 1e-9)

            yaw_was_seeded = R_last[LP_Meas.YAW, LP_Meas.YAW] < ignore_above

            if yaw_was_seeded:
                # This stream also measured yaw directly (e.g. AprilTag) - project
                # the fitted velocity vector onto that independently-measured heading.
                yaw0 = self.x[LP_State.YAW]
                self.x[LP_State.V] = vx * np.cos(yaw0) + vy * np.sin(yaw0)
                self.P[LP_State.V, LP_State.V] = max(4.0 * var_v, 1e-3)
            else:
                # No yaw from this stream (YOLO fallback: R_yaw ~ 1e9). Derive heading
                # AND speed from the direction of travel - valid for a target that
                # points the way it moves, but only trustworthy if it's actually
                # moving: near-zero displacement gives a heading dominated by
                # position noise, not motion.
                MIN_SPEED_FOR_HEADING = 0.3  # m/s - tune against your position noise floor
                if speed_mag >= MIN_SPEED_FOR_HEADING:
                    self.x[LP_State.YAW] = self._wrap(np.arctan2(vy, vx))
                    self.x[LP_State.V] = speed_mag
                    # sigma_yaw ~ sigma_v / speed via atan2 error propagation -
                    # correctly blows up as speed -> 0.
                    var_yaw = var_v / max(speed_mag**2, 1e-6)
                    self.P[LP_State.YAW, LP_State.YAW] = max(4.0 * var_yaw, 1e-3)
                    # Optional cap (yaw_seed_var_max): the yaw sigma points sit at
                    # +/- gamma*sigma (gamma = sqrt(7) here), and beyond ~1.2 rad they wrap
                    # past +/- pi, so the next predict COLLAPSES P_yaw (e.g. 2.5 -> 0.15
                    # rad) and the filter claims confidence in an arbitrary heading.
                    if self._yaw_seed_var_max is not None:
                        self.P[LP_State.YAW, LP_State.YAW] = min(
                            self.P[LP_State.YAW, LP_State.YAW], self._yaw_seed_var_max)
                    self.P[LP_State.V, LP_State.V] = max(4.0 * var_v, 1e-3)
                elif self._yaw_seed_var_max is not None:
                    # Can't distinguish "barely moving" from "stationary" given this
                    # stream's noise floor, so don't force a heading out of it. Leave
                    # x[V]/x[YAW] at reset() defaults, but make the yaw variance honest:
                    # yaw = 0 is arbitrary, not a 0.45 rad (P_init) estimate.
                    self.P[LP_State.YAW, LP_State.YAW] = self._yaw_seed_var_max
                # else: leave x[V]/x[YAW]/P at reset() defaults (original behaviour)

        if not seeded_any:
            return

        self._last_update_time = t[-1]
        self._reseed_sigma_points()

        self._UKF_buffer.clear() # Clear buffer incase any predicts were called prior (shouldnt be)
        self._push_update_event(t[-1], self._Meas(samples[-1][1], R_last, None))


    def _update_apply(self, meas, replay: bool = False) -> bool:
        """ Do update step for UKF. Handles lower level UKF update functions such as
            actual covariance and state updates. Called via the public .update()
            function (both the in-order path and OOSM replay).

        :param meas:   _Meas(z, R, cam). cam None -> pose model, else box model. R is
                       stored per event so replays use the R that was actually in
                       effect when they first happened.
        :param replay: True when re-applying a buffered event during OOSM replay
        :return: True on success, False on failure.
        """
        box = meas.cam is not None

        # Propagate sigma points through the measurement model
        if box:
            pad = self.X_prop[:, [LP_State.PX, LP_State.PY, LP_State.PZ]]
            self.Z[:] = self._box_edges(pad, self.X_prop[:, LP_State.YAW], meas.cam)
        else:
            self._hx_vectorized(self.X_prop, self.Z)

        # Predicted measurement mean, deviations and innovation (yaw is circular)
        z_pred = (self.Wm[:, None] * self.Z).sum(axis=0)
        if not box:
            z_pred[LP_Meas.YAW] = self._circular_mean(self.Z[:, LP_Meas.YAW], self.Wm)
        dZ = self.Z - z_pred
        y = meas.z - z_pred
        if not box:
            dZ[:, LP_Meas.YAW] = self._wrap(dZ[:, LP_Meas.YAW])
            y[LP_Meas.YAW] = self._wrap(y[LP_Meas.YAW])

        # State deviations
        dX = self.X_prop - self.x
        dX[:, LP_State.YAW] = self._wrap(dX[:, LP_State.YAW])

        # Cross covariance
        P_xz = (dX.T * self.Wc) @ dZ

        # Innovation covariance
        S = (dZ.T * self.Wc) @ dZ + meas.R

        # Mahalanobis / NIS measurement gate
        try:
            # NIS = y^T S^-1 y
            nis = float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            return False

        if not replay:
            self._last_nis = nis

        # Reject statistically implausible measurements BEFORE updating
        # state or covariance.
        if nis > self._mahalanobis_threshold:
            if not replay:
                self._rejected_measurements += 1
                self.consecutive_rejections += 1
            return False

        # Kalman gain, if S is singular skip update
        try:
            K = np.linalg.solve(S.T, P_xz.T).T
        except np.linalg.LinAlgError:
            return False

        # Measurement passed the gate: the filter is still tracking
        if not replay:
            self.consecutive_rejections = 0

        # Update state
        self.x += K @ y
        self.x[LP_State.YAW] = self._wrap(self.x[LP_State.YAW])

        # Covariance update
        self.P -= K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Re-seed sigma points from the posterior
        self._reseed_sigma_points()

        # Keep the fallback fit's roof height in step with the filtered estimate
        # (x is at the image stamp here, matching the stored altitude)
        if box and not replay:
            self._roof_agl += self._car_roof_gain * (
                self.x[LP_State.PZ] + meas.cam[3] - self._roof_agl
            )

        return True


    def _box_edges(self, pad, yaw, cam) -> np.ndarray:
        """ Project the car shape (union of boxes) into the image and return its
            overall bounding box.

        :param pad: (N, 3) landing pad positions, drone-relative, local-level frame
        :param yaw: (N,) car headings in the local-level frame
        :param cam: (R_level_cam, t_level_cam, K, altitude) at the image stamp
        :return: (N, 4) [u_min, u_max, v_min, v_max] in pixels
        """
        R_lc, t_lc, K, _ = cam
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        fwd, left, up = self._car_corners.T

        # Corners in the level frame: forward along (c, s), left along (-s, c)
        pts = np.stack([
            pad[:, None, 0] + c * fwd - s * left,
            pad[:, None, 1] + s * fwd + c * left,
            pad[:, None, 2] + up,
        ], axis=-1)                                         # (N, M, 3)

        uv = ((pts - t_lc) @ R_lc) @ K.T                    # camera frame -> pixels
        uv = uv[..., :2] / np.maximum(uv[..., 2:], 0.1)
        return np.stack([
            uv[..., 0].min(axis=1), uv[..., 0].max(axis=1),
            uv[..., 1].min(axis=1), uv[..., 1].max(axis=1),
        ], axis=1)


    def _process_noise(self, dt: float, yaw: float) -> np.ndarray:
        Q = self.Q * dt
        idx = [LP_State.PX, LP_State.PY]

        c, s = np.cos(yaw), np.sin(yaw)
        Rm = np.array([[c, -s], [s, c]])
        Q_pos = Rm @ np.diag([self._q_pos_long * dt, self._q_pos_lat * dt]) @ Rm.T
        Q_pos += np.eye(2) * self._q_pos_floor * dt

        Q[np.ix_(idx, idx)] = Q_pos      # overwrite the isotropic block
        return Q


    def get_covar_diagnostics(self) -> dict:
        """ Return per-state 1-sigma values and scalar health metrics.

        Returns a dict with:
            sigma_<state_name>  - 1-sigma (sqrt of diagonal variance) for each state
            covar_trace         - sum of all diagonal variances (scalar health metric)
            covar_det_log       - log-determinant (overall uncertainty volume)
            covar_max_eig       - largest eigenvalue (worst-case direction)
            is_pd               - True if P is positive definite (Cholesky succeeds)
        """
        diag = np.diag(self.P)

        # Clamp negatives defensively before sqrt (shouldn't happen after _repair_P)
        sigmas = np.sqrt(np.maximum(diag, 0.0))

        # Scalar metrics
        trace = float(np.trace(self.P))
        max_eig = float(np.max(np.linalg.eigvalsh(self.P)))

        sign, logdet = np.linalg.slogdet(self.P)
        det_log = float(logdet) if sign > 0 else float("nan")

        try:
            np.linalg.cholesky(self.P)
            is_pd = True
        except np.linalg.LinAlgError:
            is_pd = False

        return {
            "sigma_px": float(sigmas[LP_State.PX]),
            "sigma_py": float(sigmas[LP_State.PY]),
            "sigma_pz": float(sigmas[LP_State.PZ]),
            "sigma_v": float(sigmas[LP_State.V]),
            "sigma_a": float(sigmas[LP_State.A]),
            "sigma_yaw": float(sigmas[LP_State.YAW]),
            "sigma_yaw_rate": float(sigmas[LP_State.YAW_RATE]),
            "covar_trace": trace,
            "covar_det_log": det_log,
            "covar_max_eig": max_eig,
            "is_pd": is_pd,
            "nis": self._last_nis,
            "mahalanobis_threshold": self._mahalanobis_threshold,
            "rejected_measurements": self._rejected_measurements,
        }


    @staticmethod
    def _f_A(theta):
        small = np.abs(theta) < 1e-3
        theta_safe = np.where(small, 1.0, theta)  # avoid 0/0 in the "exact" branch
        exact = (np.cos(theta) - 1 + theta * np.sin(theta)) / theta_safe**2
        taylor = 0.5 - theta**2 / 8 + theta**4 / 144
        return np.where(small, taylor, exact)


    @staticmethod
    def _f_B(theta):
        small = np.abs(theta) < 1e-3
        theta_safe = np.where(small, 1.0, theta)
        exact = (theta * np.cos(theta) - np.sin(theta)) / theta_safe**2
        taylor = -theta / 3 + theta**3 / 30
        return np.where(small, taylor, exact)


    def _fx_vectorized(self, X, Y, quad_vel, dt) -> None:
        """ Vectorised state model updater. Implements a localised frame version of the
            CTRA kinematic model for a vehicle.

        :param X: Old state vector
        :param Y: New state vector
        :param quad_vel: Quadcopter velocity
        :param dt: UKF timestep
        """
        px = X[:, LP_State.PX]
        py = X[:, LP_State.PY]
        pz = X[:, LP_State.PZ]
        v = X[:, LP_State.V]
        a = X[:, LP_State.A]
        yaw = X[:, LP_State.YAW]
        omega = X[:, LP_State.YAW_RATE]

        theta = omega * dt  # total yaw change over the step

        quad_dx = quad_vel[0] * dt
        quad_dy = quad_vel[1] * dt
        quad_dz = quad_vel[2] * dt

        sinc_term = np.sinc(theta / (2 * np.pi))       # safe at theta = 0
        mid_yaw = yaw + theta / 2
        fA = self._f_A(theta)
        fB = self._f_B(theta)

        # Pad -> rear axle (the point CTRA actually describes), propagate, axle -> pad
        dx, dy = self.axle_to_pad
        c0, s0 = np.cos(yaw), np.sin(yaw)
        rx = px - (c0 * dx - s0 * dy)
        ry = py - (s0 * dx + c0 * dy)

        rx_new = (
            rx
            + v * dt * sinc_term * np.cos(mid_yaw)
            + a * dt**2 * (np.cos(yaw) * fA + np.sin(yaw) * fB)
        )
        ry_new = (
            ry
            + v * dt * sinc_term * np.sin(mid_yaw)
            + a * dt**2 * (np.sin(yaw) * fA - np.cos(yaw) * fB)
        )

        yaw_new = yaw + theta
        c1, s1 = np.cos(yaw_new), np.sin(yaw_new)
        px_new = rx_new + (c1 * dx - s1 * dy) - quad_dx
        py_new = ry_new + (s1 * dx + c1 * dy) - quad_dy

        Y[:, LP_State.PX] = px_new
        Y[:, LP_State.PY] = py_new
        Y[:, LP_State.PZ] = pz - quad_dz
        Y[:, LP_State.V] = v + a * dt
        Y[:, LP_State.A] = a
        Y[:, LP_State.YAW] = self._wrap(yaw + theta)
        Y[:, LP_State.YAW_RATE] = omega


    def _hx_vectorized(self, X, Z) -> None:
        """Vectorised pose measurement model

        :param X: State vector
        :param Z: Measurement vector
        """
        Z[:, LP_Meas.PX] = X[:, LP_State.PX]
        Z[:, LP_Meas.PY] = X[:, LP_State.PY]
        Z[:, LP_Meas.PZ] = X[:, LP_State.PZ]
        Z[:, LP_Meas.YAW] = self._wrap(X[:, LP_State.YAW])


    def _reseed_sigma_points(self) -> None:
        """ Regenerate X_prop as sigma points about the current (x, P), so the next
            update() (or an OOSM rewind that anchors on this state) sees sigma
            points consistent with the current estimate.
        """
        try:
            S = self.gamma * np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            self._repair_P()
            S = self.gamma * np.linalg.cholesky(self.P)

        n = self.dim_x
        self.X_prop[0] = self.x
        for i in range(n):
            self.X_prop[i + 1] = self.x + S[:, i]
            self.X_prop[n + i + 1] = self.x - S[:, i]


    def _push_predict_event(self, timestamp: float, quad_vel) -> None:
        """ Append a predict event to the buffer and prune anything now older
            than _buffer_window relative to the newest entry.

        :param timestamp: Timestamp of this predict event
        :param quad_vel:  Process input used for this predict event
        """
        self._UKF_buffer.append(
            self._Event(
                timestamp,
                "predict",
                self.x.copy(),
                self.P.copy(),
                self.X_prop.copy(),
                np.asarray(quad_vel, dtype=float).copy(),
            )
        )
        self._prune_buffer_front(timestamp)


    def _push_update_event(self, timestamp: float, meas) -> None:
        """ Append an update event to the buffer and prune anything now older
            than _buffer_window relative to the newest entry.

        :param timestamp: Timestamp of this update event
        :param meas:      _Meas(z, R, cam) applied for this update event (cam carries
                          the image-time camera pose for box measurements)
        """
        self._UKF_buffer.append(
            self._Event(
                timestamp,
                "update",
                self.x.copy(),
                self.P.copy(),
                self.X_prop.copy(),
                meas,
            )
        )
        self._prune_buffer_front(timestamp)


    def _prune_buffer_front(self, latest_timestamp: float) -> None:
        """ Drop buffered events older than _buffer_window relative to the
            newest timestamp seen.

        :param latest_timestamp: Most recent timestamp pushed to the buffer
        """
        cutoff = latest_timestamp - self._buffer_window
        while self._UKF_buffer and self._UKF_buffer[0].t < cutoff:
            self._UKF_buffer.popleft()


    def _nearest_quad_vel(self, buffer_list, idx) -> np.ndarray:
        """ Walk backward from idx to find the most recent "predict" event's
            quad_vel. Needed when an OOSM anchor turns out to be an "update"
            event (which carries no process input of its own) and a small
            residual gap still needs bridging up to the OOSM's own timestamp.

        :param buffer_list: Ordered (oldest -> newest) list of buffered events
        :param idx:         Index to walk backward from, inclusive
        :return: Most recent known quad_vel at or before idx
        """
        for i in range(idx, -1, -1):
            if buffer_list[i].kind == "predict":
                return buffer_list[i].data
        return self._last_quad_vel


    def _repair_P(self) -> None:
        """ Force P back to symmetric positive definite via eigendecomposition"""
        self.P = 0.5 * (self.P + self.P.T)
        eigvals, eigvecs = np.linalg.eigh(self.P)
        eigvals = np.maximum(eigvals, 1e-6)
        self.P = eigvecs @ np.diag(eigvals) @ eigvecs.T


    @staticmethod
    def _wrap(angle: np.ndarray) -> np.ndarray:
        """ Wraps the angle in the domain -π to π

        :param angle: Single float or vector of angles
        :return: Wrapped angle
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi


    @staticmethod
    def _circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
        """ Weighted circular mean — safe across the ±π boundary

        :param angles: Vector of angles
        :param weights: Vector of weights for the weighted average
        :return: The weighted mean of the angles
        """
        sin_mean = np.sum(weights * np.sin(angles))
        cos_mean = np.sum(weights * np.cos(angles))
        return float(np.arctan2(sin_mean, cos_mean))