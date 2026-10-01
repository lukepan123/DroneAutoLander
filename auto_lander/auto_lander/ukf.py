import numpy as np
import tf_transformations

from collections import deque
from collections import namedtuple
from rclpy.time import Time

from .state_definitions import LP_State
from .state_definitions import LP_Meas

""" UKF Class definitions to implement a target/chaser kinematic model. Consists of all 
    higher level UKF logic for prediction and updating of kinematic model, as well as 
    UKF rollback for update measurements. 
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
        kind == "update":  data = (z, R) used for that measurement correction
    
        x / P / X_prop are the UKF state, covariance, and propagated sigma
        points immediately AFTER this event was applied.
    """

    _Event = namedtuple("_Event", ["t", "kind", "x", "P", "X_prop", "data"])


    def __init__(self, P_diag_init, Q_diag_init, 
                 alpha, beta, kappa,
                 mahalanobis_threshold, 
                 init_pos, init_vel, init_yaw,
                 seed_window_n, seed_window_min_dt) -> None:
        """ Initialise the UKF. Preprocesses and computes the required sigma weights and
            stores them to reduce run-time computation.
        """
        # ---- UKF PARAMETERS ----
        # Dimensions
        self.dim_x = len(LP_State)
        self.dim_z = len(LP_Meas)

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
        self._buffer_window = 0.500

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

        # Timestamp of the last predict/update cycle, initialised to None
        self._last_update_time: float | None = None

        # Seeding buffers/variables
        self._seed_buffers = {"apriltag": deque(), "yolo_lp": deque(), "yolo_car": deque()}
        self._seed_window_n = seed_window_n
        self._seed_window_min_dt = seed_window_min_dt
        self._seed_pending = False
        self._measurement_last_stamps = {k: None for k in self._seed_buffers}
        self._measurement_diagnostics = {
            k: {
                "raw": np.zeros(4),
                "raw_stamp": 0.0,
                "meas_age": 0.0,
                "cam_to_image_lag": np.nan,
                "image_to_transform_lag": np.nan,
                "transform_to_UKF_lag": np.nan,
                "total_lag": np.nan,
            }
            for k in self._seed_buffers
        }
        self._last_measurement_type = "apriltag"
        self._last_yolo_measurement_type = "yolo_lp"

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
        self._nis_ewma = float(self.dim_z)  # start at expected value
        self._nis_ewma_beta = 0.90          # higher = slower to react, smoother


    def predict(self, quad_vel, dt: float, timestamp: float, buffer: bool = True) -> None:
        """ Do prediction step for UKF. Calls fx_vectorised which is the kinematic 
            process function for the filter. 
        
        :param quad_vel:   quadcopter velocity (in LOCAL frame) (m/s)
        :param dt:         timestep (s)
        :param timestamp:  timestamp (s)
        :param buffer:     update buffer (bool)
        """
        # Guarantee P is symmetric positive definite before proceeding, repair if needed
        try:
            S = np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            self._repair_P()
            S = np.linalg.cholesky(self.P)

        n = self.dim_x

        # Cholesky of covariance
        S = self.gamma * S

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

        self.P = (dX.T * self.Wc) @ dX + self.Q * dt # scale Q by dt (time-invariant)
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
            tf_msg,
            measurement_type: str,
            now: float,
            R,
            pipeline_timing=None,
    ) -> bool:
        """ Do update step for UKF from a TF transform. Manages higher level update
            functions such as checking the measurement timestamp and performing the UKF
            rewind and rollback. Safe to call from multiple independent measurement
            streams (e.g. AprilTag and YOLO) in any interleaving/order - out-of-sequence
            measurements are spliced into the correct chronological position and
            every event after that point (predicts AND updates, from every stream)
            is replayed against the corrected timeline.

        :param tf_msg:          Measurement of new landing pad pose (structured as a TF message)
        :param measurement_type: String for measurement type (apriltag, yolo_lp, yolo_car)
        :param now:             Current timestamp (seconds)
        :param R:               Measurement covariance
        :param pipeline_timing: Optional Vector3Stamped containing pipeline timestamps
        :return: True on success, False on failure.
        """
        measurement_timestamp = Time.from_msg(tf_msg.header.stamp).nanoseconds / 1e9
        last_timestamp = self._measurement_last_stamps[measurement_type]

        # Ignore duplicate measurements, but don't error.
        if last_timestamp is not None and measurement_timestamp == last_timestamp:
            return True

        # Build up measurement vector z
        t = np.array([
            tf_msg.transform.translation.x,
            tf_msg.transform.translation.y,
            tf_msg.transform.translation.z,
        ])
        q = tf_msg.transform.rotation
        _, _, yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )
        z = np.array([t[0], t[1], t[2], yaw])

        # Newest timestamp is tracked per stream; older OOSMs don't move it backwards.
        if last_timestamp is None or measurement_timestamp > last_timestamp:
            self._measurement_last_stamps[measurement_type] = measurement_timestamp

        # Record raw measurement/latency for this stream before any update path can fail.
        self._record_measurement_diagnostics(
            measurement_type, z, measurement_timestamp, now, pipeline_timing
        )
        self._last_measurement_type = measurement_type
        if measurement_type.startswith("yolo"):
            self._last_yolo_measurement_type = measurement_type

        # If seeding is required, run the seeding procedure and exit early.
        if self._seed_pending:
            buf = self._seed_buffers[measurement_type]
            buf.append((measurement_timestamp, z.copy(), R.copy()))
            span = buf[-1][0] - buf[0][0]
            if len(buf) >= self._seed_window_n or span >= self._seed_window_min_dt:
                self.seed_from_window(list(buf))
                self._seed_pending = False
                for b in self._seed_buffers.values():
                    b.clear()
            return True

        # In-order path: this is the newest thing we've seen, no rewind needed.
        if self._last_update_time is None or measurement_timestamp >= self._last_update_time:
            accepted = self._update_apply(z, R)
            if accepted:
                self._last_update_time = measurement_timestamp
                self._push_update_event(measurement_timestamp, z, R)
            return accepted

        # ---- OOSM path ----
        if not self._UKF_buffer:
            return False

        buffer_copy = list(self._UKF_buffer)  # ordered oldest -> newest
        anchor_idx = None
        for i, ev in enumerate(buffer_copy):
            if ev.t <= measurement_timestamp:
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
        dt_bridge = measurement_timestamp - anchor.t
        if dt_bridge > 1e-9:
            bridge_quad_vel = next(
                (ev.data for ev in future_events if ev.kind == "predict"),
                self._nearest_quad_vel(buffer_copy, anchor_idx),  # fallback: no later predict
            )
            self.predict(bridge_quad_vel, dt_bridge, measurement_timestamp, buffer=False)

        accepted = self._update_apply(z, R)

        # If the update failed, bail back to current state - real buffer untouched
        if not accepted:
            self.x, self.P, self.X_prop = x_now, P_now, X_prop_now
            self._last_update_time = t_now
            return False

        # Only commit to the rewind now that we know it succeeded: prune
        # entries forward of the anchor from the real buffer, then rebuild
        # it by splicing this OOSM in and replaying every event that originally
        # came after it - from BOTH streams - in the order it actually
        # happened, instead of blindly re-predicting over lost corrections.
        while self._UKF_buffer and self._UKF_buffer[-1].t > anchor.t:
            self._UKF_buffer.pop()

        self._last_update_time = measurement_timestamp
        self._push_update_event(measurement_timestamp, z, R)

        prev_t = measurement_timestamp
        for ev in future_events:
            if ev.kind == "predict":
                dt_step = ev.t - prev_t
                if dt_step > 1e-9:
                    self.predict(ev.data, dt_step, ev.t)  # buffer=True re-pushes it
            else:  # "update" - replay the other stream's correction too
                ev_z, ev_R = ev.data
                if self._update_apply(ev_z, ev_R, replay=True):
                    self._last_update_time = ev.t
                    self._push_update_event(ev.t, ev_z, ev_R)
                # A failed replay (rare - singular innovation covariance) is
                # simply skipped: _update_apply never mutates state before it
                # can fail, so this is safe and just drops that one stale
                # correction rather than corrupting the timeline.
            prev_t = ev.t

        return True


    def reset(self) -> None:
        """Reset runtime state; leaves parameters, weights, and preallocated
        sigma buffers untouched.
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
        self._last_measurement_type = "apriltag"
        self._last_yolo_measurement_type = "yolo_lp"

        for d in self._measurement_diagnostics.values():
            d["raw"] = np.zeros(4)
            d["raw_stamp"] = 0.0
            d["meas_age"] = 0.0
            d["cam_to_image_lag"] = np.nan
            d["image_to_transform_lag"] = np.nan
            d["transform_to_UKF_lag"] = np.nan
            d["total_lag"] = np.nan

        self.X.fill(0.0)
        self.X_prop.fill(0.0)
        self.Z.fill(0.0)

        self._rejected_measurements = 0
        self._last_nis = np.nan
        self._nis_ewma = float(self.dim_z)


    @property
    def seed_pending(self) -> bool:
        """ Return True while the initial measurement window is being collected. """
        return self._seed_pending


    def _record_measurement_diagnostics(
            self, measurement_type, z, measurement_timestamp, now, pipeline_timing=None
    ) -> None:
        """ Store raw measurement and latency diagnostics for one measurement stream. """
        d = self._measurement_diagnostics[measurement_type]
        d["raw"] = z.copy()
        d["raw_stamp"] = measurement_timestamp
        d["meas_age"] = (now - measurement_timestamp) * 1000.0
        d["cam_to_image_lag"] = np.nan
        d["image_to_transform_lag"] = np.nan
        d["transform_to_UKF_lag"] = np.nan
        d["total_lag"] = d["meas_age"]

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


    def get_measurement_diagnostics(self) -> dict:
        """ Retrieve diagnostics associated with all measurement streams. """
        d = {
            k: {
                "raw": v["raw"].copy(),
                "raw_stamp": v["raw_stamp"],
                "meas_age": v["meas_age"],
                "cam_to_image_lag": v["cam_to_image_lag"],
                "image_to_transform_lag": v["image_to_transform_lag"],
                "transform_to_UKF_lag": v["transform_to_UKF_lag"],
                "total_lag": v["total_lag"],
            }
            for k, v in self._measurement_diagnostics.items()
        }
        d["latest"] = d[self._last_measurement_type]
        d["latest_yolo"] = d[self._last_yolo_measurement_type]
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
            information for this stream (e.g. YOLO's pz/yaw, R~1e9) and are
            left untouched - neither x nor P is modified for that axis. This
            matters because seeding P from a sentinel-large R would otherwise
            blow past covar_max_eig safety thresholds on a perfectly healthy
            filter. Those axes get filled in normally (via Kalman gain) the
            next time a stream that does trust them provides an update.

            Velocity/acceleration/yaw-rate are always left at reset() defaults
            since no stream measures them directly.

        :param z:            Measurement of landing pad pose (px, py, pz, yaw)
        :param timestamp:     Timestamp of the seeding measurement (s)
        :param R:             Measurement noise for this stream. Defaults to
                            self.R. Also used to decide which axes to skip.
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

        self._push_update_event(timestamp, z, R)


    def seed_from_window(
        self, samples: list[tuple[float, np.ndarray, np.ndarray]], ignore_above: float = 1e6
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

        :param samples: (timestamp, z, R) tuples, oldest first, ALL FROM THE
                        SAME STREAM (mixing streams with different noise/rate
                        characteristics will bias the fit). z = (px,py,pz,yaw).
        :param ignore_above: as per seed_position - axes with R >= this are
                            treated as unmeasured by this stream and skipped.
        """
        n = len(samples)
        if n < 3:
            t0, z0, R0 = samples[-1]
            self._last_seed_diag = {"path": "single_sample_fallback", "n": n}
            self.seed_position(z0, t0, R0, ignore_above)
            return

        t = np.array([s[0] for s in samples])
        Z = np.array([s[1] for s in samples])
        R_last = samples[-1][2]
        dt = t - t[0]  # regress on seconds-since-window-start, not raw epoch time
        self._last_seed_diag = {
            "path": "window_fit", "n": n, "span": float(dt[-1] - dt[0]),
            "dt": dt.tolist(),
        }

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
                # No yaw from this stream (YOLO: R_yaw ~ 1e9). self.x[LP_State.YAW]
                # is still the reset() default, so projecting onto it silently
                # corrupts V - that was the bug (v0 above == vx, because yaw0 was
                # stuck at 0.0). Instead derive heading AND speed from the direction
                # of travel - valid for a target that points the way it moves, but
                # only trustworthy if it's actually moving: near-zero displacement
                # gives a heading dominated by position noise, not motion.
                MIN_SPEED_FOR_HEADING = 0.3  # m/s - tune against your position noise floor
                if speed_mag >= MIN_SPEED_FOR_HEADING:
                    self.x[LP_State.YAW] = self._wrap(np.arctan2(vy, vx))
                    self.x[LP_State.V] = speed_mag
                    # sigma_yaw ~ sigma_v / speed via atan2 error propagation -
                    # correctly blows up as speed -> 0.
                    var_yaw = var_v / max(speed_mag**2, 1e-6)
                    self.P[LP_State.YAW, LP_State.YAW] = max(4.0 * var_yaw, 1e-3)
                    self.P[LP_State.V, LP_State.V] = max(4.0 * var_v, 1e-3)
                # else: leave x[V]/x[YAW]/P at reset() defaults - can't distinguish
                # "barely moving" from "stationary" given this stream's noise floor,
                # so don't force a heading out of it.

        self._last_seed_diag.update({"vx": float(vx), "vy": float(vy),
                                "v0": float(self.x[LP_State.V]),
                                "yaw0": float(self.x[LP_State.YAW]),
                                "yaw_rate": float(self.x[LP_State.YAW_RATE])})
            
        if not seeded_any:
            return

        self._last_update_time = t[-1]
        self._reseed_sigma_points()

        self._UKF_buffer.clear() # Clear buffer incase any predicts were called prior (shouldnt be)
        self._push_update_event(t[-1], samples[-1][1], R_last)


    def _update_apply(self, z, R, replay: bool = False) -> bool:
        """ Do update step for UKF. Handles lower level UKF update functions such as 
            actual covariance and state updates. Called via the public .update() 
            function (both the in-order path and OOSM replay).

        :param z: Measurement of new landing pad pose
        :param R: Measurement noise covariance to use for this specific update.
                   Defaults to self.R for backward compatibility, but update()
                   always passes this explicitly so replayed events use the R
                   that was actually in effect when they first happened.
        :return: True on success, False on failure.
        """
        # Propagate sigma points through hx
        self._hx_vectorized(self.X_prop, self.Z)

        # Predicted measurement mean
        z_pred = (self.Wm[:, None] * self.Z).sum(axis=0)
        z_pred[LP_Meas.YAW] = self._circular_mean(
            self.Z[:, LP_Meas.YAW], self.Wm
        )  # circular mean for yaw

        # Measurement deviations
        dZ = self.Z - z_pred
        dZ[:, LP_Meas.YAW] = self._wrap(dZ[:, LP_Meas.YAW])

        # State deviations
        dX = self.X_prop - self.x
        dX[:, LP_State.YAW] = self._wrap(dX[:, LP_State.YAW])

        # Cross covariance
        P_xz = (dX.T * self.Wc) @ dZ

        # Innovation covariance
        S = (dZ.T * self.Wc) @ dZ + R

        # Innovation
        y = z - z_pred
        y[LP_Meas.YAW] = self._wrap(y[LP_Meas.YAW])

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
            return False

        # Kalman gain, if S is singular skip update
        try:
            K = np.linalg.solve(S.T, P_xz.T).T
        except np.linalg.LinAlgError:
            return False

        # Update state
        self.x += K @ y
        self.x[LP_State.YAW] = self._wrap(self.x[LP_State.YAW])

        # Covariance update
        self.P -= K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Re-seed sigma points from the posterior
        self._reseed_sigma_points()

        # EWMA NIS only gets updated for accepted measurements
        if not replay:
            self._nis_ewma = (
                self._nis_ewma_beta * self._nis_ewma
                + (1 - self._nis_ewma_beta) * nis
            )

        return True


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

        px_new = (
            px
            + v * dt * sinc_term * np.cos(mid_yaw)
            + a * dt**2 * (np.cos(yaw) * fA + np.sin(yaw) * fB)
            - quad_dx
        )
        py_new = (
            py
            + v * dt * sinc_term * np.sin(mid_yaw)
            + a * dt**2 * (np.sin(yaw) * fA - np.cos(yaw) * fB)
            - quad_dy
        )

        Y[:, LP_State.PX] = px_new
        Y[:, LP_State.PY] = py_new
        Y[:, LP_State.PZ] = pz - quad_dz
        Y[:, LP_State.V] = v + a * dt
        Y[:, LP_State.A] = a
        Y[:, LP_State.YAW] = self._wrap(yaw + theta)
        Y[:, LP_State.YAW_RATE] = omega


    def _hx_vectorized(self, X, Z) -> None:
        """Vectorised measurement model

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


    def _push_update_event(self, timestamp: float, z, R) -> None:
        """ Append an update event to the buffer and prune anything now older
            than _buffer_window relative to the newest entry.

        :param timestamp: Timestamp of this update event
        :param z:         Measurement applied for this update event
        :param R:         Measurement noise covariance applied for this event
        """
        self._UKF_buffer.append(
            self._Event(
                timestamp,
                "update",
                self.x.copy(),
                self.P.copy(),
                self.X_prop.copy(),
                (np.asarray(z, dtype=float).copy(), np.asarray(R, dtype=float).copy()),
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