import numpy as np
from collections import deque
from collections import namedtuple

from .state_definitions import LP_State
from .state_definitions import LP_Measurement

""" UKF Class definitions to implement a target/chaser kinematic model. Consists of all 
    higher level UKF logic for prediction and updating of kinematic model, as well as 
    UKF rollback for update measurements. 
"""

class UKF:
    """ Defines the UKF class for the landing platform.
    """
    # Define the UKF event-log structure. Every predict() and every accepted
    # update() call pushes one of these, in chronological order, allowing the
    # filter to rewind to any point in the buffer window and *replay* history
    # exactly rather than re-simulate over it. This is what lets multiple,
    # independent measurement streams (e.g. AprilTag + YOLO) each perform
    # out-of-sequence-measurement (OOSM) corrections without one stream's
    # correction silently erasing the other's.
    #
    #   kind == "predict": data = quad_vel used for that process step
    #   kind == "update":  data = (z, R) used for that measurement correction
    #
    # x / P / X_prop are the UKF state, covariance, and propagated sigma
    # points immediately AFTER this event was applied.
    _Event = namedtuple("_Event", ["t", "kind", "x", "P", "X_prop", "data"])

    def __init__(self) -> None:
        """ Initialise the UKF. Preprocesses and computes the required sigma weights and
            stores them to reduce run-time computation.
        """

        # ---- UKF PARAMETERS ----
        # Dimensions
        self.dim_x = len(LP_State)
        self.dim_z = len(LP_Measurement)

        # UKF scaling parameters (Merwe)
        self.alpha = 1.0
        self.beta = 2.0
        self.kappa = 0.0

        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        self.gamma = np.sqrt(self.dim_x + self.lambda_)

        # Length of UKF Historical Buffer. Must comfortably exceed the WORST
        # CASE end-to-end latency of the slowest measurement stream feeding
        # this filter (capture -> image -> transform -> UKF), or that
        # stream's OOSM corrections will fall outside the buffer and be
        # silently dropped (anchor_idx is None -> update() returns False).
        # Check the *_transform_to_UKF_lag / *_cam_to_image_lag diagnostics
        # you're already publishing to size this properly per-stream; 0.5s
        # is a safer starting point than the previous 0.2s now that a
        # second, slower (YOLO) stream is in the mix.
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

        # Initial covariance
        self.P_init = np.diag(
            [
                0.15,
                0.15,
                0.15,
                0.50,
                1.50,
                0.15,
                0.50,
            ]
        )
        self.P = self.P_init.copy()

        # Process noise
        self.Q = np.diag(
            [
                0.005,
                0.005,
                0.005,  # px, py, pz
                0.050,
                0.200,  # v, a
                0.005,
                0.100,  # yaw, yaw_rate
            ]
        )

        self._nis_ewma = float(self.dim_z)  # start at expected value
        self._nis_ewma_beta = 0.90          # higher = slower to react, smoother
        self._q_infl_cap = 8.0              # max multiplier, tune to taste

        # Measurement noise. NOTE: update() snapshots this at call time (see
        # below), so callers are still free to mutate self.R in place right
        # before calling update() per-stream, same convention as before.
        self.R = np.diag([0.01, 0.01, 0.1, 0.01])

        # Timestamp of the last predict/update cycle, initialised to None
        self._last_update_time: float | None = None

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

        # infl = np.clip(self._nis_ewma / self.dim_z, 1.0, self._q_infl_cap)
        # Q_eff = self.Q.copy()
        # Q_eff[LP_State.A, LP_State.A] *= infl
        # Q_eff[LP_State.YAW_RATE, LP_State.YAW_RATE] *= infl
        self.P = (dX.T * self.Wc) @ dX + self.Q * dt # scale Q by dt (time-invariant)

        self.P = 0.5 * (self.P + self.P.T)

        # Re-seed X_prop with sigma points from the UPDATED predicted covariance
        try:
            S_new = self.gamma * np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            self._repair_P()
            S_new = self.gamma * np.linalg.cholesky(self.P)

        self.X_prop[0] = self.x
        for i in range(self.dim_x):
            self.X_prop[i + 1] = self.x + S_new[:, i]
            self.X_prop[self.dim_x + i + 1] = self.x - S_new[:, i]

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


    def update(self, z, measurement_timestamp: float) -> bool:
        """ Do update step for UKF. Manages higher level update functions such as 
            checking the measurement timestamp and performing the UKF rewind and 
            rollback. Safe to call from multiple independent measurement streams
            (e.g. AprilTag and YOLO) in any interleaving/order - out-of-sequence
            measurements are spliced into the correct chronological position and
            every event after that point (predicts AND updates, from every stream)
            is replayed against the corrected timeline.

        :param z:                     Measurement of new landing pad pose
        :param measurement_timestamp: True timestamp of the measurement (seconds)
        :return: True on success, False on failure.
        """

        z = np.asarray(z, dtype=float)
        # Snapshot R at call time rather than reading self.R again later -
        # avoids a race where a second stream reassigns self.R before this
        # measurement's correction is actually replayed.
        R_used = self.R.copy()

        # In-order path: this is the newest thing we've seen, no rewind needed.
        if self._last_update_time is None or measurement_timestamp >= self._last_update_time:
            accepted = self._update_apply(z, R_used)
            if accepted:
                self._last_update_time = measurement_timestamp
                self._push_update_event(measurement_timestamp, z, R_used)
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
            bridge_quad_vel = self._nearest_quad_vel(buffer_copy, anchor_idx)
            self.predict(bridge_quad_vel, dt_bridge, measurement_timestamp, buffer=False)

        accepted = self._update_apply(z, R_used)

        # If the update failed, bail back to current state - real buffer untouched
        if not accepted:
            self.x, self.P, self.X_prop = x_now, P_now, X_prop_now
            self._last_update_time = t_now
            return False

        # Only commit to the rewind now that we know it succeeded: prune
        # entries forward of the anchor from the real buffer, then rebuild it
        # by splicing this OOSM in and replaying every event that originally
        # came after it - from BOTH streams - in the order it actually
        # happened, instead of blindly re-predicting over lost corrections.
        while self._UKF_buffer and self._UKF_buffer[-1].t > anchor.t:
            self._UKF_buffer.pop()

        self._last_update_time = measurement_timestamp
        self._push_update_event(measurement_timestamp, z, R_used)

        prev_t = measurement_timestamp
        for ev in future_events:
            if ev.kind == "predict":
                dt_step = ev.t - prev_t
                if dt_step > 1e-9:
                    self.predict(ev.data, dt_step, ev.t)  # buffer=True re-pushes it
            else:  # "update" - replay the other stream's correction too
                ev_z, ev_R = ev.data
                if self._update_apply(ev_z, ev_R):
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

        self.X.fill(0.0)
        self.X_prop.fill(0.0)
        self.Z.fill(0.0)


    def _update_apply(self, z, R=None) -> bool:
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

        if R is None:
            R = self.R

        # Propagate sigma points through hx
        self._hx_vectorized(self.X_prop, self.Z)

        # Predicted measurement mean
        z_pred = (self.Wm[:, None] * self.Z).sum(axis=0)
        z_pred[LP_Measurement.YAW] = self._circular_mean(
            self.Z[:, LP_Measurement.YAW], self.Wm
        )  # circular mean for yaw

        # Measurement deviations
        dZ = self.Z - z_pred
        dZ[:, LP_Measurement.YAW] = self._wrap(dZ[:, LP_Measurement.YAW])

        # State deviations
        dX = self.X_prop - self.x
        dX[:, LP_State.YAW] = self._wrap(dX[:, LP_State.YAW])

        # Cross covariance
        P_xz = (dX.T * self.Wc) @ dZ

        # Innovation covariance
        S = (dZ.T * self.Wc) @ dZ + R

        # Kalman gain, if S is singular skip update
        try:
            K = np.linalg.solve(S.T, P_xz.T).T
        except np.linalg.LinAlgError:
            return False

        # Innovation
        y = z - z_pred
        y[LP_Measurement.YAW] = self._wrap(y[LP_Measurement.YAW])

        # Update state
        self.x += K @ y
        self.x[LP_State.YAW] = self._wrap(self.x[LP_State.YAW])

        # Covariance update
        self.P -= K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Log the NIS
        self._last_nis = float(y @ np.linalg.solve(S, y))
        self._nis_ewma = (self._nis_ewma_beta * self._nis_ewma
                        + (1 - self._nis_ewma_beta) * self._last_nis)

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

        Z[:, LP_Measurement.PX] = X[:, LP_State.PX]
        Z[:, LP_Measurement.PY] = X[:, LP_State.PY]
        Z[:, LP_Measurement.PZ] = X[:, LP_State.PZ]
        Z[:, LP_Measurement.YAW] = self._wrap(X[:, LP_State.YAW])


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