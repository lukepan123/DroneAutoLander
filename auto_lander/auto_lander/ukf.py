"""UKF Class definitions to implement a target/chaser kinematic model"""

import numpy as np
from collections import deque

from .state_definitions import LP_State
from .state_definitions import LP_Measurement


class UKF:
    """Defines the UKF class for the landing platform"""

    _snapshot = tuple  # (timestamp, UKF state (x), UKF covariances (P), UKF propogated sigma points)

    def __init__(self) -> None:
        """Initialise the UKF

        Preprocesses and computes the required sigma weights and stores them to reduce run-time computation.
        """

        # Timestamp of the last predict/update cycle (seconds, ROS clock).
        # None until the first measurement arrives.
        self.last_update_time: float | None = None

        # Dimensions
        self.dim_x = len(LP_State)
        self.dim_z = len(LP_Measurement)

        # UKF scaling parameters (Merwe)
        self.alpha = 0.3
        self.beta = 2.0
        self.kappa = -4.0

        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        self.gamma = np.sqrt(self.dim_x + self.lambda_)

        # Number of sigma points
        self.num_sigma = 2 * self.dim_x + 1

        # Preallocate sigma arrays
        self.X = np.zeros((self.num_sigma, self.dim_x))  # sigma points
        self.X_prop = np.zeros((self.num_sigma, self.dim_x))  # fx result
        self.Z = np.zeros((self.num_sigma, self.dim_z))  # hx result

        # Initial state
        self.x = np.zeros(self.dim_x)

        # Initial covariance
        self.P = np.diag(
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

        # Process noise
        self.Q = np.diag(
            [
                0.005,
                0.005,
                0.005,  # px, py, pz
                0.050,
                0.100,  # v, a
                0.005,
                0.025,  # yaw, yaw_rate
            ]
        )

        # Measurement noise
        self.R = np.diag([0.01, 0.01, 0.1, 0.01])

        # Precompute weights
        self.Wm = np.full(self.num_sigma, 0.5 / (self.dim_x + self.lambda_))
        self.Wc = np.full(self.num_sigma, 0.5 / (self.dim_x + self.lambda_))

        self.Wm[0] = self.lambda_ / (self.dim_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.dim_x + self.lambda_) + (
            1 - self.alpha**2 + self.beta
        )

        # OOSM buffer, ordered from oldest -> newest
        # Each entry: (timestamp, x, P)
        self._buffer: deque[UKF._snapshot] = deque()

    def predict(self, quad_vel, quad_accel, dt: float, timestamp: float) -> None:
        """Do prediction step for UKF."""
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
        self._fx_vectorized(self.X, self.X_prop, quad_vel, quad_accel, dt)

        # Predicted mean
        self.x[:] = (self.Wm[:, None] * self.X_prop).sum(axis=0)
        self.x[LP_State.YAW] = self._circular_mean(
            self.X_prop[:, LP_State.YAW], self.Wm
        )

        # Predicted covariance
        dX = self.X_prop - self.x
        dX[:, LP_State.YAW] = self._wrap(dX[:, LP_State.YAW])  # wrap yaw deviations

        self.P = (
            dX.T * self.Wc
        ) @ dX + self.Q * dt  # scale Q by the time between predict steps
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

        # Update timestamp and store state in the buffer
        self.last_update_time = timestamp
        self._buffer_push(timestamp, quad_vel, quad_accel)

    def update(
        self, z, measurement_timestamp: float) -> bool:
        """Do update step for UKF.
        :param z: Measurement of new landing pad pose
        :param measurement_timestamp: True timestamp of the measurement (seconds)
        """

        if (
            self.last_update_time is not None
            and measurement_timestamp < self.last_update_time
            and len(self._buffer) > 0
        ):
            # Iterate forward through the buffer until the t > measurement_timestamp
            buf_list = list(self._buffer)  # Make a copy
            anchor_idx = None
            for i, (t, *_) in enumerate(buf_list):
                if t <= measurement_timestamp:
                    anchor_idx = i
                else:
                    break

            if anchor_idx is None:
                return False  # OOSM older than entire buffer, skip

            # Save the state of the UKF at this timestamp
            t_anchor, x_anchor, P_anchor, X_prop_anchor, _, _ = buf_list[
                anchor_idx
            ]

            # Collect the future snapshots (i.e. snapshots after the measurement time)
            future_snapshots = buf_list[anchor_idx + 1 :]

            # Save full current state
            x_now = self.x.copy()
            P_now = self.P.copy()
            X_prop_now = self.X_prop.copy()
            t_now = self.last_update_time

            # Rewind to anchor
            self.x, self.P, self.X_prop = (
                x_anchor.copy(),
                P_anchor.copy(),
                X_prop_anchor.copy(),
            )
            self.last_update_time = t_anchor

            # Prune the buffer forward of the anchor
            while self._buffer and self._buffer[-1][0] > t_anchor:
                self._buffer.pop()

            accepted = self.update_apply(z)
            # If the update failed, bail back to current state
            if not accepted:
                self.x = x_now.copy()
                self.P = P_now.copy()
                self.X_prop = X_prop_now.copy()
                self.last_update_time = t_now
                return False

            # Fast-forward using pre-collected future timestamps
            prev_t = t_anchor
            for snap_t, _, _, _, snap_quad_vel, snap_quad_accel in future_snapshots:
                dt_step = snap_t - prev_t
                if dt_step > 1e-6:
                    self.predict(snap_quad_vel, snap_quad_accel, dt_step, snap_t)
                prev_t = snap_t

            return True

        else:
            return self.update_apply(z)

    def update_apply(self, z) -> bool:
        """Do update step for UKF.

        :param z: Measurement of new landing pad pose
        """

        # 1) Propagate sigma points through hx
        self._hx_vectorized(self.X_prop, self.Z)

        # 2) Predicted measurement mean
        z_pred = (self.Wm[:, None] * self.Z).sum(axis=0)
        z_pred[LP_Measurement.YAW] = self._circular_mean(
            self.Z[:, LP_Measurement.YAW], self.Wm
        )  # circular mean for yaw

        # 3) Measurement deviations
        dZ = self.Z - z_pred
        dZ[:, LP_Measurement.YAW] = self._wrap(dZ[:, LP_Measurement.YAW])

        # 4) State deviations
        dX = self.X_prop - self.x
        dX[:, LP_State.YAW] = self._wrap(dX[:, LP_State.YAW])

        # 5) Cross covariance
        P_xz = (dX.T * self.Wc) @ dZ

        # 6) Innovation covariance
        S = (dZ.T * self.Wc) @ dZ + self.R

        # 7) Kalman gain
        K = np.linalg.solve(S.T, P_xz.T).T

        # 8) Innovation
        y = z - z_pred
        y[LP_Measurement.YAW] = self._wrap(y[LP_Measurement.YAW])

        # 9) Update state
        self.x += K @ y
        self.x[LP_State.YAW] = self._wrap(self.x[LP_State.YAW])

        # 10) Covariance update
        self.P -= K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True

    def get_covar_diagnostics(self) -> dict:
        """Return per-state 1-sigma values and scalar health metrics.

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
        }

    def _fx_vectorized(self, X, Y, quad_vel, quad_accel, dt):
        """Vectorised state model updater.

        :param X: Old state vector
        :param Y: New state vector
        :param quad_vel: Quadcopter velocity
        :param quad_accel: Quadcopter acceleration
        :param dt: UKF timestep
        """

        px = X[:, LP_State.PX]
        py = X[:, LP_State.PY]
        pz = X[:, LP_State.PZ]
        v = X[:, LP_State.V]
        a = X[:, LP_State.A]
        yaw = X[:, LP_State.YAW]
        omega = X[:, LP_State.YAW_RATE]

        yaw_new = yaw + omega * dt

        # Threshold for "straight line" motion
        eps = 1e-4

        turning = np.abs(omega) > eps
        straight = ~turning

        # Account for quadcopter velocity
        quad_dx = quad_vel[0] * dt + 0.5 * quad_accel[0] * dt**2
        quad_dy = quad_vel[1] * dt + 0.5 * quad_accel[1] * dt**2
        quad_dz = quad_vel[2] * dt + 0.5 * quad_accel[2] * dt**2

        # Allocate outputs
        px_new = np.empty_like(px)
        py_new = np.empty_like(py)

        # Straight-line motion
        px_new[straight] = (
            px[straight]
            + v[straight] * np.cos(yaw[straight]) * dt
            + 0.5 * a[straight] * np.cos(yaw[straight]) * dt**2
            - quad_dx
        )

        py_new[straight] = (
            py[straight]
            + v[straight] * np.sin(yaw[straight]) * dt
            + 0.5 * a[straight] * np.sin(yaw[straight]) * dt**2
            - quad_dy
        )

        # Constant Turn Rate + Acceleration
        w = omega[turning]
        y0 = yaw[turning]
        y1 = yaw_new[turning]
        vt = v[turning]
        at = a[turning]

        px_new[turning] = (
            px[turning]
            + vt / w * (np.sin(y1) - np.sin(y0))
            + at / w**2 * (np.cos(y1) - np.cos(y0) + w * dt * np.sin(y1))
            - quad_dx
        )

        py_new[turning] = (
            py[turning]
            + vt / w * (-np.cos(y1) + np.cos(y0))
            + at / w**2 * (np.sin(y1) - np.sin(y0) - w * dt * np.cos(y1))
            - quad_dy
        )

        Y[:, LP_State.PX] = px_new
        Y[:, LP_State.PY] = py_new
        Y[:, LP_State.PZ] = pz - quad_dz

        Y[:, LP_State.V] = v + a * dt
        Y[:, LP_State.A] = a

        Y[:, LP_State.YAW] = self._wrap(yaw_new)
        Y[:, LP_State.YAW_RATE] = omega

    def _hx_vectorized(self, X, Z):
        """Vectorised measurement model

        :param X: State vector
        :param Z: Measurement vector
        """
        Z[:, LP_Measurement.PX] = X[:, LP_State.PX]
        Z[:, LP_Measurement.PY] = X[:, LP_State.PY]
        Z[:, LP_Measurement.PZ] = X[:, LP_State.PZ]
        Z[:, LP_Measurement.YAW] = self._wrap(X[:, LP_State.YAW])

    def _buffer_push(self, timestamp: float, quad_vel, quad_accel):
        """Append current (x, P) snapshot and prune entries older than _OOSM_BUFFER_S relative to the newest entry.

        :param timestamp: Timestamp to add to buffer
        """
        # Save UKF state to buffer
        self._buffer.append(
            (timestamp, self.x.copy(), self.P.copy(), self.X_prop.copy(), quad_vel, quad_accel)
        )

        # Prune stale snapshots from the front
        OOSM_BUFFER_S = 0.200
        cutoff = timestamp - OOSM_BUFFER_S
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def _repair_P(self):
        """Force P back to symmetric positive definite via eigendecomposition"""
        self.P = 0.5 * (self.P + self.P.T)
        eigvals, eigvecs = np.linalg.eigh(self.P)
        eigvals = np.maximum(eigvals, 1e-6)
        self.P = eigvecs @ np.diag(eigvals) @ eigvecs.T

    @staticmethod
    def _wrap(angle: np.ndarray) -> np.ndarray:
        """Wraps the angle in the domain -π to π

        :param angle: Single float or vector of angles
        :return: Wrapped angle
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
        """Weighted circular mean — safe across the ±π boundary

        :param angles: Vector of angles
        :param weights: Vector of weights for the weighted average
        :return: The weighted mean of the angles
        """
        sin_mean = np.sum(weights * np.sin(angles))
        cos_mean = np.sum(weights * np.cos(angles))
        return float(np.arctan2(sin_mean, cos_mean))
