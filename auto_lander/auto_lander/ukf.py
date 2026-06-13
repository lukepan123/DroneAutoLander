#!/usr/bin/env python3
import numpy as np

from .state_definitions import LP_State, LP_Measurement


class UKF:
    """ Defines the UKF class for the landing platform"""
    def __init__(self,  dt):
        """ Initialise the UKF
        """

        self.dt = dt

        # Timestamp of the last predict/update cycle (seconds, ROS clock).
        # None until the first measurement arrives.
        self.last_update_time: float | None = None
        # Dimensions
        self.dim_x = len(LP_State)
        self.dim_z = len(LP_Measurement)

        # UKF scaling parameters (Merwe)
        self.alpha = 0.30
        self.beta = 2.0
        self.kappa = 0.0

        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        self.gamma = np.sqrt(self.dim_x + self.lambda_)

        # Number of sigma points
        self.num_sigma = 2 * self.dim_x + 1

        # Preallocate sigma arrays
        self.X      = np.zeros((self.num_sigma, self.dim_x))  # sigma points
        self.X_prop = np.zeros((self.num_sigma, self.dim_x))  # fx result
        self.Z      = np.zeros((self.num_sigma, self.dim_z))  # hx result

        # Initial state
        self.x = np.zeros(self.dim_x)

        self.P = np.diag([
            0.15, 0.15, 0.15,
            0.50, 1.50,
            0.15, 0.50,
        ])

        # Process noise (make dependent on dt)
        self.Q = 0.01/dt * np.diag([
            0.05, 0.05, 0.05,    # px, py, pz
            0.10, 1.00,          # v, a
            0.02, 0.10,          # yaw, yaw_rate
        ])

        # Measurement noise
        self.R = np.diag([
            0.01, 0.01, 0.01,
            0.01
        ])

        # Precompute weights
        self.Wm = np.full(self.num_sigma, 0.5 / (self.dim_x + self.lambda_))
        self.Wc = np.full(self.num_sigma, 0.5 / (self.dim_x + self.lambda_))

        self.Wm[0] = self.lambda_ / (self.dim_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.dim_x + self.lambda_) + (1 - self.alpha**2 + self.beta)


    def predict(self, dt):
        """ Do prediction step for UKF.
        """
        # 0) Guarantee P is symmetric positive definite before proceeding, repair if needed
        try:
            S = np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            self._repair_P()
            S = np.linalg.cholesky(self.P)

        n = self.dim_x

        # 1) Cholesky of covariance
        S = self.gamma * S

        # 2) Generate sigma points
        self.X[0] = self.x
        for i in range(n):
            col = S[:, i]
            self.X[i + 1]     = self.x + col
            self.X[n + i + 1] = self.x - col

        # 3) Propagate through fx
        self._fx_vectorized(self.X, self.X_prop, dt)

        # 4) Predicted mean
        self.x[:] = (self.Wm[:, None] * self.X_prop).sum(axis=0)
        self.x[LP_State.YAW] = self._circular_mean(self.X_prop[:, LP_State.YAW], self.Wm)  # yaw is index 7

        # 5) Predicted covariance
        dX = self.X_prop - self.x
        dX[:, LP_State.YAW] = self._wrap(dX[:, LP_State.YAW])  # wrap yaw deviations

        self.P = (dX.T * self.Wc) @ dX + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def get_predicted_state(self, dt):
        """
        Shadow predict — project current mean state forward by one dt
        to compensate for UKF pipeline delay. Propagates sigma points
        to avoid bias from the nonlinear CTRA model.
        """
        X_in  = np.tile(self.x, (self.num_sigma, 1))   # broadcast mean as all sigma points
        X_out = np.zeros_like(X_in)

        self._fx_vectorized(X_in, X_out, dt)

        # Only the mean row matters — all rows are identical so just take first
        x_pred = X_out[0].copy()
        x_pred[LP_State.YAW] = self._wrap(x_pred[LP_State.YAW])

        return x_pred


    def update(self, z, mahal_threshold: float = 5.0) -> bool:
        """ Do update step for UKF.

        :param z: Measurement of new landing pad pose
        :param mahal_threshold: Measurements whose Mahalanobis distance exceeds
                                this value are rejected as outliers. Returns False
                                when the measurement is rejected, True otherwise.
        """
        # 1) Propagate sigma points through hx
        self._hx_vectorized(self.X_prop, self.Z)

        # 2) Predicted measurement mean
        z_pred = (self.Wm[:, None] * self.Z).sum(axis=0)
        z_pred[LP_Measurement.YAW] = self._circular_mean(self.Z[:, LP_Measurement.YAW], self.Wm)  # circular mean for yaw

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

        # Mahalanobis gate: reject if innovation is statistically inconsistent
        # with the predicted measurement covariance S.
        mahal_sq = float(y @ np.linalg.solve(S, y))
        if mahal_sq > mahal_threshold ** 2:
            return False  # discard measurement — state and covariance unchanged

        # 9) Update state
        self.x   += K @ y
        self.x[LP_State.YAW] = self._wrap(self.x[LP_State.YAW])

        # 10) Covariance update
        self.P -= K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True
    

    def _fx_vectorized(self, X, Y, dt):
        """ Vectorised state model updater.

        :param X: Old state vector 
        :param Y: New state vector 
        :param dt: UKF timestep 
        """

        px       = X[:, LP_State.PX]
        py       = X[:, LP_State.PY]
        pz       = X[:, LP_State.PZ]
        v        = X[:, LP_State.V]
        a        = X[:, LP_State.A]
        yaw      = X[:, LP_State.YAW]
        omega    = X[:, LP_State.YAW_RATE]

        yaw_new = yaw + omega * dt

        # Threshold for "straight line" motion
        eps = 1e-4

        turning = np.abs(omega) > eps
        straight = ~turning

        # Allocate outputs
        px_new = np.empty_like(px)
        py_new = np.empty_like(py)

        # Straight-line motion
        px_new[straight] = (
            px[straight]
            + v[straight] * np.cos(yaw[straight]) * dt
            + 0.5 * a[straight] * np.cos(yaw[straight]) * dt**2
        )

        py_new[straight] = (
            py[straight]
            + v[straight] * np.sin(yaw[straight]) * dt
            + 0.5 * a[straight] * np.sin(yaw[straight]) * dt**2
        )

        # Constant Turn Rate + Acceleration
        w  = omega[turning]
        y0 = yaw[turning]
        y1 = yaw_new[turning]
        vt = v[turning]
        at = a[turning]

        px_new[turning] = (
            px[turning]
            + vt / w * (np.sin(y1) - np.sin(y0))
            + at / w**2 * (
                np.cos(y1)
                - np.cos(y0)
                + w * dt * np.sin(y1)
            )
        )

        py_new[turning] = (
            py[turning]
            + vt / w * (-np.cos(y1) + np.cos(y0))
            + at / w**2 * (
                np.sin(y1)
                - np.sin(y0)
                - w * dt * np.cos(y1)
            )
        )

        Y[:, LP_State.PX] = px_new
        Y[:, LP_State.PY] = py_new
        Y[:, LP_State.PZ] = pz

        Y[:, LP_State.V] = v + a * dt
        Y[:, LP_State.A] = a

        Y[:, LP_State.YAW] = self._wrap(yaw_new)
        Y[:, LP_State.YAW_RATE] = omega
        

    def _hx_vectorized(self, X, Z):
        """ Vectorised measurement model

        :param X: State vector
        :param Z: Measurement vector
        """  
        Z[:, LP_Measurement.PX] = X[:, LP_State.PX]
        Z[:, LP_Measurement.PY] = X[:, LP_State.PY]
        Z[:, LP_Measurement.PZ] = X[:, LP_State.PZ]
        Z[:, LP_Measurement.YAW] = self._wrap(X[:, LP_State.YAW])


    def _repair_P(self):
        """ Force P back to symmetric positive definite via eigendecomposition
        """
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