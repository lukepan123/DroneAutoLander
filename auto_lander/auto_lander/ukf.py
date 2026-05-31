import numpy as np


class UKF:
    def __init__(self, dt):
        self.dt = dt

        # -------------------------------
        # Dimensions
        # -------------------------------
        self.dim_x = 8  # px, py, pz, vx, vy, ax, ay, yaw
        self.dim_z = 4

        # UKF scaling parameters (Merwe)
        self.alpha = 0.30
        self.beta = 2.0
        self.kappa = 0.0

        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        self.gamma = np.sqrt(self.dim_x + self.lambda_)

        # Number of sigma points
        self.num_sigma = 2 * self.dim_x + 1

        # -------------------------------
        # Preallocate sigma arrays
        # -------------------------------
        self.X      = np.zeros((self.num_sigma, self.dim_x))  # sigma points
        self.X_prop = np.zeros((self.num_sigma, self.dim_x))  # fx result
        self.Z      = np.zeros((self.num_sigma, self.dim_z))  # hx result

        # -------------------------------
        # Initial state
        # -------------------------------
        self.x = np.zeros(self.dim_x)

        self.P = np.diag([
            0.15, 0.15, 0.15,
            0.5,  0.5,
            1.5,  1.5,
            0.15
        ])

        # -------------------------------
        # Process noise
        # -------------------------------
        self.Q = np.diag([
            0.001, 0.001, 0.001,  # px, py, pz
            0.20,  0.20,          # vx, vy
            2.00,  2.00,          # ax, ay
            0.01                  # yaw
        ])

        # -------------------------------
        # Measurement noise
        # -------------------------------
        self.R = np.diag([
            0.20, 0.20, 0.20,
            0.05
        ])

        # -------------------------------
        # Precompute weights
        # -------------------------------
        self.Wm = np.full(self.num_sigma, 0.5 / (self.dim_x + self.lambda_))
        self.Wc = np.full(self.num_sigma, 0.5 / (self.dim_x + self.lambda_))

        self.Wm[0] = self.lambda_ / (self.dim_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.dim_x + self.lambda_) \
                     + (1 - self.alpha**2 + self.beta)

    # =========================================================
    #                  PREDICTION STEP
    # =========================================================
    def predict(self):
        # 0) Guarantee P is symmetric positive definite before Cholesky
        self._repair_P()

        n = self.dim_x

        # 1) Cholesky of covariance
        S = np.linalg.cholesky(self.P)
        S = self.gamma * S

        # 2) Generate sigma points
        self.X[0] = self.x
        for i in range(n):
            col = S[:, i]
            self.X[i + 1]     = self.x + col
            self.X[n + i + 1] = self.x - col

        # 3) Propagate through fx
        self._fx_vectorized(self.X, self.X_prop, self.dt)

        # 4) Predicted mean
        self.x[:] = (self.Wm[:, None] * self.X_prop).sum(axis=0)
        self.x[7] = self._circular_mean(self.X_prop[:, 7], self.Wm)  # yaw is index 7

        # 5) Predicted covariance
        dX = self.X_prop - self.x
        dX[:, 7] = self._wrap_array(dX[:, 7])  # wrap yaw deviations

        self.P[:] = (dX.T * self.Wc) @ dX + self.Q
        self.P    = 0.5 * (self.P + self.P.T)

    # =========================================================
    #                  UPDATE STEP
    # =========================================================
    def update(self, z):
        # 1) Propagate sigma points through hx
        self._hx_vectorized(self.X_prop, self.Z)

        # 2) Predicted measurement mean
        z_pred    = (self.Wm[:, None] * self.Z).sum(axis=0)
        z_pred[3] = self._circular_mean(self.Z[:, 3], self.Wm)  # circular mean for yaw

        # 3) Measurement deviations
        dZ       = self.Z - z_pred
        dZ[:, 3] = self._wrap_array(dZ[:, 3])

        # 4) State deviations
        dX       = self.X_prop - self.x
        dX[:, 7] = self._wrap_array(dX[:, 7])

        # 5) Cross covariance
        P_xz = (dX.T * self.Wc) @ dZ

        # 6) Innovation covariance
        S = (dZ.T * self.Wc) @ dZ + self.R

        # 7) Kalman gain
        K = P_xz @ np.linalg.inv(S)

        # 8) Update state
        y    = z - z_pred
        y[3] = self._wrap(y[3])

        self.x   += K @ y
        self.x[7] = self._wrap(self.x[7])

        # 9) Covariance update — Joseph form (numerically stable, no _last_H needed)
        # Approximate H from sigma point deviations via least squares
        H   = np.linalg.lstsq(self.P, P_xz, rcond=None)[0].T  # (dim_z, dim_x)
        IKH = np.eye(self.dim_x) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # =========================================================
    #               Vectorized fx
    # =========================================================
    def _fx_vectorized(self, X, Y, dt):
        px  = X[:, 0]
        py  = X[:, 1]
        pz  = X[:, 2]
        vx  = X[:, 3]
        vy  = X[:, 4]
        ax  = X[:, 5]
        ay  = X[:, 6]
        yaw = X[:, 7]

        # Position update (with acceleration term)
        Y[:, 0] = px + vx * dt + 0.5 * ax * dt**2
        Y[:, 1] = py + vy * dt + 0.5 * ay * dt**2
        Y[:, 2] = pz

        # Velocity update
        Y[:, 3] = vx + ax * dt
        Y[:, 4] = vy + ay * dt

        # Acceleration — random walk
        Y[:, 5] = ax
        Y[:, 6] = ay

        # Yaw — random walk
        Y[:, 7] = self._wrap_array(yaw)

    # =========================================================
    #               Vectorized hx
    # =========================================================
    def _hx_vectorized(self, X, Z):
        Z[:, 0] = X[:, 0]
        Z[:, 1] = X[:, 1]
        Z[:, 2] = X[:, 2]
        Z[:, 3] = self._wrap_array(X[:, 7])

    # =========================================================
    #               Covariance repair
    # =========================================================
    def _repair_P(self):
        """Force P back to symmetric positive definite via eigendecomposition."""
        self.P = 0.5 * (self.P + self.P.T)
        eigvals, eigvecs = np.linalg.eigh(self.P)
        eigvals = np.maximum(eigvals, 1e-6)
        self.P = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # =========================================================
    #               Angle helpers
    # =========================================================
    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _wrap_array(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
        """Weighted circular mean — safe across the ±π boundary."""
        sin_mean = np.sum(weights * np.sin(angles))
        cos_mean = np.sum(weights * np.cos(angles))
        return float(np.arctan2(sin_mean, cos_mean))