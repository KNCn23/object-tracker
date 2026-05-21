"""
Constant-velocity Kalman filter for bounding-box tracking.
State vector: [cx, cy, s, r, vx, vy, vs]
    cx, cy : center coordinates
    s      : scale (area = w * h)
    r      : aspect ratio (w / h)
    v*     : velocities of cx, cy, s
"""

import numpy as np


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox: np.ndarray):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # State and covariance
        self.x = np.zeros((7, 1))
        self.x[:4] = self._bbox_to_z(bbox)
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # high uncertainty on velocities

        # Constant-velocity transition matrix
        self.F = np.eye(7)
        self.F[0, 4] = 1
        self.F[1, 5] = 1
        self.F[2, 6] = 1

        # Observation matrix (we observe cx, cy, s, r)
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        # Noise
        self.Q = np.eye(7) * 0.01
        self.Q[4:, 4:] *= 0.1
        self.R = np.eye(4) * 1.0
        self.R[2:, 2:] *= 10.0

        self.time_since_update = 0
        self.hits = 1
        self.age = 0

    # ── Bounding-box ↔ state conversion ────────────────────────────────
    @staticmethod
    def _bbox_to_z(bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        s = w * h
        r = w / max(h, 1e-6)
        return np.array([[cx], [cy], [s], [r]])

    @staticmethod
    def _z_to_bbox(z):
        cx, cy, s, r = z[0, 0], z[1, 0], z[2, 0], z[3, 0]
        w = np.sqrt(max(s * r, 1e-6))
        h = s / max(w, 1e-6)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    # ── Predict / update ───────────────────────────────────────────────
    def predict(self):
        # Prevent negative scale
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self._z_to_bbox(self.x[:4])

    def update(self, bbox):
        z = self._bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1

    @property
    def bbox(self):
        return self._z_to_bbox(self.x[:4])
