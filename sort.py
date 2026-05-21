"""
Simple Online and Realtime Tracker (SORT-style).
Associates detections to tracks using IoU + Hungarian assignment.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_tracker import KalmanBoxTracker


def iou(b1: np.ndarray, b2: np.ndarray) -> float:
    xx1 = max(b1[0], b2[0]); yy1 = max(b1[1], b2[1])
    xx2 = min(b1[2], b2[2]); yy2 = min(b1[3], b2[3])
    w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
    inter = w * h
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


class Sort:
    """
    Args:
        max_age:    frames a track may go undetected before deletion
        min_hits:   minimum hits before a track is reported
        iou_thresh: minimum IoU to allow association
    """
    def __init__(self, max_age: int = 10, min_hits: int = 3,
                 iou_thresh: float = 0.3):
        self.max_age    = max_age
        self.min_hits   = min_hits
        self.iou_thresh = iou_thresh
        self.trackers   = []
        self.frame_count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        detections : (N, 4) array of [x1, y1, x2, y2]
        Returns    : (M, 5) array of [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        # ── Predict existing tracks forward ────────────────────────────
        predicted = []
        to_keep = []
        for t in self.trackers:
            pred = t.predict()
            if np.any(np.isnan(pred)):
                continue
            predicted.append(pred)
            to_keep.append(t)
        self.trackers = to_keep
        predicted = np.array(predicted) if predicted else np.empty((0, 4))

        # ── Associate via Hungarian on IoU cost ────────────────────────
        matched, unmatched_dets, unmatched_trks = \
            self._associate(detections, predicted)

        # Update matched trackers
        for d_idx, t_idx in matched:
            self.trackers[t_idx].update(detections[d_idx])

        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[d_idx]))

        # Remove stale trackers
        self.trackers = [t for t in self.trackers
                         if t.time_since_update <= self.max_age]

        # Output confirmed tracks
        out = []
        for t in self.trackers:
            if t.time_since_update == 0 and \
               (t.hits >= self.min_hits or self.frame_count <= self.min_hits):
                out.append(np.concatenate([t.bbox, [t.id]]))
        return np.array(out) if out else np.empty((0, 5))

    def _associate(self, dets, trks):
        if len(dets) == 0 or len(trks) == 0:
            return [], list(range(len(dets))), list(range(len(trks)))

        cost = np.zeros((len(dets), len(trks)))
        for i, d in enumerate(dets):
            for j, t in enumerate(trks):
                cost[i, j] = -iou(d, t)  # negate so we minimize

        row, col = linear_sum_assignment(cost)
        matched, u_d, u_t = [], [], []
        for i in range(len(dets)):
            if i not in row:
                u_d.append(i)
        for j in range(len(trks)):
            if j not in col:
                u_t.append(j)
        for i, j in zip(row, col):
            if -cost[i, j] < self.iou_thresh:
                u_d.append(i); u_t.append(j)
            else:
                matched.append((i, j))
        return matched, u_d, u_t
