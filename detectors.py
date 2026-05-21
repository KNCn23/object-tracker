"""Two interchangeable detectors: motion (frame-diff) and color-blob."""

import cv2
import numpy as np


class MotionDetector:
    """Background subtraction via MOG2 — works on any moving camera/video."""
    def __init__(self, min_area: int = 800):
        self.sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=False)
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> np.ndarray:
        fg = self.sub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append([x, y, x + w, y + h])
        return np.array(boxes) if boxes else np.empty((0, 4))


class ColorBlobDetector:
    """Detects blobs of a chosen HSV color range. Demo-friendly."""
    def __init__(self, lower=(35, 80, 60), upper=(85, 255, 255),
                 min_area: int = 400):
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append([x, y, x + w, y + h])
        return np.array(boxes) if boxes else np.empty((0, 4))
