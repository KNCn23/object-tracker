"""
Real-time multi-object tracker — main entry point.
Usage:
    python main.py                # webcam, motion detector
    python main.py --source video.mp4
    python main.py --detector color
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

from detectors import MotionDetector, ColorBlobDetector
from sort import Sort


def hsv_color(track_id: int) -> tuple:
    h = (track_id * 37) % 180
    bgr = cv2.cvtColor(np.uint8([[[h, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",   default="0",
                    help="Webcam index or path to video")
    ap.add_argument("--detector", choices=["motion", "color"],
                    default="motion")
    ap.add_argument("--max-age",  type=int, default=15)
    ap.add_argument("--min-hits", type=int, default=2)
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {args.source}")

    detector = (MotionDetector() if args.detector == "motion"
                else ColorBlobDetector())
    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits)

    fps_buf = deque(maxlen=30)
    trails  = {}  # track_id -> deque of past centers

    while True:
        ok, frame = cap.read()
        if not ok: break

        t0 = time.time()
        dets   = detector.detect(frame)
        tracks = tracker.update(dets)
        dt = (time.time() - t0) * 1000.0  # ms
        fps_buf.append(dt)
        fps = 1000.0 / (np.mean(fps_buf) + 1e-6)

        # Draw raw detections (thin gray)
        for x1, y1, x2, y2 in dets:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (180, 180, 180), 1)

        # Draw tracks
        for x1, y1, x2, y2, tid in tracks:
            tid = int(tid)
            color = hsv_color(tid)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          color, 2)
            label = f"ID {tid}"
            cv2.putText(frame, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
            trails.setdefault(tid, deque(maxlen=30)).append((cx, cy))
            pts = list(trails[tid])
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], color, 2)

        # HUD: FPS, latency, active tracks
        hud = f"FPS {fps:5.1f}   Latency {dt:5.1f} ms   Tracks {len(tracks)}"
        cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.imshow("object-tracker", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
