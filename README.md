# Real-time Multi-Object Tracker

A lightweight, dependency-light multi-object tracker built around a constant-velocity Kalman filter and Hungarian assignment — the classic SORT pipeline, written from scratch in roughly 200 lines.

## Pipeline

```
   ┌──────────┐    ┌───────────────────┐    ┌──────────────┐    ┌────────┐
   │  frame   │──▶ │ Detector          │──▶ │ Hungarian    │──▶ │ Kalman │──▶ tracked boxes
   │ (BGR)    │    │  (motion / color) │    │ matching (IoU)│    │ update │
   └──────────┘    └───────────────────┘    └──────────────┘    └────────┘
```

## Features

| Component | Details |
|---|---|
| **Kalman filter** | 7-state constant-velocity (cx, cy, s, r, vx, vy, vs) |
| **Data association** | Hungarian algorithm on IoU cost matrix |
| **Track management** | Birth (min-hits), death (max-age), unique IDs |
| **Detectors** | MOG2 motion-difference, HSV color-blob |
| **HUD overlay** | Live FPS, per-frame latency, active track count |
| **Trails** | Last 30 center positions per track |

## Run

```bash
pip install -r requirements.txt

# Webcam, motion-based detection
python main.py

# Video file, color-blob detection (green by default)
python main.py --source race.mp4 --detector color
```

Press `q` or `Esc` to quit.

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--source`   | `0` (webcam)  | Camera index or video path |
| `--detector` | `motion`      | `motion` or `color` |
| `--max-age`  | `15`          | Frames a lost track can persist |
| `--min-hits` | `2`           | Detections needed before reporting |

## Files

```
├── kalman_tracker.py  # KalmanBoxTracker: predict / update / state ↔ bbox
├── sort.py            # Sort: per-frame association + lifecycle management
├── detectors.py       # MotionDetector (MOG2), ColorBlobDetector (HSV)
├── main.py            # Capture loop + visualization
└── requirements.txt
```

## Notes

- Plug in any detector that outputs an `(N, 4)` array of `[x1, y1, x2, y2]` boxes — e.g. swap in YOLOv8 (`pip install ultralytics`) by replacing the `detector.detect()` call in `main.py`.
- The Kalman state assumes bounding-box scale and aspect ratio change slowly. Adjust `Q` / `R` in `kalman_tracker.py` for very fast or very small targets.

## License

MIT
