# Webcam Pose Detection

The pose subsystem supplies the MediaPipe landmarks used by the safety monitor. It ships with three runtimes that differ only in structure and performance characteristics, all aligned with the final project configuration.

## Module Snapshot

| File | Purpose |
| --- | --- |
| `webcam_pose_minimal.py` | Compact entry point for verifying the MediaPipe runtime and camera access. |
| `webcam_pose_simple.py` | Class-based implementation with FPS statistics for downstream integrations. |
| `pose33_realtime_optimized.py` | Threaded producer/consumer pipeline for latency-sensitive deployments. |
| `download_model.py` | Fetches `pose_landmarker_full.task` into `models/`. |
| `test_setup.py` | Optional health check to validate camera access and dependency versions. |

## Usage Notes

- Install dependencies with `pip install -r requirements.txt` and download the model once via `python WebcamPoseDetection/download_model.py`.
- Every script accepts `--source` so camera indices and prerecorded footage can be swapped without code changes.
- The optimized runtime (`pose33_realtime_optimized.py`) pairs well with the integrated monitor when the cup safety workflow needs lower latency pose updates.

## Integration Highlights

- Pose landmarks are consumed directly by `IntegratedSafetyMonitor` to detect cup fence intrusions.
- Frame processing uses MediaPipe’s VIDEO mode to minimise jitter and align landmarks with YOLO detections.
- All scripts emit English-language logs to match the updated UI across the repository.
