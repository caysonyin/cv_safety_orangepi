# WebcamPoseDetection Structure

```
WebcamPoseDetection/
├── download_model.py
├── pose33_realtime_optimized.py
├── webcam_pose_minimal.py
├── webcam_pose_simple.py
├── test_setup.py
└── __init__.py (optional)
```

## Code Walkthrough

- **`webcam_pose_minimal.py`**  
  Straightforward script wiring camera capture, MediaPipe inference, and landmark rendering. Ideal for sanity checks.

- **`webcam_pose_simple.py`**  
  Wraps the runtime in `SimpleWebcamPoseDetector`, exposing FPS measurements and predictable error handling for reuse.

- **`pose33_realtime_optimized.py`**  
  Splits acquisition and inference across threads using a queue, adds resolution scaling helpers, and exposes CLI switches for deployment scenarios.

- **`download_model.py`**  
  Fetches `pose_landmarker_full.task` to `models/`. Run once before the other scripts if the model is missing.

- **`test_setup.py`**  
  Optional pre-flight check covering camera availability and dependency versions.

## Link to the Safety Monitor

- The integrated safety monitor imports `download_model.py` to locate or download the pose task file.
- Landmark output from any pose runtime can be forwarded directly into `IntegratedSafetyMonitor` without additional adaptation.
- Dependencies are shared with the rest of the repository through the root `requirements.txt`, ensuring a single environment definition for the final release.
