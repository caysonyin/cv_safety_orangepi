# System Overview

This note captures the final integration state of the CV Safety System. It highlights the coupling between the cup tracking workflow and the pose-driven safety monitor so the project can be evaluated without referring to earlier build logs.

## Runtime Components

- **WebcamPoseDetection** supplies MediaPipe pose landmarks that power intrusion analysis.
- **object_protection/video_relic_tracking.py** tracks cups, applies centroid IDs, and exposes interaction hooks for designating protected inventory.
- **object_protection/integrated_safety_monitor.py** fuses YOLO detections, pose landmarks, and tracker state to monitor cups and detect tennis rackets.

## Final Behaviour

1. Cups are the only selectable protected objects. Clicking a detection toggles its selection and spawns an expanded safety fence.
2. Tennis rackets are treated as the sole hazardous objects. When one is detected, the system associates it with the nearest tracked person and raises a carry alert.
3. Pose landmarks determine whether a person breaches a cup’s safety fence. Intrusions and hazard alerts are logged in the on-screen history pane.
4. All overlays, keyboard shortcuts, and logging statements are aligned with the English-language UI and the cup/tennis racket configuration.

Use this summary as the canonical reference for demonstrations, acceptance tests, and future hand-off documents.
