# CV Safety System

A consolidated computer vision toolkit for safeguarding exhibit areas. The repository provides two runtime-ready subsystems that can operate independently or in tandem:

- **WebcamPoseDetection** — MediaPipe-based, 33-keypoint human pose estimation with minimal, developer-friendly, and performance-optimized pipelines.
- **object_protection** — MindYOLO (MindSpore) powered detection with interactive object tracking and an integrated safety monitor that cross-references pose landmarks with detected items. The pipeline is tuned for Ascend/CANN hardware such as the Orange Pi acceleration module.

The current configuration treats **cups as the protected exhibit proxy** and flags a **tennis racket as the hazardous object**. All UI labels, filtering logic, and alerts align with this setup.

## Key Capabilities

- Real-time MindYOLO YOLOv7-tiny inference accelerated by MindSpore on Ascend/CANN devices, with centroid tracking and interactive selection of protected cups.
- Automatic safety fence calculation around selected cups with intrusion detection based on MediaPipe pose landmarks.
- Hazard detection workflow that highlights tennis rackets and binds them to the nearest tracked person for alerting.
- Ascend-friendly runtime tested on Python 3.9 / Ubuntu 22.04 with MindSpore 2.3 and CANN 8.x.

## Quick Start

```bash
# Install MindSpore (Ascend build). See https://www.mindspore.cn/install for board-specific wheels.
pip install mindspore-ascend==2.3.1

# Install shared Python dependencies (OpenCV, MediaPipe, etc.)
pip install -r requirements.txt

# Download MediaPipe pose model
python WebcamPoseDetection/download_model.py

# Run the cup tracker (default webcam)
python object_protection/video_relic_tracking.py --source 0

# Run the integrated safety monitor (cups + tennis rackets)
python object_protection/integrated_safety_monitor.py --source 0
```

Each script accepts `--source` to select a camera index or video file.  
Use `--device-target`, `--device-id`, or `--weight` if you need to override the default Ascend runtime or point to a custom MindYOLO checkpoint.  
The first MindYOLO run downloads the official MindSpore checkpoint (`yolov7-tiny.ckpt`) automatically unless the file already exists.

```bash
curl -L -o yolov7-tiny.ckpt \
  https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-d8972c94.ckpt
```

To use the weights fine-tuned on our museum dataset, download them from Google Drive and place the file alongside the default weights:

- https://drive.google.com/drive/folders/1xjeqz_GMzl5gKie2LgOHv-iQns_v3mU4?usp=share_link

## MindYOLO / MindSpore / CANN Integration

- Ascend hardware is the primary target. CPU (`--device-target CPU`) and GPU fallbacks also work when the corresponding MindSpore wheels are installed.
- SyncBatchNorm in the MindYOLO config is auto-disabled unless MindSpore’s distributed communication has been initialised, preventing single-card crashes.
- When OpenCV cannot create a GUI window (common on headless servers), the safety monitor switches to a console-only mode and logs toast messages with a `[提示]` prefix.

See [docs/mindyolo_mindspore_cann.md](docs/mindyolo_mindspore_cann.md) for a full setup checklist, troubleshooting tips, and offline deployment guidance.

## Repository Layout

```
cv_safety_sys/
├── WebcamPoseDetection/       # Pose estimation subsystem
├── object_protection/         # Cup tracking and safety monitor
└── docs/                      # Detailed subsystem documentation (see mindyolo_mindspore_cann.md for Ascend setup)
```

See the `docs/` directory for subsystem deep dives and implementation notes that correspond to the final configuration described above.
