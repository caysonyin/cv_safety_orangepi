# MindYOLO / MindSpore / CANN Integration

This guide explains how the `object_protection` subsystem wires MindYOLO into MindSpore for deployment on Ascend / CANN hardware, as well as how the project degrades gracefully on hosts without Ascend runtime support or GUI access.

## Runtime Targets

| Device target | How to enable | Notes |
| --- | --- | --- |
| Ascend (default) | `python object_protection/integrated_safety_monitor.py --device-target Ascend` | Requires an Ascend NPU, the CANN runtime, and an Ascend-enabled MindSpore build. |
| CPU fallback | `--device-target CPU` | Available out of the box; throughput is much lower than Ascend. |
| GPU fallback | `--device-target GPU` | Requires a CUDA-enabled MindSpore build; not validated on this repo. |

The detector honours `--device-id` (defaults to `DEVICE_ID` env var or `0`).

If the requested target is unsupported by the local MindSpore build, the adapter surfaces a descriptive error such as:

```
当前MindSpore构建不支持设备类型 'Ascend'，请确认已正确安装对应的硬件运行时，或使用 --device-target CPU/GPU 运行。
```

Switch to a supported target or reinstall the correct MindSpore wheel.

## Ascend Setup Checklist

1. **Install Ascend firmware + driver** matching your hardware (e.g., Orange Pi AI Pro or Atlas).  
   Refer to the official device documentation for kernel/driver requirements.
2. **Install the CANN toolkit** (8.x tested). Export the main path so MindSpore can locate runtime libraries:
   ```bash
   export ASCEND_HOME=/usr/local/Ascend
   export LD_LIBRARY_PATH=$ASCEND_HOME/runtime/lib64:$LD_LIBRARY_PATH
   ```
3. **Install MindSpore for Ascend** that matches your Python, OS, and CANN version:
   ```bash
   pip install mindspore-ascend==2.3.1
   ```
4. **Verify device visibility**:
   ```bash
   python - <<'PY'
   import mindspore as ms
   ms.set_context(device_target="Ascend", device_id=0)
   print("Ascend context ready")
   PY
   ```

### Distributed (Multi-Device) Notes

- MindYOLO’s config enables `sync_bn: True`. The adapter automatically **disables SyncBatchNorm** unless MindSpore’s distributed communication has already been initialised (to avoid the “Distributed Communication has not been inited” crash on single-card runs).
- For true multi-device inference you must initialise communication yourself (e.g., via `msrun`, `mpirun`, or adding a call to `mindspore.communication.init()` before you construct the detector).

## Checkpoint & Config Management

- The first run downloads the official MindYOLO YOLOv7-tiny checkpoint to `yolov7-tiny.ckpt`.  
  Offline hosts can place the file manually and skip the download step.
- The default config lives at `mindyolo/configs/yolov7/yolov7-tiny.yaml`.  
  Custom configs may override the `sync_bn` and precision settings; the adapter respects those values but applies the distributed safety checks described above.

## Headless (No GUI) Operation

Many Ascend servers ship without a desktop session. When OpenCV cannot create a GUI window:

- `video_relic_tracking.py` and the integrated monitor auto-detect the failure and switch to **headless mode**.
- The selection workflow is skipped; the monitor enters `monitoring` stage immediately.
- Toast messages are echoed as console logs (prefixed with `[提示]`).
- Frame rendering continues in memory, but no windows are shown. This allows unattended inference or remote execution over SSH.

If you need the interactive selection UI, run the scripts within an environment that supports X11/Wayland (e.g., `ssh -X`, VNC, or a local desktop).

## Troubleshooting

| Symptom | Likely cause | Mitigation |
| --- | --- | --- |
| `Unsupported device target Ascend` | MindSpore installed without Ascend support | Install the Ascend-specific wheel or switch to CPU/GPU targets. |
| `Distributed Communication has not been inited` | Config requested SyncBatchNorm with no communication | The adapter now disables SyncBatchNorm automatically. For distributed runs, call `mindspore.communication.init()` first. |
| `libge_runner.so: cannot open shared object file` | Missing CANN runtime in `LD_LIBRARY_PATH` | Source the CANN environment scripts or export the runtime paths manually. |
| No window appears | Running without a display server | Headless mode is expected; check console logs for `[提示]`. |
| Pose model download failure | No network access | Pre-download `models/pose_landmarker_full.task` (see `WebcamPoseDetection/download_model.py`). |

For additional background on the safety monitor pipeline, see `docs/object_protection.md`. The root `README.md` summarises quick-start commands and links back to this document.

