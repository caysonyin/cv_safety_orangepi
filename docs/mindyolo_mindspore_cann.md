# MindYOLO / MindSpore / CANN 集成指南

本文档介绍 `object_protection` 子系统如何在 Ascend / CANN 硬件上通过 MindSpore 调用 MindYOLO，并说明在没有 Ascend 运行时或缺乏图形界面的环境下，系统如何优雅降级。

## 运行目标

| 设备类型 | 启动方式 | 说明 |
| --- | --- | --- |
| Ascend（默认） | `python object_protection/integrated_safety_monitor.py --device-target Ascend` | 需要 Ascend NPU、CANN 运行时以及支持 Ascend 的 MindSpore 构建。 |
| CPU 回退 | `--device-target CPU` | 即装即用，但吞吐量远低于 Ascend。 |
| GPU 回退 | `--device-target GPU` | 需要 CUDA 版本的 MindSpore，本仓库未做完整验证。 |

检测器同样支持 `--device-id`，默认读取环境变量 `DEVICE_ID`，若未设置则取 0。

当本地 MindSpore 构建不支持指定设备类型时，适配层会给出类似提示：

```
当前MindSpore构建不支持设备类型 'Ascend'，请确认已正确安装对应的硬件运行时，或使用 --device-target CPU/GPU 运行。
```

此时请切换到已支持的设备类型，或安装相匹配的 MindSpore 轮子。

## Ascend 环境核查清单

1. **安装与硬件匹配的 Ascend 固件与驱动**（如 Orange Pi AI Pro 或 Atlas），按官方指南满足内核与驱动要求。  
2. **安装 CANN 工具链**（已在 8.x 版本验证），并导出核心路径，便于 MindSpore 查找到运行时库：
   ```bash
   export ASCEND_HOME=/usr/local/Ascend
   export LD_LIBRARY_PATH=$ASCEND_HOME/runtime/lib64:$LD_LIBRARY_PATH
   ```
3. **安装对应版本的 Ascend 版 MindSpore**，确保与 Python、操作系统和 CANN 版本匹配：
   ```bash
   pip install mindspore-ascend==2.3.1
   ```
4. **验证设备可见性**：
   ```bash
   python - <<'PY'
   import mindspore as ms
   ms.set_context(device_target="Ascend", device_id=0)
   print("Ascend context ready")
   PY
   ```

### 分布式与多设备提示

- MindYOLO 默认配置中 `sync_bn: True`。适配层会在 MindSpore 分布式通信尚未初始化时**自动关闭 SyncBatchNorm**，避免单卡运行时出现 “Distributed Communication has not been inited” 崩溃。
- 若确实需要多卡推理，请提前自行初始化通信（例如使用 `msrun`、`mpirun`，或在构建检测器前调用 `mindspore.communication.init()`）。

## 模型与配置管理

- 首次运行会自动将官方 MindYOLO YOLOv7-tiny 权重下载为 `yolov7-tiny.ckpt`。  
  离线环境可手动放置此文件，以跳过下载。
- 默认配置位于 `mindyolo/configs/yolov7/yolov7-tiny.yaml`。  
  如需自定义 `sync_bn` 或精度参数，可在配置中修改；适配层会在此基础上应用上述分布式安全策略。

## 无图形界面（Headless）模式

许多 Ascend 服务器没有桌面环境。一旦 OpenCV 无法创建窗口：

- `video_relic_tracking.py` 与集成监控脚本会自动进入**无头模式**。
- 跳过交互式选择流程，脚本直接进入 `monitoring` 阶段。
- 提示信息以 `[提示]` 前缀输出到控制台。
- 帧渲染仍在内存中完成，但不会弹出窗口，可用于无人值守推理或通过 SSH 远程执行。

若需要交互式界面，请在支持 X11/Wayland 的环境中运行（如 `ssh -X`、VNC 或本地桌面）。

## 常见问题排查

| 现象 | 可能原因 | 解决方案 |
| --- | --- | --- |
| `Unsupported device target Ascend` | MindSpore 未安装 Ascend 支持 | 安装 Ascend 专用轮子，或改用 CPU/GPU 目标。 |
| `Distributed Communication has not been inited` | 配置启用了 SyncBatchNorm 但未初始化通信 | 适配层现已自动关闭 SyncBatchNorm；若需分布式，请手动调用 `mindspore.communication.init()`。 |
| `libge_runner.so: cannot open shared object file` | `LD_LIBRARY_PATH` 未包含 CANN runtime | 加载 CANN 环境脚本，或手动导出运行时路径。 |
| 无窗口显示 | 缺少显示服务器 | 进入无头模式属正常，可查看带 `[提示]` 的控制台日志。 |
| 姿态模型下载失败 | 无网络连接 | 预先放置 `models/pose_landmarker_full.task`（参考 `WebcamPoseDetection/download_model.py`）。 |

需要了解安全监控流水线的更多背景，可查阅 `docs/object_protection.md`；根目录 `README.md` 汇总了快捷启动命令，并指向本指南。
