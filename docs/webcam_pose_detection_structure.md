# WebcamPoseDetection 目录结构

```
WebcamPoseDetection/
├── download_model.py
├── pose33_realtime_optimized.py
├── webcam_pose_minimal.py
├── webcam_pose_simple.py
├── test_setup.py
└── __init__.py（可选）
```

## 代码导览

- **`webcam_pose_minimal.py`**  
  将摄像头采集、MediaPipe 推理与关键点绘制串联起来的轻量脚本，适合做快速自检。
- **`webcam_pose_simple.py`**  
  通过 `SimpleWebcamPoseDetector` 封装运行时，提供 FPS 统计和稳定的错误处理，方便在其他模块中直接复用。
- **`pose33_realtime_optimized.py`**  
  利用队列将采集与推理拆到不同线程，配合分辨率缩放辅助函数，并暴露适合部署场景的命令行开关。
- **`download_model.py`**  
  将 `pose_landmarker_full.task` 下载至 `models/` 目录。缺少模型时先执行一次即可。
- **`test_setup.py`**  
  可选的起飞前检查：确认摄像头可用并打印依赖版本。

## 与安全监控的衔接

- 集成安全监控会导入 `download_model.py`，用于定位或拉取姿态模型。
- 任意姿态脚本输出的关键点都能直接馈入 `IntegratedSafetyMonitor`，无需额外适配层。
- 全部依赖统一写在根目录 `requirements.txt` 中，最终发布时只需维护这一份环境定义。
