# 摄像头姿态检测

姿态子系统提供安全监控所需的 MediaPipe 关键点。我们保留了三种结构不同、性能侧重点各异的运行入口，均围绕最终项目配置进行了统一。

## 模块速览

| 文件 | 功能说明 |
| --- | --- |
| `webcam_pose_minimal.py` | 极简入口，用于快速验证 MediaPipe 运行环境与摄像头连通性。 |
| `webcam_pose_simple.py` | 基于类的实现，附带 FPS 统计，方便在其他模块中复用。 |
| `pose33_realtime_optimized.py` | 使用生产者/消费者线程队列，为低延迟场景优化的实时管线。 |
| `download_model.py` | 将 `pose_landmarker_full.task` 下载到 `models/` 目录。 |
| `test_setup.py` | 可选的健康检查，确认摄像头可用并输出依赖版本信息。 |

## 使用提示

- 执行 `pip install -r requirements.txt` 安装依赖，并通过 `python WebcamPoseDetection/download_model.py` 下载模型（一次即可）。
- 所有脚本都支持 `--source` 参数，可按需切换摄像头编号或播放预录视频，无需改动代码。
- 当杯子安全流程对姿态更新延迟较敏感时，推荐使用优化版管线 `pose33_realtime_optimized.py`。

## 集成要点

- `IntegratedSafetyMonitor` 直接消费姿态关键点，用于判定杯子围栏是否被入侵。
- 帧处理采用 MediaPipe 的 VIDEO 模式，既降低抖动，又能让关键点与 YOLO 检测对齐。
- 所有脚本现在统一输出英文日志，便于与仓库整体 UI 保持一致。
