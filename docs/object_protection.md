# 杯子跟踪与安全监控

本模块结合 YOLOv7-tiny 检测、基于质心的跟踪以及 MediaPipe 姿态关键点，用于守护指定杯子并将网球拍标记为危险物体。整套流程围绕最终演示配置进行优化，已去除早期实验阶段的内容。

## 目录速览

```
object_protection/
├── video_relic_tracking.py       # YOLOv7-tiny 杯子检测与交互式跟踪
├── integrated_safety_monitor.py  # 杯子围栏监控 + 姿态融合 + 网球拍告警
├── general.py                    # 精简版 YOLOv7 工具函数
├── yolov7-tiny.pt                # 推理权重（首次运行自动下载）
└── yolov7/                       # 检测器所需的第三方目录
```

## 最终功能

- **围绕杯子的检测流程**：`video_relic_tracking.py` 将可选目标限制为 `cup` 类别。选中的实例会获得持久 ID、置信度提示，并扩展安全围栏用于入侵监控。
- **网球拍危险识别**：`integrated_safety_monitor.py` 仅保留 `{cup, person, tennis racket}` 的 YOLOv7 检测结果。一旦发现网球拍，系统会高亮显示并与附近人员建立关联，生成携带告警。
- **姿态驱动的风险分析**：将 MediaPipe 姿态关键点投影到视频帧上，用以判断关节是否越过杯子的安全围栏。
- **统一的交互体验**：两套脚本共享鼠标/键盘操作和可视化规范，确保整个子系统的交互方式保持一致。

## 运行方式

```bash
# 交互式杯子跟踪（默认摄像头）
python object_protection/video_relic_tracking.py --source 0

# 集成监控（杯子 + 人体 + 网球拍）
python object_protection/integrated_safety_monitor.py --source 0
```

可选参数：
- `--source`：摄像头编号或视频文件路径。
- `--conf`：置信度阈值（默认 `0.1`）。

集成监控脚本会自动加载通过 `WebcamPoseDetection/download_model.py` 下载的 MediaPipe 姿态模型，并复用共享的 `SimpleTracker`。

## 实现要点

- 请拉取上游 YOLOv7 仓库，以便检测器导入相应工具模块：
  ```bash
  git clone --depth 1 https://github.com/WongKinYiu/yolov7.git object_protection/yolov7
  ```
- YOLOv7-tiny 权重来自官方发布，首次运行后会缓存在本地；若运行环境不能联网，可手动放置：
  ```bash
  curl -L -o object_protection/yolov7-tiny.pt \
    https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
  ```
- 基于博物馆数据集微调的权重可手动下载：
  - https://drive.google.com/file/d/1w83ZJ-HFex8NhGZyN_RGNQmmzGNDFXZ4/view?usp=share_link
- 危险物体过滤依赖共享常量 `DANGEROUS_CLASSES = {'tennis racket'}`。若需调整危险策略，请在各模块中同步更新。
- 在参与跟踪前，所有检测框都会映射回原始帧尺寸，确保姿态关键点与安全围栏精确对齐。
