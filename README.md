# CV安全系统

这是一个面向展陈安全场景的视觉工具库，可在不同硬件环境下完成实时检测、姿态估计与安全策略联动。仓库中提供两套可以单独运行、也可以彼此协作的子系统：

- **WebcamPoseDetection**：基于 MediaPipe 的 33 关键点人体姿态估计，包含轻量版、易于调试的类封装版本，以及针对实时性优化的线程化版本。
- **object_protection**：依托 MindYOLO（MindSpore）完成检测，并提供交互式目标跟踪与集成安全监控，将姿态关键点与检测结果关联。默认配置特别针对 Ascend/CANN 平台（如 Orange Pi 加速模组）进行优化。

当前演示方案将**纸杯视作待保护展品**，并把**网球拍视为危险物体**。界面文案、告警逻辑和筛选策略都围绕这一设定设计。

## 核心亮点

- MindYOLO YOLOv7-tiny 借助 MindSpore 在 Ascend/CANN 设备上实现实时推理，并结合质心跟踪支持交互式选择待保护的杯子。
- 针对已选杯子自动生成安全围栏，利用 MediaPipe 姿态关键点判断是否有人越界。
- 危险识别流程会高亮网球拍，并与最近的被跟踪人员建立关联，从而发出携带告警。
- 运行环境已在 Python 3.9 / Ubuntu 22.04、MindSpore 2.3 与 CANN 8.x 上完成验证，同时提供 CPU/GPU 回退方案。

## 快速入门

```bash
# 安装 Ascend 版 MindSpore，不同板卡请参考官方安装指引
pip install mindspore-ascend==2.3.1

# 安装通用依赖（OpenCV、MediaPipe 等）
pip install -r requirements.txt

# 下载 MediaPipe 姿态模型
python WebcamPoseDetection/download_model.py

# 启动杯子跟踪（默认摄像头）
python object_protection/video_relic_tracking.py --source 0

# 启动集成安全监控（杯子 + 网球拍）
python object_protection/integrated_safety_monitor.py --source 0
```

所有脚本都可通过 `--source` 指定摄像头编号或视频文件。  
若需覆盖默认 Ascend 运行时或加载自定义 MindYOLO 权重，可使用 `--device-target`、`--device-id` 或 `--weight`。  
MindYOLO 首次运行会自动拉取官方 MindSpore 检查点 `yolov7-tiny.ckpt`；若文件已存在则直接使用。

```bash
curl -L -o yolov7-tiny.ckpt \
  https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-d8972c94.ckpt
```

如需基于我们在博物馆数据集上训练的模型继续实验，可从 Google Drive 下载权重并与默认文件放在一起：

- https://drive.google.com/drive/folders/1xjeqz_GMzl5gKie2LgOHv-iQns_v3mU4?usp=share_link

由于测试条件有限，我们尚未在真实展品上做实地验证，目前示例仅使用官方权重，并以纸杯替代展品、以网球拍替代危险物体。如需在真实场景中评估，可下载上述微调权重、调整相关代码后再进行测试。  

## MindYOLO / MindSpore / CANN 集成说明

- 默认面向 Ascend NPU；满足依赖后也可通过 `--device-target CPU` 或 GPU 回退运行。
- MindYOLO 配置中的 SyncBatchNorm 会在未初始化 MindSpore 分布式通信时自动关闭，避免单卡运行崩溃。
- 当 OpenCV 因缺少图形界面无法弹出窗口（无头服务器场景）时，安全监控会切换为控制台模式，并以前缀 `[提示]` 输出交互信息。

更多环境部署、故障排查与离线使用技巧，请参阅 [docs/mindyolo_mindspore_cann.md](docs/mindyolo_mindspore_cann.md)。

## 仓库结构

```
cv_safety_sys/
├── WebcamPoseDetection/       # 姿态估计子系统
├── object_protection/         # 杯子跟踪与安全监控
└── docs/                      # 详细说明文档（Ascend 配置详见 mindyolo_mindspore_cann.md）
```

`docs/` 目录提供各子系统的深入说明与实现细节，可与本 README 一起作为最终配置的参考资料。
