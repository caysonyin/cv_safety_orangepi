# MindSpore / CANN 适配说明

## 1. 运行时依赖

| 组件 | 版本建议 | 说明 |
| --- | --- | --- |
| CANN Runtime | >= 8.0.RC3 | 提供 `aclruntime`、驱动与烧写工具，请严格按照昇腾官方指南安装。 |
| MindSpore | 2.2.x (Ascend) | 使用 `pip install mindspore-ascend==2.2.10` 或对应芯片型号的 whl 包。 |
| Python | 3.10 | 与仓库的虚拟环境保持一致，便于直接 `pip install -r requirements.txt`。 |

> **提示**：`aclruntime` 会随着 CANN 安装自动部署，无需额外 pip 安装；确保 `LD_LIBRARY_PATH` 包含 CANN 的 `lib` 目录。

## 2. 模型准备

1. **MindIR 权重**：仓库默认从 MindSpore 官方仓库下载 `yolov7_tiny.mindir`，路径位于 `models/yolov7-tiny.mindir`；如需用于文物检测的迁移学习模型，可下载我们在开源数据集上训练的 [yolov7tiny.pt](https://drive.google.com/file/d/1w83ZJ-HFex8NhGZyN_RGNQmmzGNDFXZ4/view?usp=share_link) 并替换该文件。可能需要进行模型格式转换。
2. **自定义模型**：若需替换模型，可在原始 PyTorch 权重上使用 `Ascend Model Zoo` 提供的导出脚本生成 MindIR，然后覆盖 `models/` 中的文件。
3. **校验**：在设备上执行 `python -m cv_safety_sys.detection.yolov7_tracker --source 0 --device-target Ascend` 以验证模型加载是否成功。

## 3. 目录与代码重构

- 新增 `src/cv_safety_sys/inference/` 目录，集中管理 MindSpore YOLO 后端。
- 所有入口统一使用 MindSpore Runtime。
- 命令行参数仅保留 `--device-target` 与 `--device-id`，用于在多 NPU 场景下切换设备。
- `requirements.txt` 仅 Python 依赖；MindSpore 与 CANN 需按平台文档安装。

## 4. 运行指引

```bash
# 激活 Ascend 运行环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装 Python 依赖
pip install -r requirements.txt

# 启动桌面客户端
python run.py --source 0 --device-target Ascend --device-id 0
```

若使用远程摄像头或视频文件，可将 `--source` 替换为 RTSP 地址或本地路径。所有 UI/CLI 工具共享同一套参数。

## 5. 常见问题

| 问题 | 可能原因 | 解决方式 |
| --- | --- | --- |
| MindSpore 提示找不到 Ascend 设备 | CANN 环境变量未生效 | 运行 `set_env.sh` 或检查 `npu-smi info` 是否能看到设备。 |
| 模型加载报 MindIR 版本不兼容 | MindSpore 版本与模型导出版本不一致 | 使用同一 MindSpore 版本重新导出 MindIR 或升级/降级 runtime。 |
| 推理速度慢 | 运行在 CPU 模式 | 检查 `--device-target` 是否为 `Ascend`，或确认 CANN 驱动是否加载成功。 |

更多调优细节可参考昇腾社区文档或 MindSpore 官方指南。
