"""MindSpore/CANN YOLO 推理后端。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class BackendMetadata:
    """运行时元信息。"""

    name: str
    device: str


class BaseYoloBackend:
    """YOLO 推理后端基类，封装 MindSpore 共享的预处理逻辑。"""

    input_size: Tuple[int, int] = (640, 640)

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)

    # ------------------------------------------------------------------
    # 通用工具
    # ------------------------------------------------------------------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize + 归一化 + NHWC->NCHW 转换。"""

        resized = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalised = rgb.astype(np.float32) / 255.0
        nchw = np.transpose(normalised, (2, 0, 1))
        return np.expand_dims(nchw, axis=0)

    def forward(self, frame: np.ndarray) -> np.ndarray:
        """执行一次前向推理并返回原始输出。"""

        input_tensor = self.preprocess(frame)
        outputs = self._forward_impl(input_tensor)
        if outputs is None:
            raise RuntimeError("推理后端未返回有效输出")
        if outputs.ndim == 3 and outputs.shape[0] == 1:
            outputs = outputs[0]
        return outputs

    def close(self) -> None:
        """释放资源（可选实现）。"""

    # ------------------------------------------------------------------
    # 子类实现
    # ------------------------------------------------------------------
    def _forward_impl(self, input_tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MindSporeYoloBackend(BaseYoloBackend):
    """使用 MindSpore GraphCell/Mindir 的 Ascend 适配实现。"""

    def __init__(
        self,
        model_path: Path,
        *,
        device_target: str = "Ascend",
        device_id: int = 0,
    ) -> None:
        super().__init__(model_path)
        try:
            import mindspore as ms
            from mindspore import context
            from mindspore.train.serialization import load_mindir
        except ImportError as exc:  # pragma: no cover - 运行环境相关
            raise RuntimeError("MindSpore 未安装，请先根据昇腾环境部署指南安装 mindspore-ascend。") from exc

        context.set_context(
            mode=context.GRAPH_MODE,
            device_target=device_target,
            device_id=device_id,
        )

        suffix = self.model_path.suffix.lower()
        if suffix != ".mindir":
            raise ValueError(
                f"MindSpore 后端仅支持 MindIR 模型文件 (.mindir)，当前: {self.model_path}"
            )

        self.ms = ms
        self.graph = load_mindir(str(self.model_path))
        if hasattr(self.graph, "set_train"):
            self.graph.set_train(False)
        self.device_target = device_target
        self.device_id = device_id

    def _forward_impl(self, input_tensor: np.ndarray) -> np.ndarray:
        tensor = self.ms.Tensor(input_tensor, self.ms.float32)
        outputs = self.graph(tensor)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        if hasattr(outputs, "asnumpy"):
            return outputs.asnumpy()
        raise RuntimeError("MindSpore 返回了不受支持的输出类型")


def create_backend(
    model_path: Path,
    *,
    device_target: str = "Ascend",
    device_id: int = 0,
) -> Tuple[BaseYoloBackend, BackendMetadata]:
    """构建 MindSpore YOLO 后端。"""

    backend = MindSporeYoloBackend(
        model_path,
        device_target=device_target,
        device_id=device_id,
    )
    device_name = f"{device_target}:{device_id}" if device_target.lower() != "cpu" else "CPU"
    return backend, BackendMetadata(name="mindspore", device=device_name)
