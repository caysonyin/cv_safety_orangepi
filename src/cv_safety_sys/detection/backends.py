from __future__ import annotations

"""Inference backends for YOLO-based detection on different runtimes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import cv2
import numpy as np


BackendName = Literal["torch", "mindspore"]


@dataclass
class BackendMetadata:
    """Runtime metadata returned alongside the backend instance."""

    name: str
    device: str


class BaseYoloBackend:
    """Base class that implements common pre/post utilities for YOLO backends."""

    input_size: Tuple[int, int] = (640, 640)

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalise a frame for YOLO style models."""

        resized = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalised = rgb.astype(np.float32) / 255.0
        nchw = np.transpose(normalised, (2, 0, 1))
        return np.expand_dims(nchw, axis=0)

    def forward(self, frame: np.ndarray) -> np.ndarray:
        """Run inference and return the raw predictions (N, 85)."""

        input_tensor = self.preprocess(frame)
        outputs = self._forward_impl(input_tensor)
        if outputs is None:
            raise RuntimeError("backend failed to return inference outputs")
        if outputs.ndim == 3 and outputs.shape[0] == 1:
            outputs = outputs[0]
        return outputs

    def close(self) -> None:
        """Release backend specific resources (optional)."""

    # ------------------------------------------------------------------
    # Backend specific implementation
    # ------------------------------------------------------------------
    def _forward_impl(self, input_tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TorchBackend(BaseYoloBackend):
    """Inference backend that relies on PyTorch."""

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        super().__init__(model_path)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "PyTorch 未安装，请在使用 torch 后端前先安装 torch。"
            ) from exc

        load_kwargs = {}
        signature = getattr(torch.load, "__signature__", None)
        if signature is None:
            try:
                from inspect import signature as _sig

                sig = _sig(torch.load)
                if "weights_only" in sig.parameters:
                    load_kwargs["weights_only"] = False
            except (TypeError, ValueError):
                pass

        map_location = torch.device(device)
        checkpoint = torch.load(self.model_path, map_location=map_location, **load_kwargs)
        model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.torch = torch
        self.model = model.to(map_location).float().eval()
        self.device = str(map_location)

    def _forward_impl(self, input_tensor: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(input_tensor).to(self.device)
        with self.torch.no_grad():  # type: ignore[attr-defined]
            outputs = self.model(tensor)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        return outputs.detach().cpu().numpy()  # type: ignore[return-value]


class MindSporeBackend(BaseYoloBackend):
    """Inference backend powered by MindSpore (GraphCell/Mindir)."""

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
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "MindSpore 未安装，请在使用 mindspore 后端前先安装 mindspore。"
            ) from exc

        context.set_context(mode=context.GRAPH_MODE, device_target=device_target, device_id=device_id)

        suffix = self.model_path.suffix.lower()
        if suffix != ".mindir":
            raise ValueError(
                f"MindSpore 后端仅支持 MindIR 模型文件 (.mindir)，当前文件: {self.model_path}"
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
    backend: BackendName,
    model_path: Path,
    *,
    device: str = "cpu",
    device_target: str = "Ascend",
    device_id: int = 0,
) -> Tuple[BaseYoloBackend, BackendMetadata]:
    """Instantiate a backend and return it with runtime metadata."""

    backend = backend.lower()
    if backend == "torch":
        torch_backend = TorchBackend(model_path, device=device)
        return torch_backend, BackendMetadata(name="torch", device=torch_backend.device)
    if backend == "mindspore":
        ms_backend = MindSporeBackend(model_path, device_target=device_target, device_id=device_id)
        device_name = f"{device_target}:{device_id}" if device_target.lower() != "cpu" else "CPU"
        return ms_backend, BackendMetadata(name="mindspore", device=device_name)

    raise ValueError(f"未支持的后端: {backend}")
