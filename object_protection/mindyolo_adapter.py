#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MindYOLO inference adapter for the cultural relic protection pipeline.

This module loads MindYOLO checkpoints with MindSpore on Ascend/CANN devices
and exposes a simple `detect` method that returns YOLO-format detections
compatible with the rest of the project.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import requests

try:
    import mindspore as ms
    from mindspore import Tensor
except ImportError as exc:  # pragma: no cover - MindSpore is an external dependency
    raise ImportError(
        "MindSpore is required to use the MindYOLO adapter. Please install an Ascend-enabled "
        "MindSpore build before running the safety monitor."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MINDYOLO_ROOT = PROJECT_ROOT / "mindyolo"
if str(MINDYOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(MINDYOLO_ROOT))

from mindyolo.models import create_model
from mindyolo.utils.config import Config, load_config
from mindyolo.utils.metrics import non_max_suppression, scale_coords


DEFAULT_WEIGHT_URL = (
    "https://download.mindspore.cn/toolkits/mindyolo/yolov7/"
    "yolov7-tiny_300e_mAP375-d8972c94.ckpt"
)
DEFAULT_WEIGHT_PATH = PROJECT_ROOT / "yolov7-tiny.ckpt"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "mindyolo" / "configs" / "yolov7" / "yolov7-tiny.yaml"


def download_mindyolo_yolov7_tiny(
    destination: Path = DEFAULT_WEIGHT_PATH,
    *,
    url: str = DEFAULT_WEIGHT_URL,
    chunk_size: int = 1 << 15,
) -> Optional[Path]:
    """Download the official MindYOLO YOLOv7-tiny checkpoint if necessary."""

    destination = destination.expanduser().resolve()
    if destination.exists():
        print(f"MindYOLO权重已存在: {destination}")
        return destination

    print(f"正在下载MindYOLO YOLOv7-tiny权重: {url}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        tmp_path = destination.with_suffix(destination.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
        tmp_path.rename(destination)
        print(f"权重下载完成: {destination}")
        return destination
    except Exception as exc:  # pragma: no cover - network errors are environment-specific
        print(f"下载MindYOLO权重失败: {exc}")
        return None


class MindYOLODetector:
    """Thin wrapper around MindYOLO checkpoints for Ascend inference."""

    def __init__(
        self,
        config_path: Path,
        weight_path: Path,
        *,
        device_target: str = "Ascend",
        device_id: Optional[int] = None,
        ms_mode: int = ms.GRAPH_MODE,
        precision_mode: Optional[str] = None,
        amp_level: str = "O0",
        enable_graph_kernel: bool = False,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        conf_free: bool = False,
        exec_nms: bool = True,
        nms_time_limit: float = 60.0,
        warmup: bool = True,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.weight_path = Path(weight_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"MindYOLO配置文件不存在: {self.config_path}")
        if not self.weight_path.is_file():
            raise FileNotFoundError(f"MindYOLO权重文件不存在: {self.weight_path}")

        # Configure MindSpore runtime to leverage Ascend/CANN.
        device_id = int(os.getenv("DEVICE_ID", "0")) if device_id is None else int(device_id)
        try:
            ms.set_context(mode=ms_mode, device_target=device_target, device_id=device_id)
        except RuntimeError as exc:
            if "Unsupported device target" in str(exc):
                raise RuntimeError(
                    f"当前MindSpore构建不支持设备类型 {device_target!r}，"
                    "请确认已正确安装对应的硬件运行时，或使用 --device-target CPU/GPU 运行。"
                ) from exc
            raise
        ms.set_recursion_limit(2048)
        if enable_graph_kernel:
            ms.set_context(enable_graph_kernel=True)
        if ms_mode == ms.GRAPH_MODE:
            ms.set_context(jit_config={"jit_level": "O2"})

        cfg_dict, _, _ = load_config(str(self.config_path))
        self.cfg = Config(cfg_dict)
        self.img_size = int(self.cfg.img_size) if isinstance(self.cfg.img_size, int) else int(self.cfg.img_size[0])
        self.stride = int(self.cfg.network.stride[-1]) if hasattr(self.cfg.network, "stride") else 32
        self.class_names: Sequence[str] = list(self.cfg.data.names)
        self.num_classes = int(self.cfg.data.nc)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.conf_free = conf_free
        self.exec_nms = exec_nms
        self.nms_time_limit = nms_time_limit

        sync_bn = bool(getattr(self.cfg, "sync_bn", False))
        if sync_bn and not _is_distributed_initialized():
            print(
                "MindYOLO未启用SyncBatchNorm: 未检测到已初始化的分布式通信，自动回退到普通BatchNorm。"
            )
            sync_bn = False
            with contextlib.suppress(Exception):
                setattr(self.cfg, "sync_bn", False)
        precision_setting = precision_mode or getattr(self.cfg, "precision_mode", None)
        if precision_setting and device_target.lower() == "ascend":
            ms.device_context.ascend.op_precision.precision_mode(precision_setting)

        self.network = create_model(
            model_name=self.cfg.network.model_name,
            model_cfg=self.cfg.network,
            num_classes=self.num_classes,
            checkpoint_path=str(self.weight_path),
            sync_bn=sync_bn,
        )
        self.network.set_train(False)
        if (
            amp_level
            and amp_level.upper() != "O0"
            and hasattr(ms, "amp")
            and hasattr(ms.amp, "auto_mixed_precision")
        ):
            ms.amp.auto_mixed_precision(self.network, amp_level)

        if warmup:
            self._warmup(device_target)

    def _warmup(self, device_target: str) -> None:
        """Run a dummy forward pass to trigger graph compilation for faster inference."""

        dummy = np.zeros((1, 3, self.img_size, self.img_size), dtype=np.float32)
        tensor = Tensor(dummy, ms.float32)
        try:
            outputs = self.network(tensor)
            self._extract_prediction(outputs)
        except Exception as exc:  # pragma: no cover - depends on hardware availability
            print(f"MindYOLO预热失败 ({device_target}): {exc}")

    def detect(
        self,
        frame: np.ndarray,
        *,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        """Run detection on a BGR frame and return YOLO-style detections."""

        if frame is None or frame.size == 0:
            return []

        conf = float(self.conf_threshold if conf_threshold is None else conf_threshold)
        iou = float(self.iou_threshold if iou_threshold is None else iou_threshold)

        img = self._preprocess(frame)
        tensor = Tensor(img[None], ms.float32)

        outputs = self.network(tensor)
        preds = self._extract_prediction(outputs)

        preds_np = preds.asnumpy()
        results = non_max_suppression(
            preds_np,
            conf_thres=conf,
            iou_thres=iou,
            conf_free=self.conf_free,
            multi_label=True,
            time_limit=self.nms_time_limit,
            need_nms=self.exec_nms,
        )

        detections: List[Dict[str, object]] = []
        h_ori, w_ori = frame.shape[:2]
        for pred in results:
            if pred is None or len(pred) == 0:
                continue
            instances = np.copy(pred)
            scale_coords(img.shape[1:], instances[:, :4], (h_ori, w_ori))

            for det in instances:
                x1, y1, x2, y2, score, cls = det[:6]
                class_id = int(cls)
                if class_id < 0 or class_id >= len(self.class_names):
                    class_name = f"class_{class_id}"
                else:
                    class_name = self.class_names[class_id]

                x1_i = max(0, min(int(round(x1)), w_ori - 1))
                y1_i = max(0, min(int(round(y1)), h_ori - 1))
                x2_i = max(x1_i + 1, min(int(round(x2)), w_ori))
                y2_i = max(y1_i + 1, min(int(round(y2)), h_ori))

                detections.append(
                    {
                        "bbox": [x1_i, y1_i, x2_i, y2_i],
                        "confidence": float(score),
                        "class_id": class_id,
                        "class_name": class_name,
                        "area": float(max(0, (x2_i - x1_i) * (y2_i - y1_i))),
                    }
                )
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize/letterbox frame to match model input expectations."""

        h_ori, w_ori = frame.shape[:2]
        r = self.img_size / max(h_ori, w_ori)
        if r != 1:
            interpolation = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (int(w_ori * r), int(h_ori * r)), interpolation=interpolation)

        h, w = frame.shape[:2]
        if h < self.img_size or w < self.img_size:
            new_h = math.ceil(h / self.stride) * self.stride
            new_w = math.ceil(w / self.stride) * self.stride
            dh, dw = (new_h - h) / 2, (new_w - w) / 2
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img = frame[:, :, ::-1].transpose(2, 0, 1) / 255.0
        img = np.ascontiguousarray(img, dtype=np.float32)
        return img

    def _extract_prediction(self, outputs):
        """Normalize MindYOLO network outputs to the final prediction tensor."""

        preds = outputs
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if isinstance(preds, (list, tuple)):
            preds = preds[-1]
        return preds


def create_default_detector(
    *,
    weight_path: Optional[Path] = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.65,
    device_target: str = "Ascend",
    device_id: Optional[int] = None,
) -> Optional[MindYOLODetector]:
    """Convenience helper used by the CLI entry point."""

    weight_path = weight_path or download_mindyolo_yolov7_tiny()
    if weight_path is None:
        return None
    return MindYOLODetector(
        config_path=config_path,
        weight_path=weight_path,
        device_target=device_target,
        device_id=device_id,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )


def _is_distributed_initialized() -> bool:
    """Best-effort check whether MindSpore distributed communication is ready."""

    with contextlib.suppress(ImportError):
        from mindspore.communication.management import get_group_size  # type: ignore

        try:
            return get_group_size() > 1
        except RuntimeError:
            return False
        except ValueError:
            return False
    return False
