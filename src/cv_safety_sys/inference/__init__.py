"""MindSpore/CANN 推理后端统一入口。"""

from .mindspore_runtime import (
    BackendMetadata,
    BaseYoloBackend,
    MindSporeYoloBackend,
    create_backend,
)

__all__ = [
    "BackendMetadata",
    "BaseYoloBackend",
    "MindSporeYoloBackend",
    "create_backend",
]
