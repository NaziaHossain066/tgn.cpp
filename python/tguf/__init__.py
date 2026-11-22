import torch as _torch

from ._build_info import TORCH_VERSION as _BUILT_TORCH_VERSION


def _major_minor(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("+")[0].split(".")[:2])


if _major_minor(_torch.__version__) != _major_minor(_BUILT_TORCH_VERSION):
    raise ImportError(
        f"tguf was built against torch {_BUILT_TORCH_VERSION}, but torch "
        f"{_torch.__version__} is installed. Reinstall tguf to rebuild it."
    )

from ._tguf_py import (
    Batch,
    IndexRange,
    LabelBatch,
    LabelEvent,
    LabelEventTable,
    NegStrategy,
    TGStore,
    TGUFBuilder,
    TGUFSchema,
)

__all__ = [
    "Batch",
    "IndexRange",
    "LabelBatch",
    "LabelEvent",
    "LabelEventTable",
    "NegStrategy",
    "TGStore",
    "TGUFBuilder",
    "TGUFSchema",
]
