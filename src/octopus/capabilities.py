from __future__ import annotations

from typing import Any

from octopus._types import ShardingMode
from octopus.exceptions import ShardingError

_BACKEND_CAPABILITIES: dict[str, dict[str, Any]] = {
    "pytorch": {
        "replica_workers": True,
        "shared_model_multi_gpu": True,
        "shared_model_sharding_strategies": ("pp", "tp"),
        "notes": (
            "Pipeline parallel and tensor parallel shared-model runtimes are supported."
        ),
    },
    "onnx": {
        "replica_workers": True,
        "shared_model_multi_gpu": True,
        "shared_model_sharding_strategies": ("pp",),
        "notes": (
            "Replica-worker mode supported. Shared-model runtime supports "
            "pipeline parallel (pp) only."
        ),
    },
    "onnx_quantsim": {
        "replica_workers": True,
        "shared_model_multi_gpu": True,
        "shared_model_sharding_strategies": ("pp",),
        "notes": (
            "Replica-worker mode supported. Shared-model runtime supports "
            "pipeline parallel (pp) only."
        ),
    },
    "quantsim": {
        "replica_workers": True,
        "shared_model_multi_gpu": False,
        "shared_model_sharding_strategies": (),
        "notes": "Replica-worker mode supported. Shared-model sharding not implemented.",
    },
}


def get_backend_capabilities(model_type_name: str) -> dict[str, Any]:
    """Return backend execution capabilities for a model adapter name."""
    normalized = str(model_type_name).lower()
    caps = _BACKEND_CAPABILITIES.get(normalized)
    if caps is None:
        return {
            "model_type_name": normalized,
            "replica_workers": True,
            "shared_model_multi_gpu": False,
            "shared_model_sharding_strategies": (),
            "notes": "Unknown backend. Replica-workers only unless implemented.",
        }

    return {
        "model_type_name": normalized,
        "replica_workers": bool(caps["replica_workers"]),
        "shared_model_multi_gpu": bool(caps["shared_model_multi_gpu"]),
        "shared_model_sharding_strategies": tuple(
            caps["shared_model_sharding_strategies"]
        ),
        "notes": str(caps["notes"]),
    }


def validate_sharding_configuration(
    model_type_name: str,
    sharding_strategy: ShardingMode,
) -> None:
    """Fail fast when backend/sharding combination is not supported."""
    if sharding_strategy == "none":
        return

    caps = get_backend_capabilities(model_type_name)
    if not caps["shared_model_multi_gpu"]:
        raise ShardingError(
            "Shared-model multi-GPU runtime not supported for this backend. "
            "Use sharding_strategy='none' for replica workers."
        )
    if sharding_strategy not in caps["shared_model_sharding_strategies"]:
        supported = ", ".join(
            [f"'{mode}'" for mode in caps["shared_model_sharding_strategies"]]
        ) or "'none'"
        raise ShardingError(
            f"sharding_strategy={sharding_strategy!r} not supported for backend "
            f"{model_type_name!r}. Supported shared-model strategies: {supported}."
        )
