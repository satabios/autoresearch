"""Octopus: GPU worker parallelization for model inference."""

from octopus._types import GPUInfo, PoolPlan, VRAMProfile, WorkerAllocation
from octopus._version import __version__
from octopus.core import Octopus, get_backend_capabilities
from octopus.exceptions import (
    InsufficientVRAMError,
    NoGPUsFoundError,
    OctopusError,
    ProfilingError,
    ShardingError,
    WorkerCrashedError,
    WorkerOOMError,
)

__all__ = [
    "Octopus",
    "GPUInfo",
    "VRAMProfile",
    "PoolPlan",
    "WorkerAllocation",
    "OctopusError",
    "InsufficientVRAMError",
    "ProfilingError",
    "WorkerOOMError",
    "WorkerCrashedError",
    "ShardingError",
    "NoGPUsFoundError",
    "get_backend_capabilities",
    "__version__",
]
