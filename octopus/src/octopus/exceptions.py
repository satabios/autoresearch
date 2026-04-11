class OctopusError(Exception):
    """Base for all octopus errors."""


class InsufficientVRAMError(OctopusError):
    """Model does not fit on any single GPU, and sharding is disabled."""


class ProfilingError(OctopusError):
    """Dry-run VRAM profiling failed."""


class WorkerOOMError(OctopusError):
    """A worker hit OOM during inference."""


class WorkerCrashedError(OctopusError):
    """A Ray actor died unexpectedly."""


class ShardingError(OctopusError):
    """Model cannot be sharded with the requested strategy."""


class NoGPUsFoundError(OctopusError):
    """No CUDA GPUs discovered."""
