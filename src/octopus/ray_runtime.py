from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import ray

from octopus._logging import get_logger

_log = get_logger()

_RAY_TMPDIR_ENV_VARS = ("OCTOPUS_RAY_TMPDIR", "RAY_TMPDIR")
_DEFAULT_RAY_TEMP_DIRS = (
    "/local/mnt/o-ray",
    "/dev/shm/octopus-ray",
    "/mnt/o-ray",
    "/var/tmp/ray",
    "/tmp/r",
)
_SOCKET_SUFFIX = "ray/session_2099-12-31_23-59-59_123456_123456/sockets/plasma_store"
_MAX_UNIX_SOCKET_PATH_BYTES = 107


def _socket_path_is_safe(path: Path) -> bool:
    socket_path = path / _SOCKET_SUFFIX
    return len(os.fspath(socket_path).encode("utf-8")) <= _MAX_UNIX_SOCKET_PATH_BYTES


def _path_free_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return int(stats.f_bavail * stats.f_frsize)


def _prepare_dir(raw_path: str) -> Optional[Path]:
    path = Path(raw_path).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    if not _socket_path_is_safe(path):
        _log.warning("Skipping Ray temp dir %s: UNIX socket path would be too long.", path)
        return None
    try:
        _path_free_bytes(path)
    except OSError:
        return None
    return path


def resolve_ray_temp_dir() -> Optional[str]:
    """Pick short writable Ray temp dir with most free space.

    Priority:
        1. Explicit env vars: OCTOPUS_RAY_TMPDIR, then RAY_TMPDIR
        2. Built-in short fallback paths, ranked by free space
    """
    for env_var in _RAY_TMPDIR_ENV_VARS:
        raw = os.environ.get(env_var)
        if not raw:
            continue
        path = _prepare_dir(raw)
        if path is not None:
            return os.fspath(path)

    best_path: Optional[Path] = None
    best_free = -1

    for raw in _DEFAULT_RAY_TEMP_DIRS:
        path = _prepare_dir(raw)
        if path is None:
            continue
        free_bytes = _path_free_bytes(path)
        if free_bytes > best_free:
            best_path = path
            best_free = free_bytes

    if best_path is None:
        return None
    return os.fspath(best_path)


def build_ray_init_kwargs(
    *,
    address: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build Ray init kwargs with safe local temp dir when we start local Ray."""
    init_kwargs = dict(kwargs)
    if address is not None:
        init_kwargs["address"] = address
        return init_kwargs

    temp_dir = resolve_ray_temp_dir()
    if temp_dir is not None:
        init_kwargs["_temp_dir"] = temp_dir
    return init_kwargs


def init_ray(*, address: Optional[str] = None, **kwargs: Any) -> Any:
    """Call ray.init with stable local temp dir defaults."""
    return ray.init(**build_ray_init_kwargs(address=address, **kwargs))
