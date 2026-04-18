from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Callable

import ray  # type: ignore[import-untyped]

_log = logging.getLogger(__name__)

_WORKER_PROVIDERS = [
    ("CUDAExecutionProvider", {"device_id": 0}),
    "CPUExecutionProvider",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _find_encodings_file(tmpdir: str, prefix: str) -> str:
    for suffix in (".encodings", ".encodings.json"):
        path = os.path.join(tmpdir, f"{prefix}{suffix}")
        if os.path.exists(path):
            return path
    raise RuntimeError(
        f"No encodings file found in {tmpdir!r} with prefix {prefix!r}. "
        "AIMET sim.export() may have failed or used an unexpected naming convention."
    )


def _load_encodings_to_sim(sim: Any, enc_path: str, strict: bool = False) -> None:
    try:
        from aimet_onnx.quantsim import load_encodings_to_sim  # type: ignore[import-untyped]
    except ImportError:
        from aimet_onnx.utils import load_encodings_to_sim  # type: ignore[import-untyped]
    load_encodings_to_sim(sim, enc_path, strict=strict)


def _split_callback_args(callback_args: Any, K: int) -> list:
    """Split calibration data into K shards for parallel workers.

    Handles list, and torch DataLoader with accessible dataset.
    Falls back to single worker on unsplittable iterables.
    """
    if K <= 1:
        return [callback_args]

    if isinstance(callback_args, list):
        shards = [callback_args[i::K] for i in range(K)]
        return [s for s in shards if s]

    try:
        import torch  # type: ignore[import-untyped]
        from torch.utils.data import DataLoader, Subset  # type: ignore[import-untyped]

        if isinstance(callback_args, DataLoader):
            dataset = callback_args.dataset
            n = len(dataset)
            k = min(K, n)
            chunk_size = max(1, (n + k - 1) // k)
            shards = []
            for i in range(k):
                start = i * chunk_size
                end = min(start + chunk_size, n)
                if start >= end:
                    break
                subset = Subset(dataset, list(range(start, end)))
                shards.append(
                    DataLoader(
                        subset,
                        batch_size=callback_args.batch_size or 1,
                        num_workers=0,  # avoid multiprocessing inside Ray workers
                        collate_fn=callback_args.collate_fn,
                    )
                )
            return shards
    except (ImportError, AttributeError, TypeError) as e:
        _log.debug("Cannot split DataLoader (%s) — using single worker", e)

    # Cannot split: use single worker (correct for min/max observers, just slower)
    _log.warning(
        "callback_args type %s cannot be split; falling back to single calibration worker.",
        type(callback_args).__name__,
    )
    return [callback_args]


# ---------------------------------------------------------------------------
# Case B: Replicated parallel calibration
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
class _CalibrationWorker:
    """Ray actor: reconstruct fresh QuantSim, calibrate on a data shard, return encodings JSON."""

    def __init__(self, model_proto_bytes: bytes, quant_init_kwargs: dict) -> None:
        self._model_proto_bytes = model_proto_bytes
        self._quant_init_kwargs = quant_init_kwargs

    def calibrate(self, forward_pass_callback: Callable, data_shard: Any) -> bytes:
        """Run compute_encodings on data_shard. Returns .encodings file as bytes."""
        import onnx  # type: ignore[import-untyped]

        from octopus_aimet._patch import _Original as _OriginalQSim

        model_proto = onnx.ModelProto()
        model_proto.ParseFromString(self._model_proto_bytes)

        # Override providers: device 0 within this actor's CUDA_VISIBLE_DEVICES scope
        kwargs = dict(self._quant_init_kwargs)
        kwargs["providers"] = _WORKER_PROVIDERS

        sim = _OriginalQSim(model_proto, **kwargs)
        sim.compute_encodings(forward_pass_callback, data_shard)

        with tempfile.TemporaryDirectory() as tmpdir:
            sim.export(path=tmpdir, filename_prefix="_cal", export_model=False)
            enc_path = _find_encodings_file(tmpdir, "_cal")
            with open(enc_path, "rb") as f:
                return f.read()


def parallel_compute_encodings(
    sim: Any,
    forward_pass_callback: Callable,
    callback_args: Any,
) -> None:
    """Distribute compute_encodings across K GPU workers; merge per-tensor min/max.

    Algorithm:
        1. Discover GPUs → K workers (one per GPU).
        2. Split callback_args into K shards.
        3. Spawn K _CalibrationWorker Ray actors (num_gpus=1 each).
        4. Each worker: fresh QuantSim → compute_encodings(callback, shard) → export .encodings.
        5. Merge K encoding dicts via per-tensor min/max (exact for QuantScheme.post_training_tf).
        6. Load merged encodings into the driver sim via load_encodings_to_sim.

    Limitation: histogram-based calibration (percentile, KL) requires global statistics.
    If sim.quant_scheme is not post_training_tf, this merge is approximate (wider range).
    """
    from octopus.discovery import discover_gpus
    from octopus_aimet._merge_encodings import merge_encoding_dicts

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    gpus = discover_gpus()
    K = len(gpus)
    shards = _split_callback_args(callback_args, K)
    K = len(shards)  # may be less than GPU count if data too small

    _log.info("Parallel calibration: %d workers, %d data shards", K, K)

    workers = [
        _CalibrationWorker.options(num_gpus=1).remote(  # type: ignore[attr-defined]
            model_proto_bytes=sim._octopus_model_proto_bytes,
            quant_init_kwargs=sim._octopus_init_kwargs,
        )
        for _ in range(K)
    ]

    try:
        futures = [
            w.calibrate.remote(forward_pass_callback, shard)
            for w, shard in zip(workers, shards)
        ]
        encoding_bytes_list: list[bytes] = ray.get(futures)
    finally:
        for w in workers:
            try:
                ray.kill(w)
            except Exception as e:
                _log.debug("Worker kill failed: %s", e)

    encoding_dicts = [json.loads(b) for b in encoding_bytes_list]
    merged = merge_encoding_dicts(encoding_dicts)

    with tempfile.TemporaryDirectory() as tmpdir:
        merged_path = os.path.join(tmpdir, "merged.encodings")
        with open(merged_path, "w") as f:
            json.dump(merged, f)
        _load_encodings_to_sim(sim, merged_path, strict=False)

    _log.info("Parallel calibration complete. Merged encodings loaded into sim.")


# ---------------------------------------------------------------------------
# Case A: Sharded (pipeline-parallel) calibration
# ---------------------------------------------------------------------------


class _OutputCapturingSession:
    """Wraps an ORT InferenceSession to capture specified output tensors during run().

    Used to collect stage boundary tensors during compute_encodings without
    requiring a second forward pass.
    """

    def __init__(
        self,
        real_session: Any,
        boundary_output_names: list[str],
        buffer: list[dict],
    ) -> None:
        self._sess = real_session
        self._boundary_names: frozenset[str] = frozenset(boundary_output_names)
        self._buffer = buffer

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict,
        *args: Any,
        **kwargs: Any,
    ) -> list:
        # Build extended name list: original request + missing boundary names
        if output_names is None:
            requested = [o.name for o in self._sess.get_outputs()]
        else:
            requested = list(output_names)

        req_set = set(requested)
        extra = [n for n in self._boundary_names if n not in req_set]
        extended = requested + extra

        results = self._sess.run(extended, input_feed, *args, **kwargs)
        result_dict = dict(zip(extended, results))

        # Capture boundary tensors for next stage
        boundary_feed = {n: result_dict[n] for n in self._boundary_names if n in result_dict}
        if boundary_feed:
            self._buffer.append(boundary_feed)

        return [result_dict[n] for n in requested]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sess, name)


def _make_boundary_callback(boundary_batches: list[dict]) -> Callable:
    """Return a compute_encodings callback that feeds pre-collected boundary tensors."""

    def _callback(stage_sim: Any, _ignored_args: Any) -> None:
        for feed_dict in boundary_batches:
            stage_sim.session.run(None, feed_dict)

    return _callback


def _make_stage_sim(
    stage_model: Any,
    quant_init_kwargs: dict,
    device_id: int,
) -> Any:
    """Construct a fresh QuantSim for one pipeline stage on a specific GPU."""
    from octopus_aimet._patch import _Original as _OriginalQSim

    kwargs = dict(quant_init_kwargs)
    kwargs["providers"] = [
        ("CUDAExecutionProvider", {"device_id": device_id}),
        "CPUExecutionProvider",
    ]
    return _OriginalQSim(stage_model, **kwargs)


def sharded_compute_encodings(
    sim: Any,
    forward_pass_callback: Callable,
    callback_args: Any,
) -> None:
    """Sequential stage-by-stage calibration for models that exceed single-GPU VRAM.

    Algorithm:
        1. Partition ONNX graph into N contiguous stages (N = GPU count).
        2. Materialize N sub-models via onnx.utils.extract_model.
        3. For each stage i:
            a. Construct fresh QuantSim on GPU i.
            b. Instrument stage_sim.session to capture stage i's output tensors.
            c. Run compute_encodings:
               - stage 0: uses user's original callback + args
               - stage i>0: uses boundary_callback that feeds tensors captured from stage i-1
            d. Export stage encodings.
            e. Free GPU memory (del stage_sim).
        4. Merge all stage encoding dicts (per-tensor min/max).
        5. Load merged encodings into the driver sim.

    Activation buffer: boundary tensors are captured in-memory as numpy arrays.
    For very large models, boundary tensor size = batch_size × N_batches × tensor_dims.
    Monitor RAM usage when calibrating with many batches.
    """
    import onnx  # type: ignore[import-untyped]

    from octopus.discovery import discover_gpus
    from octopus.sharding.onnx_partition import materialize_stage_models, plan_stages_from_onnx
    from octopus_aimet._merge_encodings import merge_encoding_dicts

    gpus = discover_gpus()
    num_stages = len(gpus)

    _log.info("Sharded calibration: %d stages across %d GPUs", num_stages, num_stages)

    all_encoding_dicts: list[dict] = []
    boundary_batches: list[dict] | None = None

    with tempfile.TemporaryDirectory() as model_dir:
        onnx_path = os.path.join(model_dir, "model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(sim._octopus_model_proto_bytes)

        stage_plans = plan_stages_from_onnx(onnx_path, num_stages=num_stages)

        with tempfile.TemporaryDirectory() as stage_dir:
            artifacts = materialize_stage_models(onnx_path, stage_plans, output_dir=stage_dir)

            for stage_idx, (artifact, gpu) in enumerate(zip(artifacts, gpus)):
                _log.info("Calibrating stage %d on GPU %d", stage_idx, gpu.device_id)

                stage_model = onnx.load(artifact.onnx_path)
                stage_sim = _make_stage_sim(
                    stage_model, sim._octopus_init_kwargs, gpu.device_id
                )

                if stage_idx == 0:
                    stage_callback = forward_pass_callback
                    stage_callback_args = callback_args
                else:
                    assert boundary_batches is not None
                    stage_callback = _make_boundary_callback(boundary_batches)
                    stage_callback_args = []  # not used by boundary_callback

                # Instrument session to capture boundary tensors (all but last stage)
                is_last = stage_idx == num_stages - 1
                capture_buffer: list[dict] = []
                real_session = None
                if not is_last:
                    real_session = stage_sim.session
                    stage_sim.session = _OutputCapturingSession(
                        real_session,
                        list(artifact.output_names),
                        capture_buffer,
                    )

                stage_sim.compute_encodings(stage_callback, stage_callback_args)

                # Restore real session and harvest captured boundary tensors
                if not is_last:
                    stage_sim.session = real_session
                    boundary_batches = capture_buffer
                    _log.debug(
                        "Stage %d captured %d boundary batches (%d tensors each)",
                        stage_idx,
                        len(capture_buffer),
                        len(next(iter(capture_buffer), {}).keys()),
                    )

                # Export stage encodings
                with tempfile.TemporaryDirectory() as enc_dir:
                    stage_sim.export(
                        path=enc_dir,
                        filename_prefix=f"stage{stage_idx}",
                        export_model=False,
                    )
                    enc_path = _find_encodings_file(enc_dir, f"stage{stage_idx}")
                    with open(enc_path) as f:
                        all_encoding_dicts.append(json.load(f))

                # Release stage GPU memory before next stage
                if hasattr(stage_sim, "session") and stage_sim.session is not None:
                    stage_sim.session = None
                del stage_sim

    merged = merge_encoding_dicts(all_encoding_dicts)

    with tempfile.TemporaryDirectory() as merged_dir:
        merged_path = os.path.join(merged_dir, "merged.encodings")
        with open(merged_path, "w") as f:
            json.dump(merged, f)
        _load_encodings_to_sim(sim, merged_path, strict=False)

    _log.info("Sharded calibration complete. Merged encodings loaded into sim.")
