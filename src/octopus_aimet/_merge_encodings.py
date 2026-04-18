from __future__ import annotations

from typing import Any


def merge_encoding_dicts(encoding_dicts: list[dict]) -> dict:
    """Merge per-tensor min/max across K calibration workers' encoding dicts.

    Correct for QuantScheme.post_training_tf (min/max observers): the global
    min/max is the min/min and max/max across all workers' local observations.

    Approximate for histogram-based schemes (percentile, KL divergence):
    splitting the dataset and merging min/max gives a conservative (wider)
    range. For those schemes, use single-worker calibration instead.

    Encoding dict structure (AIMET ONNX):
        {
          "activation_encodings": {
            "<tensor>": {"min": float, "max": float, "bitwidth": int, ...}
          },
          "param_encodings": {
            "<weight>": [{"min": float, "max": float, ...}, ...]  # per-channel list
          },
          "version": "1.0"
        }
    """
    if not encoding_dicts:
        return {}

    merged_act: dict[str, Any] = {}
    merged_param: dict[str, list] = {}
    version = encoding_dicts[0].get("version", "1.0")

    for enc in encoding_dicts:
        for tensor, stats in enc.get("activation_encodings", {}).items():
            if tensor not in merged_act:
                merged_act[tensor] = dict(stats)
            else:
                merged_act[tensor]["min"] = min(merged_act[tensor]["min"], stats["min"])
                merged_act[tensor]["max"] = max(merged_act[tensor]["max"], stats["max"])

        for tensor, per_channel in enc.get("param_encodings", {}).items():
            if tensor not in merged_param:
                merged_param[tensor] = [dict(s) for s in per_channel]
            else:
                for i, stats in enumerate(per_channel):
                    if i < len(merged_param[tensor]):
                        merged_param[tensor][i]["min"] = min(
                            merged_param[tensor][i]["min"], stats["min"]
                        )
                        merged_param[tensor][i]["max"] = max(
                            merged_param[tensor][i]["max"], stats["max"]
                        )

    return {
        "activation_encodings": merged_act,
        "param_encodings": merged_param,
        "version": version,
    }
