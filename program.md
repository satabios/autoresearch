# Octopus Program Status

## Goal

Unified for-loop wrapper for GPU parallel inference:

```python
with Octopus(model=model, eval_fn=eval_fn) as o:
    for x in inputs:
        o.submit(x)
    results = o.gather()
```

Also supported:

```python
results = o.map(inputs)
for y in o(dataset):
    ...
```

## Current Support

Flow A:
- one worker per usable GPU via `workers_per_gpu=1`

Flow B:
- K parallel workers via `max_workers=K` (or auto-pack)

Shared-model across N GPUs:
- PyTorch `pp`: supported
- PyTorch `tp`: supported (current scope: `nn.Sequential` linear stacks)
- ORT `pp`: supported
- AIMET ONNX `pp`: supported

## Runtime Notes

- `tp` remains PyTorch-only
- ONNX-family shared-model runtime is `pp` only
- `tp` for ONNX-family is not implemented

## Canonical Shared-Model Snippets

PyTorch TP:

```python
with Octopus(model=seq_model, eval_fn=eval_fn, sharding_strategy="tp", gpu_ids=[0, 1]) as o:
    out = o.map(batches)
```

ORT PP:

```python
with Octopus(model="/path/model.onnx", eval_fn=eval_fn, sharding_strategy="pp", gpu_ids=[0, 1]) as o:
    out = o.map(feed_dict_batches)
```

AIMET ONNX PP:

```python
with Octopus(model=calibrated_sim, eval_fn=eval_fn, sharding_strategy="pp", gpu_ids=[0, 1]) as o:
    out = o.map([None] * N)
```

## Regression Status

- non-GPU suite: pass
- GPU suites: PyTorch + ORT + AIMET ONNX pass together

## Next Engineering Tasks

1. Repo cleanup:
   - reduce duplication between core/worker/adapters
   - tighten module boundaries for sharding runtime
   - remove stale compatibility code paths where safe
2. Docs refresh:
   - align README + this file + tests with current capability matrix
   - add concise examples for `tp`, ORT `pp`, AIMET ONNX `pp`
