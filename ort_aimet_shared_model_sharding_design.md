# ORT/AIMET Shared-Model Sharding Design (Stub)

## Why

Current state:
- ORT and AIMET ONNX support replica workers only.
- Shared-model sharding (`1 model across N GPUs`) is not implemented.
- Capability API and constructor guardrails now enforce this policy.

Goal of this doc: define first concrete path to true shared-model runtime.

## Product Goal

Keep same user API:

```python
with Octopus(model=model, eval_fn=eval_fn, sharding_strategy="pp") as o:
    for batch in dataset:
        o.submit(batch)
    out = o.gather()
```

For ORT / AIMET ONNX:
- `pp` means pipeline-parallel graph partitions across GPU stages.
- `max_workers=K` means K parallel shard-groups when enough GPUs exist.

## Non-Goals (first prototype)

- No tensor-parallel (`tp`) for ORT/AIMET.
- No automatic inter-node cluster placement.
- No dynamic repartition during runtime.
- No heterogeneous-batch micro-pipeline optimization.

## Proposed Architecture

### 1) Graph Partition Planner

Input:
- ONNX model path
- available GPU ids
- optional partition hints

Output:
- ordered stage plans (node sets)
- stage input/output tensor names
- estimated stage memory/compute cost

Prototype strategy:
- topological split into contiguous stage windows
- greedy balance by node-count fallback
- optional cost hints from ORT profiling run

### 2) Partition Materializer

Build per-stage ONNX subgraphs:
- `stage_0.onnx ... stage_n.onnx`
- explicit interface tensor schema between stages
- external data preserved when model uses `.onnx.data`

### 3) Runtime Worker Group (pipeline)

Add new worker-group runtime for ONNX stages:
- one actor per stage GPU
- stage actor loads stage session on pinned GPU
- batch flow: stage0 -> stage1 -> ... -> stageN
- return final outputs to caller

Concurrency:
- one shard-group = one full stage chain
- K parallel groups via existing `max_workers` semantics

### 4) AIMET ONNX Layer

AIMET ONNX uses ONNX export/session path; piggyback on ORT staged runtime:
- export calibrated sim model to ONNX
- partition exported model
- keep quant params/encodings intact

## API / Capability Changes

No user API change in first pass.

Capability matrix plan:
- keep ORT/AIMET `shared_model_multi_gpu=False` until prototype stable
- flip to `True` only after runtime + tests pass

## Milestones

1. M1: Planner + partition materializer (offline, unit-tested)
2. M2: ORT staged runtime for `pp` single shard-group
3. M3: ORT staged runtime for `max_workers=K` multiple shard-groups
4. M4: AIMET ONNX integration on staged runtime
5. M5: GPU integration + regression tests + docs

## Testing Plan

Unit:
- partition graph validity (all nodes covered once, topo-safe)
- stage IO contract correctness
- external-data ONNX partition copy/restore

GPU integration:
- ORT tiny model on 2 GPUs, `pp`, deterministic compare vs baseline
- ORT `pp + max_workers=K` on >=4 GPUs (or simulated reduced K)
- AIMET ONNX staged path numeric sanity vs replica baseline

Failure tests:
- not enough GPUs for requested stage count
- invalid partition output shape mismatch
- worker crash propagation and cleanup

## Open Technical Risks

- ONNX graph partition correctness for control-flow / dynamic shapes
- stage boundary tensor transfer overhead may erase speed gains
- AIMET quant graph metadata may need explicit carry-over handling
- extra memory pressure from intermediate tensors between stages

## Exit Criteria to Mark Feature Done

- ORT `pp` shared-model path production-usable for simple encoder models
- AIMET ONNX `pp` path works on calibrated sims for sensitivity eval/inference
- submit/gather/map loop API unchanged
- constructor guardrails updated to reflect new supported backends
