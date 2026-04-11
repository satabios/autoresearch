# Fastron

An optimized neural-network graph visualizer built on [Netron](https://github.com/lutzroeder/netron).

Reference implementation: [satabios/fastron](https://github.com/satabios/fastron)

## What It Does

Loads ONNX, PyTorch, TFLite, CoreML, SafeTensors, and 40+ other model formats and renders their computation graphs interactively in a browser. Designed for production-scale models with thousands of nodes and multi-gigabyte weight files.

Key capabilities:

- **Google-Maps-style viewport rendering** — only visible nodes and edges are in the DOM.
- **On-demand weight loading** — tensor data is metadata-only by default; binary weights load per-tensor on click.
- **Adaptive layout** — Dagre for small graphs, fast topological layout for large ones, force-directed as opt-in.
- **Model comparison** — side-by-side diff with structural matching.

## Repository Contents

| File | Purpose |
|---|---|
| [program.md](program.md) | Full architecture spec, known issues, and optimization roadmap |
| [README.md](README.md) | This file |

## Usage

Point your coding agent at [program.md](program.md) to understand the architecture, implement fixes, or extend the application.

## License

MIT
