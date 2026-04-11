# Fastron — Program Specification

Reference implementation: [satabios/fastron](https://github.com/satabios/fastron) (optimized fork of Netron).

## What This Application Does

Fastron is a neural-network graph visualizer. It loads model files (ONNX, TFLite, PyTorch, CoreML, SafeTensors, and 40+ other formats), parses their computation graph, and renders an interactive, navigable visualization of every node, edge, and tensor in the model.

The core problem it solves: production models can have tens of thousands of nodes, multi-gigabyte weight files, and deeply nested subgraphs. A naive visualizer that renders everything at once will freeze the browser, exhaust memory, or produce an unusable wall of boxes. Fastron addresses this with a Google-Maps-style approach — only the nodes and edges currently in the viewport are rendered, weights are loaded on demand, and the rendering pipeline is incremental.

## Architecture Overview

The application has six main subsystems. All rendering is pure SVG in the browser DOM — there is no Canvas or WebGL rendering path.

### 1. Graph Parsing and Model Loading

**Key files:** `source/base.js`, format-specific handlers (e.g., ONNX, TFLite)

The model file is read via chunked I/O. `browser.BrowserFileContext.request()` reads local files in configurable chunks (`fileReadChunkSizeMB`, default 512 MB). For large files, data flows through `browser.FileStream`, which maintains a sliding buffer window (`streamWindowSizeMB`, default 512 MB) and refills on demand.

During parsing, the `base.Tensor` constructor checks `window.NETRON_CONFIG.skipTensorWeights`. When true (the default), the binary tensor data is never read — only metadata (shape, dtype, name) is extracted. This means a 5 GB ONNX file can be loaded in seconds because the parser skips over the weight blobs entirely.

**Configuration defaults** (set in `source/index.html` via `window.NETRON_CONFIG`):

| Key | Default | Purpose |
|---|---|---|
| `skipTensorWeights` | `true` | Skip weight data during parse |
| `cacheMaxMemoryMB` | `2048` | LRU cache budget |
| `streamingChunkSizeMB` | `10` | Streaming chunk size |
| `streamingThresholdMB` | `50` | File size threshold for streaming mode |
| `fileReadChunkSizeMB` | `512` | Browser file read chunk |
| `streamWindowSizeMB` | `512` | In-memory sliding window |
| `maxLayoutWorkers` | `4` | Web Worker budget for layout |
| `gpuAcceleration` | `true` | GPU compositor hints on SVG layers |

### 2. Google-Maps-Style Viewport Rendering

**Key files:** `source/grapher.js` (`TileManager`, `ViewportObserver`, `Graph`)

This is the core optimization. The graph canvas is divided into a spatial hash grid by `TileManager`. Tile size adapts to node count:

| Node count | Tile size (px) |
|---|---|
| < 500 | 200 |
| < 2000 | 300 |
| < 5000 | 500 |
| >= 5000 | 800 |

Each node and edge is registered into the tiles it overlaps. When the user scrolls or zooms, `ViewportObserver` fires (debounced at 150 ms, with a 50 px movement threshold), and `TileManager.queryViewport()` returns the set of nodes and edges visible in the current viewport (plus a 2-tile buffer in each direction).

The visibility update pipeline:

1. `view.Graph.register()` sets up scroll/wheel/pointer handlers and the `ViewportObserver`.
2. `_onViewportChange(viewport)` converts the scroll position to graph coordinates via `_getViewportBounds()`.
3. `grapher.Graph.updateViewportVisibility(bounds)` queries the `TileManager`, computes the delta (added/removed nodes and edges), and ensures edge endpoint nodes are always visible even if outside the viewport.
4. `grapher.Graph.updateVisibleElements(document, delta)` applies the delta: showing nodes by reattaching DOM elements or toggling `display:none`, and hiding nodes inversely.

There are three tiers of deferred DOM construction based on graph size:

| Node count | Strategy |
|---|---|
| < 500 | Full SVG build upfront |
| 500-5000 | Simplified `<rect>` placeholders until the node enters the viewport |
| > 5000 | Zero DOM — nothing is created until the node scrolls into view |

When a node first enters the viewport, `_ensureNodeElement()` builds the full SVG subtree (header, argument lists, attributes). This is done in chunks of 30 nodes per `requestIdleCallback` frame via `_buildVisibleNodes()`. Edges are built 60 per frame.

After initial build, `hideAllNodes()` hides everything, then `restore()` triggers the first viewport update to reveal only what is visible.

### 3. On-Demand Weight Loading

**Key files:** `source/base.js` (`Tensor` class), `source/view.js` (`TensorView`, `toggleWeights`)

Weights are off by default. The `base.Tensor` constructor sets `this._skipWeights = true` when the global config says so, and `_read()` short-circuits. The `values` and `data` getters re-check at access time, so toggling the config at runtime takes effect on the next read.

When a user clicks a node and opens its tensor detail panel, `view.TensorView.get content()` checks `value._deferred` and `skipWeights`. If either is true, only metadata is shown (shape, type, estimated size). On explicit user click, `value.read()` loads the binary data for that single tensor without reparsing the entire file.

The toggle button (`view.View.toggleWeights()`) flips the config flag, updates the toolbar button CSS class, and reloads the model.

### 4. Layout Engine

**Key files:** `source/grapher.js`, `source/view.js` (worker management)

Three layout engines are available, selected by node count:

1. **Dagre via Web Worker** (< 3000 nodes): Standard layered graph layout. `view.Worker` manages lifecycle with budget limiting (`_workerLimit`) and timeout fallback.
2. **`_fastLayout()`** (>= 3000 nodes): O(N+E) topological sort with longest-path rank assignment and barycenter crossing minimization (4 sweeps). Much faster than Dagre for large graphs.
3. **`_forceLayout()`** (opt-in via `layout='force'`): Spring-repulsion simulation with Coulomb repulsion, Hooke springs, centroid gravity, and AABB collision detection. 400 iterations with cooling.

For graphs > 500 nodes with deferred rendering, `useEstimatedNodeSizes()` returns true and layout uses fixed size constants (`_estimatedNodeWidth=150`, `_estimatedNodeHeight=65`) instead of calling `getBBox()`, avoiding expensive forced reflows.

### 5. Memory Management

**Key files:** `source/cache-manager.js`, `source/browser.js`

`CacheManager` is a Map-based LRU cache with a configurable memory ceiling (default 2048 MB). `set(key, data)` evicts oldest entries when the budget is exceeded. `getCacheKey(file)` generates keys from File objects.

DOM detachment mode (`_detachInvisible`): when enabled, hidden nodes and edges are removed from the DOM entirely instead of being set to `display:none`. This reduces the browser layout tree size for very large graphs, at the cost of more expensive per-toggle reattachment.

GPU compositor hints (`will-change: transform`, `transform: translateZ(0)`, `contain: strict`) are applied to the SVG container during `view.Graph.build()`. These promote layers to the GPU compositor but do not change the rendering path — all drawing remains SVG.

### 6. Model Comparison

**Key files:** `source/comparator.js`, `source/comparator.html`

Side-by-side comparison of two models uses `comparator.Controller` with two independent `Graph` instances, each with its own layout worker and viewport culling. Navigation is synchronized — scroll and zoom mirror between the left and right panels.

Matching pipeline:

1. **Fast O(n) pass:** Weisfeiler-Leman structural hashing (2 rounds), then exact name matching (two passes: by type+name+group, then by type+name).
2. **Deferred Hungarian pass:** Unmatched nodes are grouped by op type. Groups <= 200 nodes use O(n^3) Hungarian matching on a similarity cost matrix. Groups > 200 use greedy matching. Cross-type matching groups remaining nodes by category and repeats. Thresholds: 40 (same-type), 50 (cross-type). One type-group is processed per frame via `setTimeout(..., 0)`.

Similarity is scored on a 100-point scale: op type (50), attributes (30), shapes + connectivity (20). Diff results are rendered with CSS classes (`node-diff-modified`, `node-diff-added`, `node-diff-removed`).

## Known Issues and Required Fixes

These are bugs and optimization gaps in the current codebase that should be addressed.

### 1. LRU Cache Does Not Update Access Order

**File:** `source/cache-manager.js`

`CacheManager.get()` returns the cached value but does not move the entry to the end of the Map. This means frequently accessed entries can be evicted while rarely accessed entries survive, which is the opposite of LRU behavior.

**Fix:** On `get()`, delete the key and re-insert it so it moves to the tail of Map iteration order.

### 2. ViewportObserver Debounce Has No Leading Edge

**File:** `source/grapher.js` (`ViewportObserver`)

The 150 ms debounce is trailing-only. The first frame of any scroll or zoom interaction shows stale content because the callback does not fire until 150 ms after motion starts.

**Fix:** Use a leading+trailing debounce. Fire immediately on the first event, then suppress until the trailing edge. This removes the initial stale frame while still batching rapid scroll events.

### 3. Force Layout is O(n^2) Without Spatial Acceleration

**File:** `source/grapher.js` (`_forceLayout`)

Coulomb repulsion computes every node-pair distance. For 5000+ nodes this is 25 million distance calculations per iteration x 400 iterations.

**Fix:** Replace the all-pairs loop with a Barnes-Hut quadtree approximation (theta ~ 0.9). This reduces repulsion from O(n^2) to O(n log n) per iteration. The quadtree can be rebuilt each iteration in O(n log n).

### 4. TileManager Registration Cost for Oversized Nodes

**File:** `source/grapher.js` (`TileManager.addNode`)

A node whose bounding box spans many tiles is registered into every tile it overlaps. For a node spanning 10x10 tiles, that is 100 tile entries for a single node.

**Fix:** For nodes larger than a configurable tile-span threshold (e.g., 4x4 tiles), register them in a separate oversized-node set that is always included in viewport queries. This caps per-node registration cost at O(1).

### 5. Hungarian Matching Blocks the Main Thread

**File:** `source/comparator.js`

The inter-group scheduling uses `setTimeout(..., 0)` to yield between type-groups, but the Hungarian algorithm within a single type-group is fully synchronous. A group of 200 Conv nodes runs O(200^3) = 8 million iterations on the main thread without yielding.

**Fix:** Chunk the Hungarian inner loops. After every N iterations of the outer loop (e.g., 50), yield via `setTimeout` or `requestIdleCallback` and resume. Alternatively, move the entire matching phase into a Web Worker.

### 6. No Adaptive Threshold for DOM Detachment vs Display Toggle

**File:** `source/grapher.js`

`_detachInvisible` is a binary flag with no documented heuristic for when to enable it. DOM detachment reduces layout tree size but increases scroll jank because reattaching elements is more expensive than toggling `display:none`.

**Fix:** Set a node-count threshold. For graphs below a threshold (e.g., 10000 nodes), use `display:none` toggling. Above it, use DOM detachment. Expose the threshold in `NETRON_CONFIG` so users can tune it.

### 7. SVG ViewBox Bottleneck at Extreme Scale

At 100k+ nodes, the SVG coordinate space itself becomes a browser performance bottleneck regardless of viewport culling, because the SVG renderer must maintain the full coordinate system.

**Fix (long-term):** For models above a configurable node threshold (e.g., 50000), switch the rendering backend from SVG to HTML5 Canvas with a virtual coordinate system. The Canvas path draws only what is visible and does not maintain persistent DOM elements. This is a significant architectural change and should be feature-flagged.

## Optimization Roadmap

These are enhancements beyond bug fixes, ordered by impact.

### Phase 1 — Immediate Wins

1. Fix LRU cache access-order bug.
2. Add leading-edge debounce to ViewportObserver.
3. Add adaptive DOM detachment threshold.
4. Add oversized-node bypass in TileManager.

### Phase 2 — Large Model Performance

5. Barnes-Hut quadtree for force layout.
6. Chunk or worker-ify Hungarian matching in comparator.
7. Add level-of-detail (LOD) rendering: at low zoom levels, collapse subgraphs into single summary nodes with an expand-on-zoom interaction. This reduces visible node count without losing information.

### Phase 3 — Extreme Scale

8. Canvas rendering backend for 50k+ node models.
9. Streaming graph parse: begin layout and rendering before the entire file is parsed. Feed nodes to the layout engine incrementally as they are decoded.
10. Web Worker graph parse: move the full model parse off the main thread so the UI remains responsive during load.

## Design Principles

1. **Viewport first.** Never render what the user cannot see. Every node and edge must be gated by viewport visibility before any DOM work is done.
2. **Weights are opt-in.** Tensor data is metadata-only by default. Binary weight data is loaded per-tensor on explicit user action. This keeps load times proportional to graph topology, not file size.
3. **Progressive disclosure.** The initial view shows the top-level pipeline. Subgraphs, attributes, and tensor details are revealed on interaction, not on load.
4. **Degrade gracefully.** Small models (< 500 nodes) get full upfront rendering for instant interactivity. Medium models get placeholder nodes. Huge models get zero-DOM deferred construction. The transitions should be invisible to the user.
5. **Keep the main thread free.** Layout, parsing, and matching computations that exceed ~16 ms should be chunked, deferred, or moved to Web Workers.
