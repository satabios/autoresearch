# autoresearch

This repository now serves a different task: turn a research paper plus an optional GitHub repository/URL into a visual, clickable curation folder that explains how the work functions.

The output should feel like a guided map of the paper. A reader should be able to start from a top-level pipeline view, click into any block, and then descend into progressively more detailed “inception-like” subgraphs, with short Markdown explanations and relevant code snippets beside each step.

## Goal

Given:

1. A research paper, and
2. Optionally a GitHub repo or URL that implements or relates to the paper,

build a new folder for that paper that visually explains:

1. The paper’s overall pipeline.
2. Each major stage of the method.
3. The dependencies between stages.
4. The implementation details in the code, when source is available.
5. The most important snippets that support each explanation.

The result should be easy to browse in Obsidian, but also readable as plain Markdown and Mermaid files without any special tooling.

## Operating principles

Use the paper as the source of truth. Use the repo or URL as supporting evidence when it exists. Do not invent mechanisms, data paths, or implementation details that are not grounded in the supplied sources.

Prefer clarity over decoration. The visuals should make the structure of the work obvious, not merely attractive.

Prefer a shallow top-level map with deeper drill-down pages underneath. The top canvas should show the entire method at a glance; each node should then open a more detailed page or subgraph for that step.

Keep the artifact set deterministic so a reader can predict where to find each explanation, diagram, and snippet.

## Required input handling

When the user gives a paper, do the following:

1. Identify the paper title, core claim, and main method blocks.
2. If a GitHub repo or URL is available, inspect it as the implementation reference.
3. Extract a compact outline of the method before writing any visual artifacts.
4. Decide the paper slug and use it consistently for the generated folder name and all nested files.

If the paper or repo is ambiguous, proceed with the best supported interpretation and note the uncertainty in the overview page rather than blocking.

## Output contract

Create a new folder for each paper, named from a short slug. The folder should contain at least the following:

```text
<paper-slug>/
   README.md
   overview.md
   canvas.canvas
   pipeline.mmd
   sources.md
   steps/
      01-<stage>.md
      02-<stage>.md
   snippets/
      01-<stage>-<snippet>.md
      02-<stage>-<snippet>.md
   diagrams/
      01-<stage>.mmd
      02-<stage>.mmd
```

Use more files if needed, but keep the structure predictable:

1. `README.md` is the human entry point.
2. `overview.md` explains the paper in prose and links to everything else.
3. `canvas.canvas` is the top-level Obsidian canvas that acts as the primary navigation surface.
4. `pipeline.mmd` is the high-level Mermaid map of the whole method.
5. `sources.md` records the paper citation, repo URL, and any other source anchors.
6. `steps/` holds one Markdown page per major stage.
7. `snippets/` holds short code or pseudocode excerpts tied to a single stage.
8. `diagrams/` holds stage-level Mermaid diagrams and subgraphs.

## Visual hierarchy

The structure should be hierarchical:

1. Top level: a single pipeline view that names the major blocks of the method.
2. Middle level: one page and one Mermaid diagram per block, each describing how that block works.
3. Deep level: nested subgraphs or child pages for the important internal operations inside a block.
4. Evidence level: code snippets, equations, or paper passages that justify the explanation.

Every block on the top canvas should have a clear click target to one of the step pages. Each step page should then expose links to its Mermaid subgraph, deeper breakdown pages, and snippet files.

If a step contains multiple internal operations, represent them as an inner graph rather than flattening them into a paragraph. Use the Mermaid diagram for structure and the Markdown page for interpretation.

## Canvas rules

The canvas is the main navigation layer, not the place for long prose.

Use the canvas to show:

1. The full pipeline.
2. The main blocks of the method.
3. The most important transitions between blocks.
4. Clickable links to the step pages.

Each node should point to a file that explains that node in more depth. Avoid duplicate content on the canvas itself.

## Markdown page rules

Each step page should answer four questions:

1. What is this step doing?
2. Why does it exist in the overall method?
3. What are the inputs and outputs?
4. What code or paper text supports this interpretation?

Keep each page short and focused. If a page is getting large, split it into a parent step page and a child page for the sub-operation.

At the top of each step page, include links back to the overview and to any sibling stages so a reader can navigate laterally.

## Mermaid rules

Use Mermaid for the method’s shape and control flow.

The top `pipeline.mmd` should be simple and readable at a glance. The stage-level diagrams can be more detailed and may use nested subgraphs where appropriate.

Prefer Mermaid diagrams that explain:

1. Data flow.
2. Control flow.
3. Stage ordering.
4. Branches, merges, and repeated loops.

Do not use Mermaid for dense prose or for unsupported detail.

## Snippet rules

Snippets are sidecar evidence, not a code dump.

Each snippet file should:

1. Be short.
2. Tie to one step only.
3. Include a brief note about why it matters.
4. Point back to the step page that references it.

If the repo is available, prefer small source excerpts, function signatures, or pseudocode mirrors of the implementation. If source is not available, use paper equations or algorithmic pseudocode instead.

## Sources and grounding

Record the paper citation and repo URL in `sources.md`.

When describing a stage, ground it in one of three ways:

1. Paper text or figure.
2. Repository code.
3. Reasonable synthesis that is explicitly labeled as inference.

Do not present inference as fact. If something is inferred, say so clearly in the page text.

## Completion criteria

A paper run is complete when the folder contains:

1. A top-level overview.
2. A navigable canvas.
3. A Mermaid pipeline map.
4. One page per major stage.
5. One or more nested subgraphs for the deeper operations that matter.
6. Short code or pseudocode snippets that support the explanations.

The final folder should let a reader move from the broad method to the implementation details without losing context.

## Writing style

Write for a technically literate reader who wants structure first and detail second.

Use concise prose, direct labels, and stable file names. Avoid hype, avoid speculation, and avoid dense wall-of-text explanations when a diagram would communicate the same idea more clearly.

If two representations say the same thing, keep the simpler one.
