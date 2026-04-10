# autoresearch

This repository is now a minimal workspace for an agentic research-paper visualizer.

The project is driven by one file: [program.md](program.md). It instructs an agent to take a research paper (plus optional GitHub repo/URL) and generate a structured, clickable curation folder using:

1. Obsidian canvas (`.canvas`)
2. Mermaid diagrams (`.mmd`)
3. Markdown explanation pages
4. Linked code snippet sidecars

## What This Repo Contains

1. [program.md](program.md): the execution spec for the visual curation workflow.
2. [README.md](README.md): this high-level description.

## Usage

1. Open your coding agent in this repository.
2. Ask it to follow [program.md](program.md) for a paper and optional code repository.
3. The agent should create a new paper-specific folder with overview, diagrams, nested step pages, and snippet evidence.

## License

MIT
