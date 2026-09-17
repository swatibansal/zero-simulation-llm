# zero-sim showcase page — design

Date: 2026-09-16. Status: approved (content generated from notebook, full story, docs/ on main).

## Goal

A single static page at `docs/index.html`, served by GitHub Pages from `main:/docs` at
`https://swatibansal.github.io/zero-simulation-llm/`, presenting the whole lab: cluster +
collectives, the four training strategies (DP, ZeRO-1/2/3) each with its real code and real
output, the memory/communication/time results with plots, the 7B–405B scaling analysis, and
the mapping to real systems.

## Approach

The executed notebook `zero_sim_demo.ipynb` is already the single source of truth for code,
outputs, and figures. A generator, `tools/build_site.py` (Python stdlib only, no new
dependencies), reads it and emits the page. Hand-crafting the page was rejected because
copied code/outputs drift from the notebook; plain nbconvert was rejected because it looks
like a notebook, not a designed page.

## Components

- `tools/build_site.py` — the generator.
  - Markdown cells → HTML via a converter handling exactly the constructs the notebook
    uses: `#`/`##`/`###` headings, `**bold**`, `*italic*`, `` `code` ``, tables,
    blockquotes, unordered (`*`) and ordered (`1.`) lists, paragraphs. It raises on any
    unrecognized block construct so content is never silently mangled.
  - Code cells → visible-by-default code blocks with an "In" gutter label, highlighted
    client-side by highlight.js (CDN).
  - Outputs → `stream` / `execute_result` text as terminal-styled "Out" panels;
    `display_data` `image/png` embedded as base64 `<img>` (self-contained page, figures can
    never mismatch their code). Unknown output types are an error.
  - The first markdown cell's `#` title is dropped; a hero header replaces it with the
    title, a one-line summary, links to the repo and notebook, and the three headline
    claims (bitwise-identical weights; 16Ψ → 16Ψ/N memory; 2Ψ vs 3Ψ traffic).
  - A sticky nav is generated from `##` headings (anchor slugs), with a small scroll-spy
    script. No JS framework; only highlight.js + ~15 lines of vanilla JS.
- `docs/index.html` — committed generated output (~1 MB with embedded figures).
- `docs/.nojekyll` — disable Jekyll processing.
- `tests/test_build_site.py` — builder runs; one `<h2>` per notebook `##` heading; one code
  block per executed code cell; one `<img>` per image output; no course-specific strings.
  Runnable under pytest and as a plain script (`python3 tests/test_build_site.py`).

## Guardrails

- Builder asserts every code cell has an `execution_count` (page can only be built from an
  executed notebook).
- Regeneration command documented in CLAUDE.md next to the notebook re-execution command:
  rebuild the site whenever the notebook is re-executed.

## Deployment

Feature branch → PR → squash-merge to `main` (hook-enforced workflow), then enable Pages via
`gh api` with source `main:/docs`, and verify the published URL responds.

## Out of scope

Interactive/client-side simulation, dark-mode toggle, multi-page layout, editorial rewrites
of notebook prose (the notebook text is rendered as-is).
