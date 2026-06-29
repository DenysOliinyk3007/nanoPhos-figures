# Repository structure

This document describes the folder layout used in `uPhosHT` and the
conventions behind it. The same structure is intended to be reused for
other projects (proteomics or otherwise) — the directory names and
their roles are general, only the specific filenames are domain-bound.

---

## Top-level layout

```
<project_root>/
├── README.md                  Project overview, install, quick start, citation
├── .gitignore                 Tracking policy (see "Tracked vs untracked")
├── requirements.txt           Pinned Python dependencies
│
├── src/                       Production Python modules (tracked)
├── docs/                      Documentation, context, change logs (tracked)
├── data/                      Small reference data / metadata (tracked)
│
├── raw_data/                  Vendor exports, untouched (untracked, large)
├── processed_data/            Derived intermediate tables (untracked, regenerable)
├── cc_runs/                   Tool-specific run outputs, organized by experiment
│                              (untracked except project-level summary tables)
├── archive/                   Frozen older code, exploratory notebooks, refs
│                              (untracked)
│
├── .streamlit/                Streamlit app config (untracked)
├── .claude/                   Claude Code settings (untracked)
│
└── <project>_Figure{N}_v{NN}.ipynb     One self-contained notebook per figure
```

A concrete snapshot for `uPhosHT`:

```
uPhosHT/
├── README.md
├── .gitignore
├── src/
│   ├── PeptideCollapse_v3.py        Site-level collapse (production)
│   ├── PeptideCollapse_v4.py        Next-version draft
│   ├── core.py                      Shared helpers
│   ├── phosphoscape_qc.py           Plate QC
│   ├── phosphoscape_app.py          Streamlit QC app
│   └── phosphoscape_cc.py           CurveCurator wrappers
├── docs/
│   ├── REPO_STRUCTURE.md            (this file)
│   ├── UPHOSHT_MANUSCRIPT_CONTEXT.md
│   ├── UPHOS_HT_FIGURE{1,2,3,4}_CONTEXT.md
│   ├── PeptideCollapse_v{2,3}_changelog.md
│   ├── PeptideCollapse_theory_and_roadmap.md
│   ├── NOTEBOOK_REFACTORING_GUIDE.md
│   └── PHOSPHOSCAPE_ANALYSIS_CLAUDE.md
├── data/
│   ├── names_rapamycin_columns.csv
│   ├── selectivity.csv
│   ├── kinases.xlsx
│   └── PhosphoScape_Dilution_Protocol_*.doc
├── raw_data/                        Spectronaut DIA `_Report.tsv` exports
├── processed_data/                  Collapsed site tables (`*_collapsed.csv`)
├── cc_runs/
│   └── rapamycin/                   One subdir per experiment / drug
│       ├── t{5,15,30,60,120,240,480}/   Per-timepoint CurveCurator runs
│       │   ├── config.toml          Run config
│       │   ├── input.tsv            CurveCurator-formatted input
│       │   ├── curves.txt           Per-curve fits + statistics
│       │   ├── decoys.txt           Decoy fits (FDR estimation)
│       │   ├── fdr.txt              FDR table
│       │   ├── dashboard.html       Interactive QC report
│       │   └── curveCurator.log
│       ├── curves_all_timepoints.tsv    Cross-timepoint summary
│       ├── kora_per_cluster.tsv         Kinase-substrate enrichment
│       ├── reactome_per_cluster.tsv     Pathway enrichment
│       ├── kinetic_4pl_fits.tsv         Per-site 4PL parameters
│       └── kinase_library_*.tsv         Johnson 2023 motif scores
├── archive/                         Older versions, exploratory work, PDFs
├── uPhosHT_Figure1_v00.ipynb        Volume optimization
├── uPhosHT_Figure2_v00.ipynb        Gradient + dilution series
├── uPhosHT_Figure3_v00.ipynb        Full-plate reproducibility
└── uPhosHT_Figure4_v00.ipynb        PhosphoScape rapamycin pilot
```

---

## What each directory is for

### `src/` — production code (tracked)
Python modules imported by the figure notebooks. One module per coherent
responsibility (collapse pipeline, QC, app, shared helpers). Versioned
filenames (`Foo_v3.py`, `Foo_v4.py`) when an algorithm change is large
enough that downstream notebooks need to pin a specific version; the
previous version stays in `src/` until all notebooks have migrated, then
moves to `archive/`.

### `docs/` — documentation (tracked)
- `*_CONTEXT.md` — per-figure briefing: experimental design, locked
  pipeline, decisions baked in, panel layout, rejected attempts,
  literature anchors, caption draft. Written so a future collaborator
  (or you, six months later) can reconstruct the reasoning.
- `*_changelog.md` — per-module change history with rationale for each
  change.
- `*_theory_and_roadmap.md` — methods reasoning that doesn't fit in a
  changelog.
- `MANUSCRIPT_CONTEXT.md` — running draft of methods/results text.
- `NOTEBOOK_REFACTORING_GUIDE.md` — checklist for getting notebooks to
  publication readiness.

### `data/` — reference / metadata (tracked, small)
Stable inputs that are not vendor exports: column-name maps, kinase
tables, protocol documents, hand-curated lists. Anything small enough
that committing it is fine and that defines the project's interpretation
of the raw data.

### `raw_data/` — vendor exports (untracked, large)
Untouched outputs from the instrument vendor / search engine
(Spectronaut `_Report.tsv` here). Read-only by convention. Filenames
preserve the vendor's date prefix and experiment tag.

### `processed_data/` — intermediate tables (untracked, regenerable)
Derived outputs that are slow to recompute but reproducible from
`raw_data/` + `src/` (e.g. site-collapsed CSVs). Untracked because they
are large and reproducible. Notebooks should be able to regenerate
everything here from scratch.

### `cc_runs/` — tool runs (mostly untracked)
External-tool output directories, organized by experiment. The
convention `cc_runs/<experiment>/t<minutes>/` came from CurveCurator
per-timepoint runs but generalizes to any tool that needs one
subdirectory per condition / split. Project-level summary tables
(`*.tsv` aggregates) sit at `cc_runs/<experiment>/` and may be tracked
selectively if they are small and load-bearing for downstream code.

### `archive/` — frozen (untracked)
Older code versions, exploratory notebooks, reference PDFs, deprecated
plots. Move things here rather than deleting — useful for reconstructing
why a decision was made. Not tracked because it would bloat the repo.

### `<project>_Figure{N}_v{NN}.ipynb` — figure notebooks (tracked)
One notebook per manuscript figure, at the repo root. Self-contained
(`Restart → Run All` reproduces the figure given `raw_data/` + `data/`).
Version suffix `v00`, `v01`, ... for major rewrites; minor edits use git
history. See `docs/NOTEBOOK_REFACTORING_GUIDE.md`.

---

## Tracked vs untracked policy

The `.gitignore` enforces:

| Tracked | Untracked |
|---|---|
| `src/`, `docs/`, `data/`, `*.ipynb` at root | `raw_data/`, `processed_data/`, `archive/` |
| `README.md`, `.gitignore`, `requirements.txt` | `.streamlit/`, `.claude/`, `uPhosHT-env/`, `.ipynb_checkpoints/`, `__pycache__/` |
| `Figures/raw/*.pdf` (figure panel exports, optional) | `*.png`, `*.html`, `*.zip`, `*.doc(x)`, `*.ai`, `*.psd`, `*.fasta`, `*.log`, `*.pyc` |

Rule of thumb:
- **Track**: code, documentation, small reference data, notebooks, the
  rendered figure panels you cite in the manuscript.
- **Don't track**: anything large (vendor exports), anything regenerable
  (intermediate tables, tool outputs), anything user-specific (env,
  IDE config), binaries (images, PDFs, archives — except final figure
  PDFs in `Figures/raw/`).

When in doubt, untrack. A regenerable artifact missing from git is
recoverable; a 2 GB raw report committed by accident is painful to
remove from history.

---

## Naming conventions

- **Notebooks**: `<project>_Figure{N}_v{NN}.ipynb` — zero-padded version,
  one notebook per figure, root level.
- **Per-condition subdirs**: `t{minutes}/` for time, `d{nM}/` for dose,
  etc. — one short letter, no separator, integer value. Avoid
  `timepoint_5_minutes/`-style verbosity.
- **Experiment subdirs in `cc_runs/`**: `<drug>/` or `<perturbation>/`,
  lowercase, no spaces.
- **Module versions**: `Foo_v3.py`, with a matching
  `docs/Foo_v3_changelog.md`. Keep `Foo_v2.py` in `src/` until all
  notebooks have migrated, then move to `archive/`.
- **Raw data filenames**: preserve the vendor prefix
  (`<YYYYMMDD>_<HHMMSS>_<tag>_Report.tsv`). Don't rename — the prefix is
  the experiment-instrument link.
- **Processed data filenames**: descriptive, ending with the
  transformation applied (`*_collapsed.csv`, `*_normalized.csv`,
  `*_filtered.csv`). One transformation per filename suffix.
- **Doc filenames**: `<TOPIC>_CONTEXT.md` for per-figure / per-experiment
  briefings, `<MODULE>_changelog.md` for module change histories, all-
  caps for project-wide docs (`README.md`, `REPO_STRUCTURE.md`).

---

## Bootstrapping a new project with this structure

```bash
# 1. Create directories
mkdir -p src docs data raw_data processed_data archive
mkdir -p Figures/raw    # optional, for tracked figure-panel PDFs

# 2. Seed the .gitignore (copy from a sibling project, then edit):
#    untrack: raw_data/, processed_data/, archive/, .streamlit/, .claude/,
#             __pycache__/, .ipynb_checkpoints/, *.log, *.png, *.html,
#             *.zip, large binaries
#    track:   src/, docs/, data/, *.ipynb, README.md, requirements.txt

# 3. Seed the docs:
#    - REPO_STRUCTURE.md (this file, copy and adapt)
#    - README.md (overview, install, quick start, citation)
#    - <PROJECT>_FIGURE1_CONTEXT.md per figure as you start it
#    - <MODULE>_changelog.md per non-trivial module

# 4. Seed src/core.py with shared helpers; add modules as needed.

# 5. Create the first figure notebook at root:
#    <project>_Figure1_v00.ipynb

# 6. requirements.txt — pin versions used in production analysis.
```

For non-proteomics projects, the directory roles map directly:

| Role | uPhosHT | Generic |
|---|---|---|
| Vendor / source data | `raw_data/` Spectronaut TSVs | any untouched input |
| Derived tables | `processed_data/` collapsed CSVs | model outputs, parsed JSON, etc. |
| External-tool runs | `cc_runs/<drug>/t<min>/` CurveCurator | any `<tool>/<experiment>/<split>/` outputs |
| Reference data | `data/` kinases, column maps | lookup tables, configs, fixtures |
| Frozen old work | `archive/` v2 collapse, exploratory plots | same |

Only `cc_runs/` has a tool-specific name. Rename it to whatever the
project's primary external tool is (`mcmc_runs/`, `simulations/`,
`benchmarks/`...) and keep the same `<experiment>/<condition>/` two-
level structure inside.

---

## Why this layout

- **Separation of inputs, code, and outputs** — `raw_data/` is read-only,
  `src/` is the only place code lives, everything else is regenerable
  from those two. No ambiguity about what is canonical.
- **Track what humans wrote, not what code generated** — tracking only
  source code, docs, and small reference data keeps git history
  meaningful. Derived artifacts go in their own dirs and are
  regenerated, not versioned.
- **One notebook per figure** — keeps notebooks short enough to read
  end-to-end and avoids cross-notebook hidden state. `v00`/`v01`
  suffixes mark major rewrites without losing history.
- **Per-figure context docs** — figures encode decisions that
  are not visible in the notebook code (why this cutoff, what was
  tried and rejected, what the caption is meant to say). The
  `*_CONTEXT.md` doc is where that reasoning lives.
- **`archive/` instead of deleting** — preserves the trail of why a
  decision was made without bloating the active workspace.
