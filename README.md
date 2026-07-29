# nanoPhos

Phosphoproteomics workflow for nanogram-scale inputs. Analysis code and figure
notebooks for the nanoPhos manuscript (Oliinyk et al., in revision at *Nature
Communications*).

## Quick start

```bash
git clone https://github.com/DenysOliinyk3007/nanoPhos-figures.git
cd nanoPhos-figures
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then set up the data layout (all paths in the notebooks are relative to the repo root):

```
pride_data/     <- extract the MassIVE deposit here, preserving its structure
                   (pride_data/analysis_data/..., incl. .../revision/figureN/ reports,
                    FASTA, Supplementary_Table_3.xlsx, proteome, aq_out results, …)
data/           <- committed reference data (funscores, mouse_kegg_annotation,
                   *_conditionSetup.tsv) — already in the repo
figures/        <- figure outputs, created automatically on run
```

Open a figure notebook and **Restart → Run All** (launch Jupyter from the repo root
so the relative data paths resolve; the notebooks also self-correct their working
directory if opened from inside `notebooks/`):

```bash
jupyter lab notebooks/nanoPhos_Figure2_v01.ipynb
```

### Notebooks

All reproduction notebooks live in [`notebooks/`](notebooks/):

- **`nanoPhos_Figure{2–5}_v01.ipynb`** and **`nanoPhos_Suppl_Figure{1–5}_v01.ipynb`** —
  the revised analysis, with strict per-run Class I localization filtering
  (probability ≥ 0.75), used for the *Nature Communications* revision. These reproduce
  the published main and supplementary figures.

Exploratory and superseded notebooks — the original-submission (`v00`) versions and a
DIA-NN cross-search comparison that is **not** part of the final manuscript — are
retained locally under `archive/` and are not version-controlled.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Production Python modules: `core` (Class I collapse), `analytics_core_V04`, `limma_utils`, `PeptideCollapse_v4`, `alphaPhosHelperFunctions` |
| `docs/` | `REPO_STRUCTURE.md`, changelogs, manuscript context |
| `data/` | Small reference CSVs cited by individual figures |
| `raw_data/` | Spectronaut DIA exports (gitignored, large) |
| `archive/` | Older versions, exploratory notebooks, deprecated work (gitignored) |
| `notebooks/` | Reproduction notebooks: one per main figure (`nanoPhos_Figure{2–5}_v01.ipynb`) and per supplementary figure (`nanoPhos_Suppl_Figure{1–5}_v01.ipynb`), revised analysis |

> **Note:** `src/alphaquant/` and `docs/phosphonetworks/` are large vendored/reference
> trees and are gitignored. AlphaQuant installs from PyPI via `requirements.txt`.

Full description and tracking policy in [`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md).

## Site-level collapse pipeline

The phosphosite collapse implementation lives in
[`src/PeptideCollapse_v4.py`](src/PeptideCollapse_v4.py). It converts Spectronaut
precursor-level BGS exports into a sites × samples quantification matrix.

The default applies **strict per-(site, run) localization filtering** at probability
≥ 0.75: for each phosphosite, the intensity in a given run is retained only if the
localization probability for that site in that specific run meets the cutoff. This
produces Class I localized phosphosite counts in the sense of Olsen et al. (2006) and
matches the per-file pre-filtering convention used by MaxQuant evidence tables and
the µPhos workflow.

A legacy `localization_strategy="global_max"` option preserves the dataset-wide-max
filtering used by the original Hogrebe et al. (2018) PeptideCollapse Perseus plugin —
available for direct A/B comparison.

## Reproducing the figures

Each figure notebook is self-contained. `Restart → Run All` regenerates panel-level
outputs given a populated `raw_data/` and the dependencies in `requirements.txt`.
The first cells of each notebook set the working directory to the repo root and
prepend `src/` to `sys.path`, so the bare module imports below them resolve to the
production code under `src/`.

## Citation

Manuscript in revision. Citation will be added once accepted.

## Contact

Denys Oliinyk · <oliinyk@biochem.mpg.de> · Max Planck Institute of Biochemistry
