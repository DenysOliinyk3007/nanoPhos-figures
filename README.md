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

Drop Spectronaut `_Report.tsv` exports into `raw_data/` (gitignored), then open a
figure notebook from the repo root and **Restart → Run All**:

```bash
jupyter lab nanoPhos_Figure2_v01.ipynb
```

### Notebooks

- **`nanoPhos_Figure{2–5}_v01.ipynb`**, **`nanoPhos_Suppl_Figure{1–4}_v01.ipynb`** —
  the revised analysis, with strict per-run Class I localization filtering
  (probability ≥ 0.75), used for the *Nature Communications* revision. These are the
  notebooks that reproduce the published figures.
- **`nanoPhos_Figure3_DIANN_v00.ipynb`** — DIA-NN cross-search comparison.

The original-submission (`v00`) notebooks are retained locally under `archive/`
(not version-controlled).

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Production Python modules: `PeptideCollapse_v{3,4}`, helpers, analytics |
| `docs/` | `REPO_STRUCTURE.md`, changelogs, manuscript context |
| `data/` | Small reference CSVs cited by individual figures |
| `raw_data/` | Spectronaut DIA exports (gitignored, large) |
| `archive/` | Older versions, exploratory notebooks, deprecated work (gitignored) |
| `nanoPhos_Figure{N}_v01.ipynb` | One notebook per manuscript figure (root level), revised analysis |
| `nanoPhos_Suppl_Figure{N}_v01.ipynb` | Supplementary figure notebooks |

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
The first cell of each notebook prepends `src/` to `sys.path`, so the bare module
imports below it resolve to the production code under `src/`.

## Citation

Manuscript in revision. Citation will be added once accepted.

## Contact

Denys Oliinyk · <oliinyk@biochem.mpg.de> · Max Planck Institute of Biochemistry
