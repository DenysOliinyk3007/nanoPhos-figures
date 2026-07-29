"""Preprocessing for the SPEC / enrichment decoupling experiment (Reviewer 1 novelty).

Each condition (branch x input) has two Spectronaut reports in the folder:
  * ``{ts}_{branch}_{ng}ng_Report.tsv``         precursor-level (R.FileName, EG.PrecursorId,
                                                EG.ModifiedSequence, EG.ApexRT, EG.TotalQuantity,
                                                EG.PTMAssayProbability, PEP.StrippedSequence);
                                                no EG.IsDecoy (already FDR-filtered).
  * ``{ts}_{branch}_{ng}ng_Report_classI.tsv``  PTM Site Report (per-run SiteProbability + Quantity),
                                                already Class I filtered; no PTM.ProteinId.

Two entry points:

  * ``load_decoupling(folder)`` -> tidy per-run DataFrame (depth / precursors / selectivity).
  * ``regenerate_all(folder, out_dir)`` -> computes EVERY comparison table used by the
    decoupling figure notebook and writes them as CSVs to ``out_dir`` (the notebook then
    reads those CSVs by default, so plotting is instant and does not re-read the ~2.7 GB of
    raw reports; set REGENERATE=True in the notebook to rebuild them from the raw path).

Branches are normalised to: classic (nanoPhos_original), hybrid, wCleanup, woCleanup.
The Class I and precursor reports join on the identical .raw file stem (R.FileName);
the replicate is the plate well (last filename token, e.g. E8).
"""
from __future__ import annotations
import os
import re
import glob
import numpy as np
import pandas as pd
from core import count_sites_per_sample_ptm_report, calculate_dilution_linearity

# ---------------------------------------------------------------------------
# canonical ordering (single source of truth, imported by the notebook)
# ---------------------------------------------------------------------------
BRANCH_ORDER = ['classic', 'hybrid', 'wCleanup', 'woCleanup']
INPUT_ORDER = [10, 20, 50, 100, 200, 500, 1000]

# check 'wocleanup' before 'wcleanup' (substring safety); 'original' -> classic
_BRANCH_MAP = [('original', 'classic'), ('hybrid', 'hybrid'),
               ('wocleanup', 'woCleanup'), ('wcleanup', 'wCleanup')]

_FNAME_RE = re.compile(
    r'^(?P<ts>\d{8}_\d{6})_(?P<branch>.+?)_(?P<ng>\d+)ng_Report(?P<cls>_classI)?\.tsv$')

# Kyte-Doolittle hydropathy (for GRAVY of recovered phosphopeptides)
_KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
       'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
       'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}


def _canon_branch(token: str) -> str:
    t = token.lower()
    for sub, name in _BRANCH_MAP:
        if sub in t:
            return name
    return token


def _parse(fname: str):
    """(canonical_branch, input_ng, is_classI, timestamp) or None.

    ``timestamp`` (``YYYYMMDD_HHMMSS``) is lexicographically sortable and is used to
    disambiguate re-exports: if a condition is exported more than once (e.g. a malformed
    long-format report later re-exported wide-format), the newest timestamp wins.
    """
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    return (_canon_branch(m.group('branch')), int(m.group('ng')),
            bool(m.group('cls')), m.group('ts'))


def discover(folder: str):
    """Map every report .tsv in ``folder`` to (branch, input_ng).

    Returns ``(prec, cls)`` dicts, each ``{(branch, ng): path}``. On a collision keep
    the NEWEST timestamp: re-exports supersede earlier (possibly malformed) ones. glob
    order is unspecified, so arrival order must never decide which report is counted.
    """
    prec, cls = {}, {}
    prec_ts, cls_ts = {}, {}
    for path in glob.glob(os.path.join(folder, '*.tsv')):
        parsed = _parse(os.path.basename(path))
        if not parsed:
            continue
        branch, ng, is_cls, ts = parsed
        store, seen = (cls, cls_ts) if is_cls else (prec, prec_ts)
        key = (branch, ng)
        if key not in seen or ts > seen[key]:
            store[key] = path
            seen[key] = ts
    return prec, cls


def _clean_run_col(col: str, kind: str) -> str:
    """Strip the ``[n] <run>.raw.PTM.<kind>`` decoration down to the run stem."""
    return re.sub(r'^\[\d+\]\s+', '', col).replace(f'.raw.PTM.{kind}', '')


# ---------------------------------------------------------------------------
# core per-run table (depth / precursors / selectivity)
# ---------------------------------------------------------------------------
def _precursors_per_run(path: str):
    """Return (phospho_precursors, total_precursors) as per-run Series indexed by R.FileName.

    Phosphopeptide precursor = unique phosphorylated EG.PrecursorId (charge-distinct),
    localization-independent (Bekker-Jensen et al. 2020). Decoys dropped if the column exists.
    """
    keep = {'R.FileName', 'EG.ModifiedSequence', 'EG.PrecursorId', 'EG.IsDecoy'}
    df = pd.read_csv(path, sep='\t', usecols=lambda c: c in keep, low_memory=False)
    if 'EG.IsDecoy' in df.columns:
        df = df[~df['EG.IsDecoy'].astype(str).str.lower().isin(['true', '1'])]
    total = df.groupby('R.FileName')['EG.PrecursorId'].nunique()
    is_phos = df['EG.ModifiedSequence'].astype(str).str.contains('Phospho', case=False, na=False)
    phospho = df[is_phos].groupby('R.FileName')['EG.PrecursorId'].nunique()
    return phospho, total


def load_decoupling(folder: str) -> pd.DataFrame:
    """Discover paired reports in `folder` and return a tidy per-run DataFrame.

    Columns: branch, input_ng, run, well, classI_sites, phosphoprecursors,
             total_precursors, phospho_selectivity_pct.
    Class I is NaN for a run if no matching *_classI report is present.
    """
    prec, cls = discover(folder)

    rows = []
    for (branch, ng), ppath in prec.items():
        phospho, total = _precursors_per_run(ppath)
        csites = {}
        if (branch, ng) in cls:
            csites = count_sites_per_sample_ptm_report(
                pd.read_csv(cls[(branch, ng)], sep='\t', low_memory=False))
        for run in phospho.index:
            ci = csites.get(run)
            tot = int(total.get(run, 0))
            rows.append({
                'branch': branch,
                'input_ng': ng,
                'run': run,
                'well': run.split('_')[-1],
                'classI_sites': (int(ci) if ci is not None else np.nan),
                'phosphoprecursors': int(phospho[run]),
                'total_precursors': tot,
                'phospho_selectivity_pct': (round(100 * phospho[run] / tot, 1) if tot else np.nan),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['branch', 'input_ng', 'well']).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (branch, input) mean/SD/CV of Class I sites and phosphopeptide precursors."""
    def _agg(g):
        return pd.Series({
            'n': len(g),
            'classI_mean': g['classI_sites'].mean(),
            'classI_sd': g['classI_sites'].std(ddof=1),
            'prec_mean': g['phosphoprecursors'].mean(),
            'prec_sd': g['phosphoprecursors'].std(ddof=1),
            'selectivity_mean': g['phospho_selectivity_pct'].mean(),
        })
    return (df.groupby(['branch', 'input_ng']).apply(_agg, include_groups=False)
              .round(1).reset_index())


def contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise branch contrasts per input, each isolating one factor.

    classic/hybrid isolates the enrichment (SPEC held constant) - the direct R1 answer.
    Ratios for depth/precursors; selectivity as a percentage-point difference.
    """
    CTR = [('classic', 'hybrid', 'enrichment (SPEC held constant)'),
           ('hybrid', 'wCleanup', 'SPEC prep (enrichment held constant)'),
           ('wCleanup', 'woCleanup', 'cleanup effect (SDB-RPS / SpeedVac)'),
           ('classic', 'wCleanup', 'full nanoPhos vs standard conventional workflow')]
    m = df.groupby(['branch', 'input_ng'])[
        ['classI_sites', 'phosphoprecursors', 'phospho_selectivity_pct']].mean()
    rows = []
    for a, b, label in CTR:
        for ng in INPUT_ORDER:
            if (a, ng) in m.index and (b, ng) in m.index:
                rows.append({
                    'contrast': f'{a} / {b}', 'isolates': label, 'input_ng': ng,
                    'classI_ratio': round(m.loc[(a, ng), 'classI_sites'] / m.loc[(b, ng), 'classI_sites'], 3),
                    'prec_ratio': round(m.loc[(a, ng), 'phosphoprecursors'] / m.loc[(b, ng), 'phosphoprecursors'], 3),
                    'selectivity_diff_pp': round(m.loc[(a, ng), 'phospho_selectivity_pct'] - m.loc[(b, ng), 'phospho_selectivity_pct'], 1),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# quantification / identification quality metrics
# ---------------------------------------------------------------------------
def _classI_site_matrix(path: str):
    """Return a (site_key x run) Class I intensity matrix from one PTM Site Report.

    Keeps multiplicity (|M suffix). Cells masked to NaN where the paired
    PTM.SiteProbability < 0.75 (localization enforced independent of the export filter).
    Duplicate keys collapsed by max intensity per run.
    """
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df = df[df['PTM.ModificationTitle'].astype(str).str.contains('Phospho', case=False, na=False)]
    df = df[df['PTM.SiteAA'].astype(str).isin({'S', 'T', 'Y'})]
    gene = df['PG.Genes'].astype(str).str.split(';').str[0]
    key = (gene.values + '|' + df['PTM.SiteAA'].astype(str).values
           + df['PTM.SiteLocation'].astype(str).values
           + '|M' + df['PTM.Multiplicity'].astype(str).values)
    qcols = [c for c in df.columns if 'PTM.Quantity' in c]
    pby = {_clean_run_col(c, 'SiteProbability'): c for c in df.columns if 'PTM.SiteProbability' in c}
    Q = df[qcols].apply(lambda s: pd.to_numeric(s.replace('Filtered', np.nan), errors='coerce')).replace(0, np.nan)
    for c in qcols:
        p = pd.to_numeric(df[pby[_clean_run_col(c, 'Quantity')]].replace('Filtered', np.nan), errors='coerce')
        Q.loc[~(p >= 0.75), c] = np.nan
    Q.index = key
    Q.columns = [_clean_run_col(c, 'Quantity') for c in qcols]
    return Q.groupby(level=0).max()


def dilution_linearity(cls: dict):
    """Per-branch dilution linearity (Fig 2C method: log2 intensity ~ log2 ng).

    Keeps multiplicity (project policy: counts collapse, quant keeps), enforces 0.75.
    Returns (summary_df, sites_df); sites_df has per-site r_squared/slope for histograms.
    """
    summ, site_frames = [], []
    for b in BRANCH_ORDER:
        dd = {ng: pd.read_csv(cls[(b, ng)], sep='\t', low_memory=False)
              for ng in INPUT_ORDER if (b, ng) in cls}
        if len(dd) < 4:
            continue
        lin = calculate_dilution_linearity(dd, min_dilutions=4,
                                           collapse_multiplicity=False, enforce_cutoff=0.75)
        r2, sl = lin['r_squared'].dropna(), lin['slope'].dropna()
        summ.append({'branch': b, 'n_sites': len(lin),
                     'median_R2': round(r2.median(), 3),
                     'pct_R2_ge_0.8': round((r2 >= 0.8).mean() * 100, 1),
                     'pct_R2_ge_0.95': round((r2 >= 0.95).mean() * 100, 1),
                     'median_slope': round(sl.median(), 3),
                     'slope_q25': round(sl.quantile(.25), 3),
                     'slope_q75': round(sl.quantile(.75), 3)})
        lin = lin[['site_key', 'gene', 'n_dilutions', 'r_squared', 'slope']].copy()
        lin.insert(0, 'branch', b)
        site_frames.append(lin)
    return pd.DataFrame(summ), pd.concat(site_frames, ignore_index=True)


def completeness_cv(cls: dict):
    """Per (branch, input) data completeness and replicate CV of Class I intensities.

    completeness = mean fraction of replicates in which a detected site is quantified.
    CV = median linear-space CV across replicates for sites present in ALL replicates.
    """
    rows = []
    for b in BRANCH_ORDER:
        for ng in INPUT_ORDER:
            if (b, ng) not in cls:
                continue
            Q = _classI_site_matrix(cls[(b, ng)])
            nrep = Q.shape[1]
            filled = Q.notna().sum(axis=1)
            full = Q[filled == nrep]
            cv = (full.std(axis=1, ddof=1) / full.mean(axis=1)) * 100
            rows.append({'branch': b, 'input_ng': ng, 'n_rep': nrep, 'n_sites': len(Q),
                         'mean_completeness_pct': round((filled / nrep).mean() * 100, 1),
                         'pct_sites_in_all_reps': round((filled == nrep).mean() * 100, 1),
                         'n_full_sites': int(len(full)),
                         'median_CV_pct': round(cv.median(), 1)})
    return pd.DataFrame(rows)


def site_overlap(cls: dict, inputs=(100, 1000)):
    """Class I site-identity overlap between classic and each other branch, per input.

    Unique (gene, AA, pos) collapsed across replicates (multiplicity collapsed).
    """
    def _sites(b, ng):
        df = pd.read_csv(cls[(b, ng)], sep='\t', low_memory=False)
        df = df[df['PTM.ModificationTitle'].astype(str).str.contains('Phospho', case=False, na=False)]
        df = df[df['PTM.SiteAA'].astype(str).isin({'S', 'T', 'Y'})]
        g = df['PG.Genes'].astype(str).str.split(';').str[0]
        return set(g.values + '|' + df['PTM.SiteAA'].astype(str).values + df['PTM.SiteLocation'].astype(str).values)

    rows = []
    for ng in inputs:
        S = {b: _sites(b, ng) for b in BRANCH_ORDER if (b, ng) in cls}
        if 'classic' not in S:
            continue
        c = S['classic']
        others = set().union(*[S[b] for b in S if b != 'classic']) if len(S) > 1 else set()
        for b in BRANCH_ORDER:
            if b not in S:
                continue
            inter = len(c & S[b]) if b != 'classic' else len(c)
            rows.append({'input_ng': ng, 'branch': b, 'n_sites': len(S[b]),
                         'shared_with_classic': inter,
                         'branch_only_vs_classic': (len(S[b] - c) if b != 'classic' else 0),
                         'classic_covers_pct': (round(100 * len(c & S[b]) / len(S[b]), 1) if b != 'classic' else 100.0)})
        rows.append({'input_ng': ng, 'branch': 'classic_vs_union_others', 'n_sites': len(c),
                     'shared_with_classic': len(c & others),
                     'branch_only_vs_classic': len(c - others),
                     'classic_covers_pct': (round(100 * len(c & others) / len(others), 1) if others else np.nan)})
    return pd.DataFrame(rows)


def _gravy(seq: str):
    s = ''.join(ch for ch in str(seq).upper() if ch in _KD)
    return float(np.mean([_KD[c] for c in s])) if s else np.nan


def phospho_signal_rt_quality(prec: dict, rt_bins=None):
    """Per (branch, input) phosphopeptide-precursor signal, RT profile and quality.

    Returns (signal_df, rt_df):
      signal_df: total & median phospho-precursor intensity, dynamic range (log10 p5-p95),
                 localization (median EG.PTMAssayProbability, %>=0.9), GRAVY (median,
                 %hydrophobic) per (branch, input).
      rt_df:     summed phospho intensity per 0.5-min EG.ApexRT bin (mean over replicates).
    """
    if rt_bins is None:
        rt_bins = np.arange(4, 16.01, 0.5)
    sig_rows, rt_rows = [], []
    cols = ['R.FileName', 'EG.ModifiedSequence', 'PEP.StrippedSequence',
            'EG.ApexRT', 'EG.PTMAssayProbability', 'EG.TotalQuantity (Settings)']
    for b in BRANCH_ORDER:
        for ng in INPUT_ORDER:
            if (b, ng) not in prec:
                continue
            d = pd.read_csv(prec[(b, ng)], sep='\t',
                            usecols=lambda c: c in set(cols), low_memory=False)
            d = d[d['EG.ModifiedSequence'].astype(str).str.contains('Phospho', case=False, na=False)]
            rt = pd.to_numeric(d['EG.ApexRT'], errors='coerce')
            q = pd.to_numeric(d['EG.TotalQuantity (Settings)'], errors='coerce').replace(0, np.nan)
            loc = pd.to_numeric(d['EG.PTMAssayProbability'], errors='coerce')
            d = d.assign(rt=rt, q=q)
            g = d.dropna(subset=['q']).groupby('R.FileName')['q']
            qpos = q.replace(0, np.nan).dropna()
            lg = np.log10(qpos)
            useq = d['PEP.StrippedSequence'].dropna().unique()
            gv = pd.Series([_gravy(s) for s in useq]).dropna()
            sig_rows.append({
                'branch': b, 'input_ng': ng,
                'phospho_sum_mean': g.sum().mean(), 'phospho_sum_sd': g.sum().std(ddof=1),
                'phospho_median_int_mean': g.median().mean(),
                'dynrange_log10_p5_95': round(lg.quantile(.95) - lg.quantile(.05), 3),
                'loc_median': round(loc.median(), 3),
                'loc_pct_ge90': round((loc >= 0.90).mean() * 100, 1),
                'gravy_median': round(gv.median(), 3),
                'gravy_pct_hydrophobic': round((gv > 0).mean() * 100, 1),
            })
            dd = d.dropna(subset=['rt', 'q']).copy()
            dd['bin'] = pd.cut(dd['rt'], rt_bins, labels=rt_bins[:-1] + 0.25)
            prof = (dd.groupby(['R.FileName', 'bin'], observed=True)['q'].sum()
                      .reset_index().groupby('bin', observed=True)['q'].mean())
            for bc, val in prof.items():
                rt_rows.append({'branch': b, 'input_ng': ng, 'rt_bin': float(bc), 'phospho_sum_mean': val})
    return pd.DataFrame(sig_rows), pd.DataFrame(rt_rows)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def regenerate_all(folder: str, out_dir: str) -> dict:
    """Compute every decoupling comparison table from the raw reports and write CSVs.

    Reads the ~2.7 GB of precursor + Class I reports in ``folder`` once, writes the
    compact result tables to ``out_dir`` (committed under data/), and returns them in a
    dict. The notebook calls this only when REGENERATE=True; otherwise it reads the CSVs.
    """
    os.makedirs(out_dir, exist_ok=True)
    prec, cls = discover(folder)

    perrun = load_decoupling(folder)
    summ = summarize(perrun)
    contr = contrasts(perrun)
    lin_summary, lin_sites = dilution_linearity(cls)
    comp_cv = completeness_cv(cls)
    overlap = site_overlap(cls)
    signal, rt_profile = phospho_signal_rt_quality(prec)

    tables = {
        'decoupling_perrun': perrun,
        'decoupling_summary': summ,
        'decoupling_contrasts': contr,
        'decoupling_linearity_summary': lin_summary,
        'decoupling_linearity_sites': lin_sites,
        'decoupling_completeness_cv': comp_cv,
        'decoupling_overlap': overlap,
        'decoupling_signal': signal,
        'decoupling_rt_profile': rt_profile,
    }
    for name, tbl in tables.items():
        tbl.to_csv(os.path.join(out_dir, f'{name}.csv'), index=False)
    return tables


def load_tables(out_dir: str) -> dict:
    """Read the committed decoupling result CSVs from ``out_dir`` (REGENERATE=False path)."""
    names = ['decoupling_perrun', 'decoupling_summary', 'decoupling_contrasts',
             'decoupling_linearity_summary', 'decoupling_linearity_sites',
             'decoupling_completeness_cv', 'decoupling_overlap',
             'decoupling_signal', 'decoupling_rt_profile']
    return {n: pd.read_csv(os.path.join(out_dir, f'{n}.csv')) for n in names}
