"""Per-run Class I phosphosite counting from Spectronaut PTM Site Reports.

This implements the per-(site, run) localization convention requested by
Reviewer 2: a phosphosite is counted as a localized (Class I) identification in
a given run only if its localization probability *in that run* meets the cutoff
(default 0.75) and it is quantified in that run. Sites are collapsed to the
unique (protein, residue, position) identity, which removes the multiplicity /
ambiguous-localization double-counting the reviewer noted.

The routine reads the wide-format PTM Site Report (one
``[n] <run>.raw.PTM.SiteProbability`` column and one matching
``...PTM.Quantity`` column per run). The ``"Filtered"`` token is treated as
missing.

It is independent of Spectronaut's normalization setting: counts depend only on
the per-run site-probability and on whether a run has a quantity, not on the
quantity value. So the same routine applies to normalization-on and
normalization-off exports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

CLASS_I_CUTOFF = 0.75
_MISSING = "Filtered"
_PROB_SUFFIX = ".PTM.SiteProbability"
_QUANT_SUFFIX = ".PTM.Quantity"
# "[12] 20250729_..._withEGF_500ng_02.raw.PTM.SiteProbability" -> run name
_RUN_RE = re.compile(r"^\[\d+\]\s*(.*?)(?:\.raw)?" + re.escape(_PROB_SUFFIX) + r"$")
# pull condition + input amount + replicate out of a run name
_COND_RE = re.compile(r"(withEGF|woEGF|noEGF)", re.IGNORECASE)
_AMT_RE = re.compile(r"_(\d+)ng")
_REP_RE = re.compile(r"_(\d+)\.raw|_(\d+)$")


@dataclass
class RunColumns:
    run: str
    prob_col: str
    quant_col: str
    condition: str | None
    input_ng: int | None
    replicate: str | None


def _parse_run_label(run: str) -> tuple[str | None, int | None, str | None]:
    cond = _COND_RE.search(run)
    amt = _AMT_RE.search(run)
    rep = _REP_RE.search(run)
    condition = cond.group(1).lower() if cond else None
    input_ng = int(amt.group(1)) if amt else None
    replicate = None
    if rep:
        replicate = rep.group(1) or rep.group(2)
    return condition, input_ng, replicate


def discover_run_columns(columns: list[str]) -> list[RunColumns]:
    """Pair every per-run SiteProbability column with its Quantity column."""
    prob_by_run: dict[str, str] = {}
    quant_by_run: dict[str, str] = {}
    for col in columns:
        if col.endswith(_PROB_SUFFIX):
            m = _RUN_RE.match(col)
            if m:
                prob_by_run[m.group(1)] = col
        elif col.endswith(_QUANT_SUFFIX):
            # mirror the prob regex but for the quant suffix
            base = col[: -len(_QUANT_SUFFIX)]
            base = re.sub(r"^\[\d+\]\s*", "", base)
            base = re.sub(r"\.raw$", "", base)
            quant_by_run[base] = col

    runs: list[RunColumns] = []
    for run, prob_col in prob_by_run.items():
        quant_col = quant_by_run.get(run)
        if quant_col is None:
            continue
        condition, input_ng, replicate = _parse_run_label(run)
        runs.append(RunColumns(run, prob_col, quant_col, condition, input_ng, replicate))
    return runs


def _site_id(df: pd.DataFrame) -> pd.Series:
    """Unique (protein, residue, position) identity, collapsing multiplicity."""
    return (
        df["PTM.ProteinId"].astype(str)
        + "_"
        + df["PTM.SiteAA"].astype(str)
        + df["PTM.SiteLocation"].astype(str)
    )


def load_site_report(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["_site_id"] = _site_id(df)
    return df


def per_run_class_I(
    df: pd.DataFrame, runs: list[RunColumns], cutoff: float = CLASS_I_CUTOFF
) -> pd.DataFrame:
    """Per-run counts: localized (prob>=cutoff & quantified) and total quantified.

    Returns one row per run with the unique-site Class I count, the unique-site
    'all quantified' count (no localization filter — the pre-revision metric),
    and the Class I fraction.
    """
    rows = []
    for rc in runs:
        prob = pd.to_numeric(df[rc.prob_col].replace(_MISSING, np.nan), errors="coerce")
        quant = pd.to_numeric(df[rc.quant_col].replace(_MISSING, np.nan), errors="coerce")
        quantified = quant.notna()
        localized = quantified & (prob >= cutoff)
        # collapse to unique site: a site counts if ANY of its rows qualifies
        sid = df["_site_id"]
        n_quant = sid[quantified].nunique()
        n_class1 = sid[localized].nunique()
        rows.append(
            {
                "run": rc.run,
                "condition": rc.condition,
                "input_ng": rc.input_ng,
                "replicate": rc.replicate,
                "class_I_sites": n_class1,
                "quantified_sites": n_quant,
                "class_I_fraction": (n_class1 / n_quant) if n_quant else np.nan,
            }
        )
    return pd.DataFrame(rows)


def dataset_class_I_union(
    df: pd.DataFrame, runs: list[RunColumns], cutoff: float = CLASS_I_CUTOFF
) -> dict:
    """Dataset-level site universe: unique sites localized in >=1 run."""
    localized_any = pd.Series(False, index=df.index)
    quantified_any = pd.Series(False, index=df.index)
    for rc in runs:
        prob = pd.to_numeric(df[rc.prob_col].replace(_MISSING, np.nan), errors="coerce")
        quant = pd.to_numeric(df[rc.quant_col].replace(_MISSING, np.nan), errors="coerce")
        q = quant.notna()
        quantified_any |= q
        localized_any |= q & (prob >= cutoff)
    sid = df["_site_id"]
    return {
        "class_I_union": int(sid[localized_any].nunique()),
        "quantified_union": int(sid[quantified_any].nunique()),
    }


def summarize_by_condition(per_run: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- SD Class I count across replicates, per condition x input."""
    grp = per_run.groupby(["condition", "input_ng"], dropna=False)["class_I_sites"]
    out = grp.agg(["mean", "std", "count"]).reset_index()
    return out.sort_values(["condition", "input_ng"]).reset_index(drop=True)
