def filter_phosphosites(dataset, how = 'all', cutoff = 0.7, condition_col = None):
    ###
    import pandas as pd
    import numpy as np
    ###

    df = dataset.copy()
    if not any('~' in str(col) for col in df.columns):
        raise ValueError('Dataset must be in the following format: Phosphosite as columns, samples as rows.')
        
    
    if (how == 'condition') and (condition_col == None):
        raise ValueError('Please provide the name of condition if you are using condition-based filtering')
    
    meta_cols = [col for col in df.columns if '~' not in col]
    num_cols = [col for col in df.columns if '~' in col]

    if how == 'all':
        df_tmp = df[num_cols].loc[:, (1 - (df[num_cols].isna().sum() / len(df[num_cols]))) >=cutoff]
        df_filtered = pd.concat([df_tmp, df[meta_cols]], axis = 1)
        return df_filtered
    
    elif how == 'condition':
        if not any(condition_col in str(col) for col in df.columns):
            raise ValueError('Condition column is not present in the dataframe.')
        
        presence_dict = {}
        for condition, group in df.groupby(condition_col):
            presence = group[num_cols].notna().sum() / len(group)
            presence_dict[condition] = presence

        presence_df = pd.DataFrame(presence_dict)
        sites_to_keep = presence_df.max(axis = 1) >= cutoff
        cols_to_keep = sites_to_keep[sites_to_keep].index.tolist()

        df_filtered = df[cols_to_keep + meta_cols]

        return df_filtered
    
    else:
        raise ValueError('Please select valid filtering convention: "all" or "condition".')
    




def impute_phosphosites(dataset, n_neighbors=None, grouping_col = None, weights='distance'):
    ###
    from sklearn.impute import KNNImputer
    import pandas as pd
    import numpy as np
    ###
    df = dataset.copy()

    if df.empty:
        raise ValueError('Dataset is empty')
    
    if not df.isna().any().any():
        print('Warning: Dataset does not contain any missing values. Returning original dataset.')
        return df
    
    if grouping_col is not None and grouping_col not in df.columns:
        raise ValueError(f'Grouping column "{grouping_col}" not found in dataset')
    
    meta_cols = [col for col in df.columns if '~' not in col]
    num_cols = [col for col in df.columns if '~' in col]
    
    if len(num_cols) == 0:
        raise ValueError("No phosphosite columns found (should contain '~')")
    
    original_index = df.index

    if grouping_col is None:
        if n_neighbors is not None:
            imputer = KNNImputer(n_neighbors = n_neighbors)
        else:
            imputer = KNNImputer(n_neighbors= int(np.sqrt(len(df))))

        all_nan_columns = df.columns[df.isna().all()]

        if len (all_nan_columns) > 0:
            print(
                f"Warning: The following columns have all NaN values and will be dropped: {all_nan_columns}"
            )
            df = df.drop(all_nan_columns, axis=1)

        meta_cols = [col for col in df.columns if '~' not in col]
        num_cols = [col for col in df.columns if '~' in col]
        original_index = df.index

        df_transposed = df[num_cols].T

        df_transposed = pd.DataFrame(imputer.fit_transform(df_transposed), columns = df_transposed.columns.astype(str), index = df_transposed.index)

        df_transposed = df_transposed.T
        df_transposed.index = original_index
    
    else:
        groups = df.groupby(grouping_col, dropna = True)
        df_transposed = df[num_cols].copy()
        for group_name, group_indices in groups.groups.items():
            n_group = len(group_indices)

            if n_neighbors is None:
                n_neighbors_group = int(np.sqrt(n_group))
        
            else:
                n_neighbors_group = n_neighbors
            
            if n_neighbors_group >= n_group:
                n_neighbors_group = max(1, n_group - 1)
                print(f"  Warning: {group_name} - adjusted n_neighbors to {n_neighbors_group} (group size: {n_group})")


            group_data = df.loc[group_indices, num_cols]

            group_T = group_data.T
            imputer = KNNImputer(n_neighbors=n_neighbors_group, weights=weights)
            group_imputed_T = pd.DataFrame(
                imputer.fit_transform(group_T),
                columns=group_T.columns,
                index=group_T.index
            )
            group_imputed = group_imputed_T.T
            df_transposed.loc[group_indices, :] = group_imputed.values
    
    n_missing_after = df_transposed.isna().sum().sum()
    
    if n_missing_after > 0:
        print(f"Warning: {n_missing_after} values still missing after imputation")

    df_imputed = pd.concat([df_transposed, df[meta_cols]], axis = 1)

    return df_imputed





def batch_correction (dataset, correction_col = None):
    ###
    from combat.pycombat import pycombat
    import pandas as pd
    ###

    if correction_col == None:
        raise ValueError('Please provide a column for batch correction')
    
    if correction_col not in dataset.columns:
        raise ValueError('Batch correction column is not found')
    
    df = dataset.copy()

    meta_cols = [col for col in df.columns if '~' not in col]
    num_cols = [col for col in df.columns if '~' in col]

    df_transposed = df[num_cols].T
    df_corrected = pycombat(df_transposed, df[correction_col].tolist())

    df_corrected = df_corrected.T

    df_corrected = pd.concat([df_corrected, df[meta_cols]], axis = 1)

    return df_corrected





def filter_samples(data, threshold_MAD = 3, drop_cols = ['UPD_seq', 'PTM_localization', 'Protein_group', 'Gene_group','PTM_Collapse_key', 'kinase_sequence']):
    ###
    from scipy import stats
    import numpy as np
    import pandas as pd
    ###

    data_copy = data.copy()
    if drop_cols is not None and any(col in data_copy.columns for col in drop_cols):
        meta = data_copy[drop_cols] 
        data_copy.drop(columns=drop_cols)
    else:
        meta = pd.DataFrame()
    nums = []
    for col in data_copy.columns:
        nums.append(len(data_copy[col].dropna()))
    
    median = np.median(nums)
    mad = stats.median_abs_deviation(nums)
    threshold = median - threshold_MAD*mad

    cols_to_keep = []
    for col in data_copy.columns:
        num = len(data_copy[col].dropna())
        if num >= threshold:
            cols_to_keep.append(col)
    
    data_copy = data_copy[cols_to_keep]

    return data_copy, threshold


def create_confusion_matrix(dataset1, dataset2, suffixes = ('_MaMu', '_top3')):
    ###
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    data1_copy = dataset1.copy()
    data2_dopy = dataset2.copy()
    ###

    merged_df = data1_copy.merge(data2_dopy, how = 'inner', on = 'Run', suffixes = suffixes)

    labels_data1 = merged_df.iloc[:,1]
    labels_data2 = merged_df.iloc[:,2]

    cm = confusion_matrix(labels_data1, labels_data2)
    labels = sorted(labels_data1.unique())
    cm_df = pd.DataFrame(cm, 
                     index=labels,
                     columns=labels)

    return cm_df



def calculate_subsets(df, col1, col2, col3, key_col='PTM_Collapse_key'):

    set1 = set(df[df[col1].notna()][key_col])
    set2 = set(df[df[col2].notna()][key_col])
    set3 = set(df[df[col3].notna()][key_col])

    only_1 = len(set1 - set2 - set3)
    only_2 = len(set2 - set1 - set3)
    only_1_and_2 = len((set1 & set2) - set3)
    only_3 = len(set3 - set1 - set2)
    only_1_and_3 = len((set1 & set3) - set2)
    only_2_and_3 = len((set2 & set3) - set1)
    all_three = len(set1 & set2 & set3)

    return (only_1, only_2, only_1_and_2, only_3, only_1_and_3, only_2_and_3, all_three)




def _hex_to_rgba(hex_color, alpha):
    """Convert '#RRGGBB' + opacity to a CSS rgba() string. Helper for plot styling."""
    h = hex_color.lstrip('#')
    return f'rgba({int(h[0:2],16)}, {int(h[2:4],16)}, {int(h[4:6],16)}, {alpha})'


def count_sites_per_sample_ptm_report(df, dedupe=True, sty_only=True,
                                       phospho_only=True, treat_zero_as_missing=True,
                                       collapse_multiplicity=True, enforce_cutoff=0.75):
    """Per-sample strict Class I phosphosite count for a Spectronaut PTM Site Report.

    Self-defending counter — applies all corrections internally so the
    output is bulletproof regardless of upstream filtering state:
      - phospho_only           drop rows where PTM.ModificationTitle is not phospho
                               (catches Carbamidomethyl, Oxidation, etc.)
      - sty_only               drop rows on amino acids other than S/T/Y
      - dedupe                 count unique site keys per sample, so the same site
                               under multiple protein IDs is counted once. The key
                               base is the gene symbol, falling back to PTM.ProteinId
                               when the gene is blank (so distinct genes-less proteins
                               are not merged into one site).
      - collapse_multiplicity  count by unique (gene, AA, pos), so a residue seen on
                               a singly- vs multiply-phosphorylated peptide (M1 vs
                               M2) is ONE site, not several. This is the convention
                               for reporting localized phosphosites (Olsen 2006,
                               Bekker-Jensen 2020, uPhos) and what Reviewer 2 asks
                               for. Set False to count multiplicity-resolved features
                               (quantitative entities, e.g. for QC) — but that is not
                               a "unique phosphosite" count and inflates the depth
                               metric by ~10%.
      - enforce_cutoff         if not None (default 0.75), localization is verified
                               HERE against the paired PTM.SiteProbability column:
                               a cell counts only if its own per-(site, run) prob
                               >= cutoff. This makes the count independent of whether
                               the export applied the localization filter. Raises if
                               the report carries no SiteProbability columns to check
                               against. Pass None to fall back to trusting the
                               export's 'Filtered' masking.
      - treat_zero_as_missing  PTM.Quantity == 0 cells are NaN-equivalent

    A cell is "Class I in this run" when, after filters, its PTM.Quantity value is
    numeric, non-NaN, (optionally) non-zero, and its paired SiteProbability meets
    `enforce_cutoff`. Probability and quantity columns are paired by run name (not
    column order), so the two blocks need not be aligned.

    Returns
    -------
    dict[clean_sample_name -> int]
    """
    ###
    import re
    import numpy as np
    import pandas as pd
    ###

    df_used = df

    if phospho_only and 'PTM.ModificationTitle' in df_used.columns:
        is_ph = df_used['PTM.ModificationTitle'].astype(str).str.contains(
            'Phospho', case=False, na=False
        )
        df_used = df_used[is_ph]

    if sty_only and 'PTM.SiteAA' in df_used.columns:
        df_used = df_used[df_used['PTM.SiteAA'].astype(str).isin({'S', 'T', 'Y'})]

    def _clean(col, kind):
        return re.sub(r'^\[\d+\]\s+', '', col).replace(f'.raw.PTM.{kind}', '')

    quant_cols = [c for c in df_used.columns if 'PTM.Quantity' in c]
    prob_cols  = [c for c in df_used.columns if 'PTM.SiteProbability' in c]
    prob_by_run = {_clean(c, 'SiteProbability'): c for c in prob_cols}

    if enforce_cutoff is not None and not prob_cols:
        raise ValueError(
            "enforce_cutoff is set but the report has no PTM.SiteProbability "
            "columns to verify localization against. Pass enforce_cutoff=None only "
            "if you deliberately trust the export's own localization filter."
        )

    def _valid_mask(col):
        """Boolean Series: rows quantified (and, if enforced, localized) in `col`."""
        c_num = pd.to_numeric(df_used[col].replace('Filtered', np.nan), errors='coerce')
        valid = c_num.notna()
        if treat_zero_as_missing:
            valid &= c_num != 0
        if enforce_cutoff is not None:
            run = _clean(col, 'Quantity')
            prob_col = prob_by_run.get(run)
            if prob_col is None:
                raise ValueError(
                    f"enforce_cutoff is set but no PTM.SiteProbability column is "
                    f"paired with quantity column '{col}' (run '{run}')."
                )
            p_num = pd.to_numeric(df_used[prob_col].replace('Filtered', np.nan),
                                  errors='coerce')
            valid &= (p_num >= enforce_cutoff)
        return valid

    counts = {}
    base_cols = {'PG.Genes', 'PTM.SiteAA', 'PTM.SiteLocation'}
    needs_mult = base_cols if collapse_multiplicity else base_cols | {'PTM.Multiplicity'}
    if dedupe and needs_mult.issubset(df_used.columns):
        gene_first = df_used['PG.Genes'].astype(str).str.split(';').str[0]
        # fall back to ProteinId where the gene symbol is blank, so distinct
        # gene-less proteins are not merged into a single (gene, AA, pos) key
        if 'PTM.ProteinId' in df_used.columns:
            prot = df_used['PTM.ProteinId'].astype(str).str.split(';').str[0]
            blank = gene_first.isin({'', 'nan', 'None', 'NaN'})
            gene_first = gene_first.where(~blank, prot)
        site_key = (gene_first + '|' +
                    df_used['PTM.SiteAA'].astype(str) +
                    df_used['PTM.SiteLocation'].astype(str))
        if not collapse_multiplicity:
            site_key = site_key + '|M' + df_used['PTM.Multiplicity'].astype(str)
        for col in quant_cols:
            counts[_clean(col, 'Quantity')] = int(site_key[_valid_mask(col)].nunique())
    else:
        for col in quant_cols:
            counts[_clean(col, 'Quantity')] = int(_valid_mask(col).sum())
    return counts


def sample_condition(col):
    """Classify a Spectronaut sample/column name by EGF condition.

    Returns 'withegf', 'woegf', or 'other'. Use this to split counts by the
    column's own name rather than the source filename — the revision
    ``withEGF_repeat`` reports carry both withEGF and woEGF columns in one file,
    so filename-based bucketing mixes conditions. Case-insensitive.
    """
    c = str(col).lower()
    if 'withegf' in c:
        return 'withegf'
    if 'woegf' in c or 'noegf' in c:
        return 'woegf'
    return 'other'


def class_I_by_condition(file_dict, condition, collapse_multiplicity=True,
                         enforce_cutoff=0.75):
    """Per-input list of per-run Class I counts for one EGF condition.

    Parameters
    ----------
    file_dict : dict[input_ng -> PTM Site Report DataFrame]
    condition : 'withegf' | 'woegf'
        Only columns whose sample name matches this condition are kept
        (via `sample_condition`), so files mixing EGF+/EGF- are handled safely.
    collapse_multiplicity, enforce_cutoff
        Passed through to `count_sites_per_sample_ptm_report`.

    Fails loud rather than silent: raises on an invalid `condition`, and warns
    if any column cannot be classified or if an input contributes no matching
    columns (so a silently missing input level is impossible to overlook).

    Returns
    -------
    dict[input_ng -> list[int]]  per-run Class I site counts for `condition`.
    """
    ###
    import warnings
    ###

    valid_conditions = {'withegf', 'woegf'}
    if condition not in valid_conditions:
        raise ValueError(
            f"condition must be one of {sorted(valid_conditions)}, got {condition!r}."
        )

    out = {}
    for ng, df in file_dict.items():
        counts = count_sites_per_sample_ptm_report(
            df, collapse_multiplicity=collapse_multiplicity,
            enforce_cutoff=enforce_cutoff,
        )
        matched = {s: n for s, n in counts.items() if sample_condition(s) == condition}
        unclassified = [s for s in counts if sample_condition(s) == 'other']
        if unclassified:
            warnings.warn(
                f"input {ng}: {len(unclassified)} column(s) not classifiable as "
                f"withEGF/woEGF and excluded: {unclassified[:3]}"
            )
        if not matched:
            warnings.warn(
                f"input {ng}: no columns matched condition '{condition}'; "
                f"this input is omitted from the result."
            )
            continue
        out[ng] = list(matched.values())
    return out


def plot_ids_box(data_dict, labels=None,
                 box_color='#8A0000', box_fill_alpha=0.15,
                 point_color='#393E46', point_size=9, point_jitter=0.3,
                 plot_width=600, plot_height=600, plot_template='plotly_white',
                 xaxis_title='Protein input', yaxis_title='Phosphosites',
                 dedupe=True, sty_only=True, phospho_only=True,
                 treat_zero_as_missing=True):
    """Boxplot + jittered individual points of per-sample Class I site counts.

    Styling defaults match alphaPhosHelperFunctions.py conventions:
      box_color=#8A0000 (Class I deep red), point_color=#393E46 (near-black),
      point_size=9, plot_template='plotly_white', 600 x 600.

    Counts come from `count_sites_per_sample_ptm_report` with all defensive
    filters on by default — see that function's docstring for what each does.

    Parameters
    ----------
    data_dict : dict[label, pd.DataFrame]
        Each value is a Spectronaut PTM Site Report (raw or pre-filtered).
    labels : list, optional
        Order of x-axis groups. Default: sorted keys of data_dict.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    ###
    import plotly.graph_objects as go
    ###

    if labels is None:
        labels = sorted(data_dict.keys())

    group_counts = {
        lab: list(count_sites_per_sample_ptm_report(
            data_dict[lab],
            dedupe=dedupe, sty_only=sty_only, phospho_only=phospho_only,
            treat_zero_as_missing=treat_zero_as_missing,
        ).values())
        for lab in labels
    }

    fill = _hex_to_rgba(box_color, box_fill_alpha)
    fig = go.Figure()
    for lab in labels:
        ys = group_counts[lab]
        x_lab = f"{lab} ng" if isinstance(lab, (int, float)) else str(lab)
        fig.add_trace(go.Box(
            y=ys, x=[x_lab] * len(ys), name=x_lab,
            boxpoints='all', jitter=point_jitter, pointpos=0,
            marker=dict(size=point_size, color=point_color,
                        line=dict(width=0.5, color='black')),
            line=dict(color=box_color, width=1.5),
            fillcolor=fill,
            showlegend=False,
        ))

    fig.update_layout(
        template=plot_template,
        width=plot_width, height=plot_height,
        xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        showlegend=False,
    )
    fig.update_yaxes(rangemode='tozero')
    return fig


def calculate_dilution_linearity(data_dict, dilution_values=None,
                                  log2_y=True, log2_x=True,
                                  min_dilutions=4,
                                  dedupe=True, sty_only=True,
                                  phospho_only=True,
                                  treat_zero_as_missing=True,
                                  collapse_multiplicity=False,
                                  enforce_cutoff=0.75):
    """Per-site linear-response analysis across a Spectronaut PTM-Site-Report
    dilution series.

    For each phosphosite, fits log2(mean_intensity) ~ log2(dilution_ng) (defaults)
    using sites observed at >= ``min_dilutions`` dilution levels, and reports
    R^2 + slope + intercept. A well-behaved site has slope ~ 1 (linear scaling
    with input) and R^2 close to 1.

    Defensive cleaning matches ``count_sites_per_sample_ptm_report``: phospho_only,
    sty_only, dedupe, treat_zero_as_missing, plus per-(site, run) localization
    enforcement and the blank-gene ProteinId fallback.

    Parameters
    ----------
    data_dict : dict[float, pd.DataFrame]
        Keys are dilution amounts (e.g. ng input), values are raw Spectronaut
        PTM Site Reports.
    dilution_values : list of float, optional
        Override the keys of ``data_dict`` as the x-axis values. Default uses
        the dict keys directly.
    log2_y, log2_x : bool
        Apply log2 to intensity / dilution before fitting. Default True/True
        (log-log) gives the classical dilution-curve slope-1 interpretation.
    min_dilutions : int
        Minimum number of dilution levels at which a site must be observed
        (after filters) to be included.
    collapse_multiplicity : bool
        Default **False** (keep): M1 vs M2 of a residue are kept as separate
        quantitative curves — this is a per-feature linearity assessment, and
        per project policy linearity keeps multiplicity (counts collapse, quant
        keeps). Set True to fit one curve per unique (gene, AA, pos), collapsing
        forms by taking the max intensity per dilution.
    enforce_cutoff : float or None
        Default 0.75. Localization is verified against the paired
        PTM.SiteProbability column (cells below cutoff are dropped before the
        per-site mean), so linearity is computed on Class I quantities regardless
        of the export's filter. Raises if no probability columns are present.
        Pass None to trust the export's masking.

    Returns
    -------
    pd.DataFrame with columns:
      - site_key, gene, n_dilutions, r_squared, slope, intercept,
        mean_log2_intensity
    """
    ###
    import re
    import numpy as np
    import pandas as pd
    from scipy import stats as sst
    ###

    def _clean(col, kind):
        return re.sub(r'^\[\d+\]\s+', '', col).replace(f'.raw.PTM.{kind}', '')

    dilutions = sorted(data_dict.keys())
    if dilution_values is None:
        dilution_values = dilutions
    dilution_lookup = dict(zip(dilutions, dilution_values))

    per_dilution = {}     # ng -> {site_key: mean intensity across replicates (raw)}
    site_meta = {}        # site_key -> first observed gene

    base_cols = {'PG.Genes', 'PTM.SiteAA', 'PTM.SiteLocation'}
    needs = base_cols if collapse_multiplicity else base_cols | {'PTM.Multiplicity'}

    for d in dilutions:
        df = data_dict[d]

        if phospho_only and 'PTM.ModificationTitle' in df.columns:
            df = df[df['PTM.ModificationTitle'].astype(str).str.contains(
                'Phospho', case=False, na=False
            )]
        if sty_only and 'PTM.SiteAA' in df.columns:
            df = df[df['PTM.SiteAA'].astype(str).isin({'S', 'T', 'Y'})]

        if not needs.issubset(df.columns):
            raise KeyError(f"Required columns missing in dilution {d} dataframe.")

        gene_first = df['PG.Genes'].astype(str).str.split(';').str[0]
        # fall back to ProteinId where the gene symbol is blank
        if 'PTM.ProteinId' in df.columns:
            prot = df['PTM.ProteinId'].astype(str).str.split(';').str[0]
            blank = gene_first.isin({'', 'nan', 'None', 'NaN'})
            gene_first = gene_first.where(~blank, prot)

        site_keys = (gene_first.values + '|' +
                     df['PTM.SiteAA'].astype(str).values +
                     df['PTM.SiteLocation'].astype(str).values)
        if not collapse_multiplicity:
            site_keys = site_keys + '|M' + df['PTM.Multiplicity'].astype(str).values

        quant_cols = [c for c in df.columns if 'PTM.Quantity' in c]
        prob_cols  = [c for c in df.columns if 'PTM.SiteProbability' in c]
        prob_by_run = {_clean(c, 'SiteProbability'): c for c in prob_cols}
        if enforce_cutoff is not None and not prob_cols:
            raise ValueError(
                f"enforce_cutoff is set but dilution {d} report has no "
                f"PTM.SiteProbability columns. Pass enforce_cutoff=None to trust "
                f"the export's localization filter."
            )

        q = df[quant_cols].apply(lambda s: pd.to_numeric(
            s.replace('Filtered', np.nan), errors='coerce'))
        if treat_zero_as_missing:
            q = q.replace(0, np.nan)
        if enforce_cutoff is not None:
            for col in quant_cols:
                prob_col = prob_by_run.get(_clean(col, 'Quantity'))
                if prob_col is None:
                    raise ValueError(
                        f"dilution {d}: no PTM.SiteProbability paired with '{col}'."
                    )
                p = pd.to_numeric(df[prob_col].replace('Filtered', np.nan),
                                  errors='coerce')
                q.loc[~(p >= enforce_cutoff), col] = np.nan
        row_means = q.mean(axis=1, skipna=True)

        tmp = pd.DataFrame({'site_key': site_keys,
                            'gene': gene_first.values,
                            'mean_intensity': row_means.values}).dropna(subset=['mean_intensity'])
        if dedupe:
            grouped = tmp.groupby('site_key', as_index=False).agg(
                gene=('gene', 'first'),
                mean_intensity=('mean_intensity', 'max'),
            )
        else:
            grouped = tmp

        per_dilution[d] = dict(zip(grouped['site_key'], grouped['mean_intensity']))
        for sk, g in zip(grouped['site_key'], grouped['gene']):
            site_meta.setdefault(sk, g)

    all_sites = set().union(*[set(d_dict.keys()) for d_dict in per_dilution.values()])

    results = []
    for sk in all_sites:
        xs, ys = [], []
        for d in dilutions:
            v = per_dilution[d].get(sk)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                xs.append(dilution_lookup[d])
                ys.append(v)
        if len(xs) < min_dilutions:
            continue

        xs_arr = np.array(xs, dtype=float)
        ys_arr = np.array(ys, dtype=float)
        if log2_x:
            xs_arr = np.log2(xs_arr)
        if log2_y:
            ys_arr = np.log2(ys_arr)

        try:
            slope, intercept, r, _, _ = sst.linregress(xs_arr, ys_arr)
            results.append({
                'site_key': sk,
                'gene': site_meta[sk],
                'n_dilutions': len(xs),
                'r_squared': r ** 2,
                'slope': slope,
                'intercept': intercept,
                'mean_log2_intensity': ys_arr.mean(),
            })
        except Exception:
            pass

    return pd.DataFrame(results)


def plot_dilution_linearity(corr_df, r2_thresholds=(0.8, 0.95),
                             box_color='#8A0000', point_color='#393E46',
                             plot_width=600, plot_height=500,
                             plot_template='plotly_white'):
    """Histogram of per-site R^2 values from `calculate_dilution_linearity`.

    Vertical reference lines at the supplied thresholds let you read off
    what fraction of sites cross common quality cutoffs (0.8, 0.95).

    Parameters
    ----------
    corr_df : pd.DataFrame
        Output of `calculate_dilution_linearity`.
    r2_thresholds : tuple of float
        R^2 cutoffs to mark on the plot.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    ###
    import plotly.graph_objects as go
    ###

    r2 = corr_df['r_squared'].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r2, nbinsx=60,
        marker=dict(color=box_color, line=dict(color='white', width=0.4)),
        opacity=0.85,
    ))

    n = len(r2)
    annotation_lines = [f"<b>n sites</b> = {n:,}", f"median R² = {r2.median():.3f}"]
    for t in r2_thresholds:
        frac = (r2 >= t).mean() * 100
        fig.add_vline(x=t, line=dict(color='black', dash='dash', width=1),
                      annotation_text=f"R²≥{t}: {frac:.1f}%",
                      annotation_position='top')
        annotation_lines.append(f"R² ≥ {t}: {(r2 >= t).sum():,} ({frac:.1f}%)")

    fig.update_layout(
        template=plot_template,
        width=plot_width, height=plot_height,
        xaxis_title='Per-site R² (log2 intensity vs log2 dilution)',
        yaxis_title='Number of sites',
        showlegend=False,
        annotations=[dict(
            x=0.02, y=0.98, xref='paper', yref='paper',
            xanchor='left', yanchor='top', showarrow=False,
            text='<br>'.join(annotation_lines), align='left',
            bordercolor='black', borderwidth=1, borderpad=6,
            bgcolor='rgba(255,255,255,0.9)', font=dict(size=11),
        )]
    )
    return fig


def process_ptm_site_report(report_df, cutoff=0.75, noise_floor_filter=True,
                             per_run_mask=True, dedupe_by_gene=True,
                             ptm_type='Phospho', valid_aas=('S', 'T', 'Y')):
    """Process a Spectronaut PTM Site Report into a sites × samples matrix.

    The PTM Site Report is Spectronaut's pre-collapsed site-level output. This
    pipeline uses Spectronaut's site-determining-fragment quantification (more
    accurate per site than precursor-aggregation) and only applies the per-(site,
    run) localization mask + downstream cleaning.

    Output schema mirrors PeptideCollapse_v4.site_data so downstream helpers
    (assign_condition_setup, prepare_ID_datasets with loc_matrices,
    normalize_phospho_median, filter_per_condition_completeness) consume it
    without modification.

    Returns
    -------
    dict with:
      - 'site_data': wide DataFrame, one row per phosphosite, sample columns
          + Protein_group, Gene_group, PTM_0_aa, PTM_pos, PTM_mult123,
            PTM_flank, PTM_Collapse_key, PTM_localization metadata.
      - 'site_localization_per_run': PTM_Collapse_key × sample matrix of
          per-run loc probs, aligned to site_data row order.
    """
    ###
    import re
    import numpy as np
    import pandas as pd
    ###

    report_df = report_df.copy()

    # Filter to chosen PTM type (drops oxidation, deamidation, etc.)
    if ptm_type is not None and 'PTM.ModificationTitle' in report_df.columns:
        mods_seen_before = sorted(report_df['PTM.ModificationTitle'].astype(str).unique())
        is_target = report_df['PTM.ModificationTitle'].astype(str).str.contains(
            ptm_type, case=False, na=False
        )
        n_dropped = int((~is_target).sum())
        report_df = report_df[is_target].reset_index(drop=True)
        if n_dropped:
            print(f"Dropped {n_dropped:,} non-{ptm_type} rows. "
                  f"Mods in input: {mods_seen_before}. "
                  f"Mods kept: {sorted(report_df['PTM.ModificationTitle'].astype(str).unique())}")

    # Filter to canonical S/T/Y sites
    if valid_aas is not None:
        valid_aas = set(valid_aas)
        aa = report_df['PTM.SiteAA'].astype(str)
        is_valid = aa.isin(valid_aas)
        bad_counts = aa[~is_valid].value_counts().to_dict()
        n_dropped = int((~is_valid).sum())
        report_df = report_df[is_valid].reset_index(drop=True)
        if n_dropped:
            print(f"Dropped {n_dropped:,} rows with site AA not in "
                  f"{sorted(valid_aas)}: {bad_counts}")

    if len(report_df) == 0:
        raise ValueError("All rows filtered out — check ptm_type / valid_aas parameters.")

    # Identify probability + quantity columns and clean sample names
    prob_cols  = [c for c in report_df.columns if 'PTM.SiteProbability' in c]
    quant_cols = [c for c in report_df.columns if 'PTM.Quantity' in c]
    if not prob_cols or not quant_cols:
        raise ValueError("No PTM.SiteProbability or PTM.Quantity columns found in the report.")

    def clean(col, kind):
        s = re.sub(r'^\[\d+\]\s+', '', col)
        return s.replace(f'.raw.PTM.{kind}', '')

    name_prob  = {c: clean(c, 'SiteProbability') for c in prob_cols}
    name_quant = {c: clean(c, 'Quantity')        for c in quant_cols}
    samples = sorted(set(name_prob.values()) & set(name_quant.values()))
    if not samples:
        raise ValueError("No matching sample names between PTM.SiteProbability "
                         "and PTM.Quantity columns.")

    # Numeric cast (Spectronaut writes "Filtered" for dropped cells)
    prob_df  = (report_df[prob_cols].rename(columns=name_prob)[samples]
                                     .replace({'Filtered': np.nan, 'NaN': np.nan})
                                     .apply(pd.to_numeric, errors='coerce'))
    quant_df = (report_df[quant_cols].rename(columns=name_quant)[samples]
                                      .replace({'Filtered': np.nan, 'NaN': np.nan})
                                      .apply(pd.to_numeric, errors='coerce'))

    # Metadata + collapse key matching PeptideCollapse_v4 format
    meta = pd.DataFrame({
        'Protein_group': report_df['PTM.ProteinId'].astype(str),
        'Gene_group':    report_df['PG.Genes'].astype(str).str.split(';').str[0],
        'PTM_0_aa':      report_df['PTM.SiteAA'].astype(str),
        'PTM_pos':       report_df['PTM.SiteLocation'].astype(int),
        'PTM_mult123':   report_df['PTM.Multiplicity'].astype(int).clip(upper=3),
        'PTM_flank':     report_df.get('PTM.FlankingRegion',
                                       pd.Series(index=report_df.index, dtype=object)),
    })
    meta['PTM_Collapse_key'] = (
        meta['Protein_group'] + '~' + meta['Gene_group'] + '_' +
        meta['PTM_0_aa']      + meta['PTM_pos'].astype(str) + '_M' +
        meta['PTM_mult123'].astype(str)
    )

    # UPD_seq — modified-peptide-style annotation built from the 15-aa flanking
    # region (PeptideCollapse_v4 produces this from the full peptide; here we
    # approximate it from FlankingRegion since the PTM Site Report doesn't carry
    # the full parent peptide sequence). Format: lowercase site AA + '*' marker
    # at the centre of the flanking window — e.g. 'TPTVQEEs*EEEEVDE'.
    def _flank_to_upd_seq(flank, aa):
        if not isinstance(flank, str) or len(flank) < 1:
            return ''
        mid = len(flank) // 2
        return flank[:mid] + flank[mid].lower() + '*' + flank[mid + 1:]
    meta['UPD_seq'] = [
        _flank_to_upd_seq(f, a)
        for f, a in zip(meta['PTM_flank'].astype(str), meta['PTM_0_aa'].astype(str))
    ]

    # Per-(site, run) mask
    if per_run_mask:
        keep = (prob_df >= cutoff)
        masked = quant_df.where(keep, np.nan)
        n_pre  = int(quant_df.notna().sum().sum())
        n_post = int(masked.notna().sum().sum())
        print(f"Per-run mask (cutoff={cutoff:.2f}): {n_pre - n_post:,} of "
              f"{n_pre:,} intensity cells masked "
              f"({100*(n_pre - n_post)/max(1,n_pre):.1f}%).")
    else:
        masked = quant_df

    combined = pd.concat([masked.reset_index(drop=True),
                          meta.reset_index(drop=True)], axis=1)

    # Drop all-NaN sites (no Class I run survived)
    all_nan = combined[samples].isna().all(axis=1)
    n_drop = int(all_nan.sum())
    combined = combined[~all_nan].reset_index(drop=True)
    if n_drop:
        print(f"Dropped {n_drop:,} sites with no remaining quant after masking.")

    # Deduplicate (same site under multiple protein IDs) — keep row with the most
    # non-NaN sample cells; stable sort so ties break in original row order.
    if dedupe_by_gene:
        before = len(combined)
        combined = combined.assign(_n_valid=combined[samples].notna().sum(axis=1))
        combined = (combined
                    .sort_values('_n_valid', ascending=False, kind='stable')
                    .drop_duplicates(
                        subset=['Gene_group', 'PTM_0_aa', 'PTM_pos', 'PTM_mult123'],
                        keep='first'
                    )
                    .drop(columns='_n_valid')
                    .sort_index()
                    .reset_index(drop=True))
        if before != len(combined):
            print(f"Deduplicated multi-protein rows: {before:,} → {len(combined):,}.")

    # log2 + noise floor
    quant_log2 = np.log2(combined[samples].replace(0, np.nan))
    if noise_floor_filter:
        quant_log2 = quant_log2.replace(0, np.nan).replace(1, np.nan)
    combined.loc[:, samples] = quant_log2.values

    # Per-(site, run) loc matrix at the key level (max if duplicates remain)
    loc_long = pd.concat([prob_df.reset_index(drop=True),
                          meta[['PTM_Collapse_key']].reset_index(drop=True)], axis=1)
    loc_per_run = (loc_long.groupby('PTM_Collapse_key')[samples].max()
                          .reindex(combined['PTM_Collapse_key']))

    # Cross-file max for compat with PeptideCollapse_v4's PTM_localization col
    combined['PTM_localization'] = loc_per_run.max(axis=1).values

    print(f"Final: {len(combined):,} sites × {len(samples)} samples.")
    return {'site_data': combined, 'site_localization_per_run': loc_per_run}