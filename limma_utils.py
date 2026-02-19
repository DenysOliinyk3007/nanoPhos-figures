"""
Limma analysis utilities for phosphoproteomics data.

This module provides modular functions for running limma differential expression
analysis with optional blocking for repeated measures designs.

STATISTICAL APPROACH:
====================
1. Repeated measures handling: duplicateCorrelation + lmFit with blocking
2. Variance moderation: eBayes with trend=True (intensity-dependent variance)
   and robust=True (downweight outliers)
3. Contrast testing: ALL contrasts fitted simultaneously for consistent variance
   estimates (not per-contrast eBayes)
4. Multiple testing correction: GLOBAL FDR by default across all features × contrasts
   to properly control family-wise false discovery rate
5. Interaction contrasts: For 2-factor designs, tests whether effects differ
   across factor levels (e.g., does fiber type effect change over time?)
6. Sample filtering: min_samples_per_group=4 default for reliable variance estimation

PUBLICATION METHODS STATEMENT:
==============================
"Differential phosphorylation analysis was performed using limma with empirical Bayes
variance moderation (trend=TRUE, robust=TRUE). For repeated measures, within-subject
correlation was estimated using duplicateCorrelation and incorporated into linear
models via lmFit. All pairwise contrasts and interaction terms were fitted simultaneously.
False discovery rate was controlled globally across all features and contrasts using
the Benjamini-Hochberg method (FDR < 0.05)."

Logs detailed output to file, displays only color-coded status flags in console.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import itertools

# Import R packages
limma = importr('limma')
base = importr('base')

# Global log file tracking
_log_file = None

def setup_logging(log_dir='limma_logs'):
    """Setup logging to file with timestamp."""
    global _log_file
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    _log_file = Path(log_dir) / f'limma_analysis_{timestamp}.log'

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(_log_file)]
    )
    print(f"📝 Logging to: {_log_file}")
    return _log_file

def get_log_file():
    """Get current log file path."""
    return _log_file

# Status display functions
def print_status(status, message):
    """Print colored status flag."""
    colors = {
        'green': '\033[92m✓',
        'yellow': '\033[93m⚠',
        'red': '\033[91m✗'
    }
    reset = '\033[0m'
    print(f"{colors.get(status, '')} {message}{reset}")

def check_sample_size(n_samples, n_blocks=None):
    """Check if sample size is adequate."""
    if n_blocks is not None:
        if n_blocks < 3:
            return 'red', f'Insufficient blocks (n={n_blocks}). Need ≥3.'
        elif n_blocks < 5:
            return 'yellow', f'Low blocks (n={n_blocks}). Power limited.'
        else:
            return 'green', f'{n_blocks} blocks.'
    else:
        if n_samples < 6:
            return 'red', f'Insufficient samples (n={n_samples}). Need ≥6.'
        elif n_samples < 10:
            return 'yellow', f'Low samples (n={n_samples}). Power limited.'
        else:
            return 'green', f'{n_samples} samples.'

def check_correlation(correlation):
    """Check correlation value."""
    if correlation < 0:
        return 'red', f'Negative correlation ({correlation:.3f}). Set to 0.'
    elif correlation > 0.9:
        return 'yellow', f'Very high correlation ({correlation:.3f}).'
    elif correlation > 0.7:
        return 'yellow', f'High correlation ({correlation:.3f}).'
    else:
        return 'green', f'Correlation = {correlation:.3f}.'

def check_efficiency(efficiency):
    """Check effective sample size efficiency."""
    if efficiency < 30:
        return 'red', f'Very low efficiency ({efficiency:.1f}%).'
    elif efficiency < 50:
        return 'yellow', f'Low efficiency ({efficiency:.1f}%).'
    else:
        return 'green', f'Efficiency = {efficiency:.1f}%.'


def align_expression_to_metadata(expression_data, metadata, sample_id_col='sample_id'):
    """
    Reorder expression data columns to match metadata row order.

    Parameters:
    -----------
    expression_data : pd.DataFrame
        Features × samples
    metadata : pd.DataFrame
        Sample metadata
    sample_id_col : str, default='sample_id'
        Column name in metadata containing sample IDs

    Returns:
    --------
    expression_data : pd.DataFrame
        Reordered expression data
    """

    if sample_id_col not in metadata.columns:
        raise ValueError(f"Column '{sample_id_col}' not found in metadata")

    meta_samples = metadata[sample_id_col].values
    expr_samples = expression_data.columns.values

    # Check all samples present
    missing_in_expr = set(meta_samples) - set(expr_samples)
    missing_in_meta = set(expr_samples) - set(meta_samples)

    if missing_in_expr:
        raise ValueError(f"Samples in metadata but not in expression data: {missing_in_expr}")

    if missing_in_meta:
        logging.warning(f"Samples in expression data but not in metadata (will be dropped): {missing_in_meta}")
        print_status('yellow', f"Warning: {len(missing_in_meta)} samples in expression data not in metadata")

    # Reorder expression data to match metadata
    expression_data_aligned = expression_data[meta_samples]

    logging.info(f"Expression data reordered: {expression_data_aligned.shape}")
    print_status('green', f"Expression data aligned to metadata order")

    return expression_data_aligned


def create_design_matrix(metadata, group_cols, block_col=None, separator='_', min_samples_per_group=4):
    """
    Create design matrix with R-valid group names.

    Parameters:
    -----------
    metadata : pd.DataFrame
        Sample metadata
    group_cols : list of str
        Columns to combine for groups (e.g., ['fiber_id', 'time_id'])
    block_col : str, optional
        Blocking variable (e.g., 'subject_id')
    separator : str, default='_'
        Separator for combining columns
    min_samples_per_group : int, default=4
        Minimum samples required per group. Groups with fewer samples are dropped.
        Raised from 3 to 4 for more reliable variance estimation, especially
        important for blocking designs where effective N is reduced by correlation.

    Returns:
    --------
    design, metadata, group_mapping
    """

    logging.info("="*70)
    logging.info("CREATING DESIGN MATRIX")
    logging.info("="*70)

    # Validate inputs
    metadata = metadata.copy()
    for col in group_cols:
        if col not in metadata.columns:
            raise ValueError(f"Column '{col}' not found in metadata")

    if block_col and block_col not in metadata.columns:
        raise ValueError(f"Block column '{block_col}' not found")

    # Create group variable
    logging.info(f"Creating groups from: {group_cols}")
    if len(group_cols) == 1:
        metadata['group'] = metadata[group_cols[0]].astype(str)
    else:
        metadata['group'] = metadata[group_cols[0]].astype(str)
        for col in group_cols[1:]:
            metadata['group'] = metadata['group'] + separator + metadata[col].astype(str)

    # Check sample counts per group
    group_counts = metadata['group'].value_counts()
    small_groups = group_counts[group_counts < min_samples_per_group]

    if len(small_groups) > 0:
        logging.warning(f"Groups with < {min_samples_per_group} samples (will be dropped):")
        for grp, count in small_groups.items():
            logging.warning(f"  {grp}: {count} samples")

        # Filter out small groups
        metadata = metadata[~metadata['group'].isin(small_groups.index)].reset_index(drop=True)

        print_status('yellow', f"Dropped {len(small_groups)} groups with < {min_samples_per_group} samples")
        logging.info(f"Dropped {len(small_groups)} groups, {len(metadata)} samples remaining")

    groups = sorted(metadata['group'].unique())
    logging.info(f"Found {len(groups)} unique groups (after filtering)")

    # Validate at least 2 groups remain
    if len(groups) < 2:
        error_msg = (
            f"ERROR: Only {len(groups)} group(s) remaining after filtering.\n"
            f"Need at least 2 groups to create contrasts.\n"
            f"Try reducing min_samples_per_group={min_samples_per_group} or using different grouping columns."
        )
        logging.error(error_msg)
        print_status('red', error_msg)
        raise ValueError(error_msg)

    # Convert to R-valid names
    logging.info("Converting to R-valid names")
    r_make_names = robjects.r['make.names']

    groups_r_valid = []
    group_mapping = {}

    for group in groups:
        r_valid_name = str(r_make_names(robjects.StrVector([group]))[0])
        groups_r_valid.append(r_valid_name)
        group_mapping[group] = r_valid_name
        if group != r_valid_name:
            logging.info(f"  {group} → {r_valid_name}")

    # Create design matrix
    design = pd.DataFrame(0, index=range(len(metadata)), columns=groups_r_valid)

    for orig_group, r_group in group_mapping.items():
        design[r_group] = (metadata['group'] == orig_group).astype(int)

    logging.info(f"Design matrix: {design.shape}")

    # Check sample size
    if block_col:
        n_blocks = metadata[block_col].nunique()
        status, msg = check_sample_size(len(metadata), n_blocks)
        print_status(status, f"Design: {len(groups)} groups, {n_blocks} blocks. {msg}")
    else:
        status, msg = check_sample_size(len(metadata))
        print_status(status, f"Design: {len(groups)} groups. {msg}")

    return design, metadata, group_mapping


def convert_to_r(expression_data, design, metadata, block_col=None):
    """
    Convert to R format.

    Parameters:
    -----------
    expression_data : pd.DataFrame
        Features × samples (columns must match metadata sample order)
    design : pd.DataFrame
        Design matrix
    metadata : pd.DataFrame
        Sample metadata (must have 'sample_id' column or index matching expression_data columns)
    block_col : str, optional
        Blocking variable

    Returns:
    --------
    r_expr, r_design, r_block
    """

    # Validate dimensions
    if expression_data.shape[1] != len(design):
        raise ValueError(f"Dimension mismatch: expression {expression_data.shape[1]} vs design {len(design)}")

    if len(design) != len(metadata):
        raise ValueError(f"Dimension mismatch: design {len(design)} vs metadata {len(metadata)}")

    # CRITICAL: Validate sample order alignment
    logging.info("Validating sample order alignment...")

    if hasattr(expression_data, 'columns'):
        expr_samples = list(expression_data.columns)

        # Try to get sample IDs from metadata
        if 'sample_id' in metadata.columns:
            meta_samples = list(metadata['sample_id'].values)
        elif hasattr(metadata.index, 'tolist'):
            meta_samples = metadata.index.tolist()
        else:
            # If no identifiable sample IDs, use integer positions and warn
            logging.warning("Cannot verify sample order: metadata has no 'sample_id' column or meaningful index")
            meta_samples = list(range(len(metadata)))

        # Check if samples match
        if len(expr_samples) == len(meta_samples):
            if expr_samples != meta_samples:
                # Check if same samples but different order
                if set(expr_samples) == set(meta_samples):
                    error_msg = (
                        f"CRITICAL ERROR: Sample order mismatch!\n"
                        f"Expression data and metadata have the same samples but in DIFFERENT ORDER.\n"
                        f"First 5 expression columns: {expr_samples[:5]}\n"
                        f"First 5 metadata samples: {meta_samples[:5]}\n"
                        f"\n*** RESULTS WOULD BE INVALID ***\n"
                        f"You must reorder expression_data columns to match metadata row order.\n"
                        f"Use: expression_data = expression_data[metadata['sample_id'].values]"
                    )
                else:
                    error_msg = (
                        f"CRITICAL ERROR: Sample mismatch!\n"
                        f"Expression data and metadata have DIFFERENT samples.\n"
                        f"Expression samples not in metadata: {set(expr_samples) - set(meta_samples)}\n"
                        f"Metadata samples not in expression: {set(meta_samples) - set(expr_samples)}\n"
                    )
                logging.error(error_msg)
                print_status('red', "Sample order validation FAILED")
                raise ValueError(error_msg)
            else:
                logging.info("✓ Sample order validated: expression columns match metadata rows")
                print_status('green', "Sample order validated")
        else:
            error_msg = f"Sample count mismatch: expression has {len(expr_samples)} but metadata has {len(meta_samples)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

    # Convert to R
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_expr = robjects.conversion.py2rpy(expression_data)
        r_design = robjects.conversion.py2rpy(design)

    logging.info(f"Converted: expression {r_expr.nrow} × {r_expr.ncol}, design {r_design.nrow} × {r_design.ncol}")

    # Create blocking factor
    r_block = None
    if block_col:
        r_block = robjects.FactorVector(metadata[block_col].values)
        logging.info(f"Blocking factor created: {len(r_block)} samples")

    print_status('green', f"Converted to R: {r_expr.nrow} features × {r_expr.ncol} samples")

    return r_expr, r_design, r_block


def estimate_correlation(r_expr, r_design, r_block):
    """
    Estimate within-block correlation.

    Parameters:
    -----------
    r_expr, r_design, r_block : R objects

    Returns:
    --------
    correlation : float
    """

    corfit = limma.duplicateCorrelation(r_expr, r_design, block=r_block)
    correlation = corfit.rx2('consensus.correlation')[0]

    original_corr = correlation
    if correlation < 0:
        logging.warning(f"Negative correlation {correlation:.3f}, setting to 0")
        correlation = 0

    logging.info(f"Within-block correlation: {correlation:.3f}")

    status, msg = check_correlation(original_corr if original_corr >= 0 else 0)
    print_status(status, f"Correlation: {msg}")

    return correlation


def fit_limma_model(r_expr, r_design, r_block=None, correlation=None, trend=True, robust=True):
    """
    Fit limma model.

    Parameters:
    -----------
    r_expr, r_design : R objects
    r_block : R FactorVector, optional
    correlation : float, optional
    trend, robust : bool

    Returns:
    --------
    fit : R MArrayLM object
    """

    # Fit model
    if r_block is not None:
        if correlation is None:
            raise ValueError("correlation required when using blocking")
        fit = limma.lmFit(r_expr, r_design, block=r_block, correlation=correlation)
        logging.info(f"Model fitted with blocking (correlation={correlation:.3f})")
    else:
        fit = limma.lmFit(r_expr, r_design)
        logging.info("Model fitted without blocking")

    # Empirical Bayes
    fit = limma.eBayes(fit, trend=trend, robust=robust)
    logging.info(f"Empirical Bayes applied (trend={trend}, robust={robust})")

    print_status('green', f"Model fitted. eBayes: trend={trend}, robust={robust}")

    return fit


def run_limma(expression_data, metadata, group_cols, block_col=None,
              separator='_', trend=True, robust=True, log_dir='limma_logs',
              auto_align=True, sample_id_col='sample_id', min_samples_per_group=4):
    """
    Complete limma workflow.

    Parameters:
    -----------
    expression_data : pd.DataFrame
        Features × samples
    metadata : pd.DataFrame
        Sample metadata
    group_cols : list of str
        Grouping columns
    block_col : str, optional
        Blocking variable
    separator : str, default='_'
        Group separator
    trend, robust : bool
        eBayes parameters
    log_dir : str
        Log directory
    auto_align : bool, default=True
        Automatically reorder expression_data to match metadata order
    sample_id_col : str, default='sample_id'
        Column in metadata with sample IDs
    min_samples_per_group : int, default=4
        Minimum samples required per group. Groups with fewer samples are dropped.
        Raised from 3 to 4 for more reliable variance estimation, especially
        important for blocking designs where effective N is reduced by correlation.

    Returns:
    --------
    fit, design, metadata, group_mapping, correlation, log_file
    """

    # Setup logging
    log_file = setup_logging(log_dir)
    print_status('green', f"LIMMA ANALYSIS STARTED")

    logging.info("="*70)
    logging.info("LIMMA ANALYSIS WORKFLOW")
    logging.info("="*70)

    # Step 0: Auto-align if requested
    if auto_align and sample_id_col in metadata.columns:
        logging.info("Auto-aligning expression data to metadata order...")
        expression_data = align_expression_to_metadata(expression_data, metadata, sample_id_col)

    # Step 1: Design matrix
    design, metadata, group_mapping = create_design_matrix(
        metadata, group_cols, block_col, separator, min_samples_per_group
    )

    # Filter expression data to match filtered metadata (if samples were dropped)
    if sample_id_col in metadata.columns:
        remaining_samples = metadata[sample_id_col].values
        expression_data = expression_data[remaining_samples]
        logging.info(f"Expression data filtered to {len(remaining_samples)} remaining samples")
        print_status('green', f"Expression data filtered to match {len(remaining_samples)} samples")

    # Step 2: Convert to R (with validation)
    r_expr, r_design, r_block = convert_to_r(
        expression_data, design, metadata, block_col
    )

    # Step 3: Estimate correlation if blocking
    correlation = None
    if r_block is not None:
        correlation = estimate_correlation(r_expr, r_design, r_block)

        # Calculate efficiency
        n_blocks = metadata[block_col].nunique()
        n_samples = len(metadata)
        n_effective = n_blocks + (n_samples - n_blocks) * (1 - correlation)
        efficiency = 100 * n_effective / n_samples
        logging.info(f"Effective N: {n_effective:.1f} / {n_samples} ({efficiency:.1f}%)")

        status, msg = check_efficiency(efficiency)
        print_status(status, f"Effective sample size: {msg}")

    # Step 4: Fit model
    fit = fit_limma_model(r_expr, r_design, r_block, correlation, trend, robust)

    print_status('green', f"LIMMA ANALYSIS COMPLETED")
    logging.info("="*70)

    return fit, design, metadata, group_mapping, correlation, log_file


def create_complete_contrasts(metadata, group_cols, group_mapping, separator='_',
                              include_interactions=True, include_trajectory_tests=False,
                              time_order=None):
    """
    Create all pairwise contrasts from grouping columns.

    Parameters:
    -----------
    metadata : pd.DataFrame
        With 'group' column
    group_cols : list of str
        Grouping columns
    group_mapping : dict
        Original → R-valid mapping
    separator : str
    include_interactions : bool, default=True
        For 2-factor designs, include interaction contrasts testing whether
        the effect of one factor differs across levels of the other factor
    include_trajectory_tests : bool, default=False
        For 2-factor designs with ≥3 levels in factor2 (time), include F-tests
        for overall trajectory differences. Tests whether the complete temporal
        pattern differs between factor1 levels (e.g., fiber types).

        Example: Tests if Type1 vs Type2A have different temporal trajectories
        across R1→P1→P24 by simultaneously testing both:
        - interaction_Type1vsType2A_P1vsR1
        - interaction_Type1vsType2A_P24vsR1

        These are multi-degree-of-freedom F-tests (2 df for 3 timepoints).
    time_order : list, optional
        Order of time levels for trajectory tests. The first element is treated as
        the reference/baseline timepoint. If None, automatically detects from
        interaction contrasts.

        Example: time_order=['R1', 'P1', 'P24']
        This makes R1 the baseline and tests interactions relative to R1.

        IMPORTANT: Only used when include_trajectory_tests=True. Specify this if
        your time levels don't sort naturally (e.g., 'R1' should be baseline but
        'P1' comes first alphabetically).

    Returns:
    --------
    contrasts : list of dict
        Each dict has keys: 'name', 'formula', 'comparison', 'type'
        For trajectory tests, 'formula' is a list of contrast names (coef indices)
    """

    logging.info("Creating contrasts")

    contrasts = []

    # Parse groups
    group_components = {}
    for orig_group in metadata['group'].unique():
        parts = orig_group.split(separator)
        if len(parts) != len(group_cols):
            continue
        group_components[orig_group] = {group_cols[i]: parts[i] for i in range(len(parts))}

    if len(group_cols) == 1:
        # Single factor
        factor = group_cols[0]
        levels = sorted(set(comp[factor] for comp in group_components.values()))

        for level1, level2 in itertools.combinations(levels, 2):
            contrasts.append({
                'name': f'{level1}_vs_{level2}',
                'formula': f'{group_mapping[level1]} - {group_mapping[level2]}',
                'comparison': f'{level1} vs {level2}',
                'type': 'pairwise'
            })

    elif len(group_cols) == 2:
        # Two factors
        factor1, factor2 = group_cols
        factor1_levels = sorted(set(comp[factor1] for comp in group_components.values()))
        factor2_levels = sorted(set(comp[factor2] for comp in group_components.values()))

        # Within-factor2 comparisons
        for f2_level in factor2_levels:
            for f1_level1, f1_level2 in itertools.combinations(factor1_levels, 2):
                group1 = separator.join([f1_level1, f2_level])
                group2 = separator.join([f1_level2, f2_level])

                if group1 in group_mapping and group2 in group_mapping:
                    contrasts.append({
                        'name': f'{f1_level1}_vs_{f1_level2}_at_{f2_level}',
                        'formula': f'{group_mapping[group1]} - {group_mapping[group2]}',
                        'comparison': f'{f1_level1} vs {f1_level2} at {f2_level}',
                        'type': f'within_{factor2}'
                    })

        # Within-factor1 comparisons
        for f1_level in factor1_levels:
            for f2_level1, f2_level2 in itertools.combinations(factor2_levels, 2):
                group1 = separator.join([f1_level, f2_level1])
                group2 = separator.join([f1_level, f2_level2])

                if group1 in group_mapping and group2 in group_mapping:
                    contrasts.append({
                        'name': f'{f1_level}_{f2_level1}_vs_{f2_level2}',
                        'formula': f'{group_mapping[group1]} - {group_mapping[group2]}',
                        'comparison': f'{f1_level}: {f2_level1} vs {f2_level2}',
                        'type': f'within_{factor1}'
                    })

        # Interaction contrasts (if requested)
        if include_interactions and len(factor1_levels) >= 2 and len(factor2_levels) >= 2:
            logging.info("Adding interaction contrasts...")
            n_interactions = 0

            # For each pair of factor1 levels and each pair of factor2 levels
            for f1_a, f1_b in itertools.combinations(factor1_levels, 2):
                for f2_a, f2_b in itertools.combinations(factor2_levels, 2):
                    # Check all 4 groups exist
                    g_f1a_f2a = separator.join([f1_a, f2_a])
                    g_f1a_f2b = separator.join([f1_a, f2_b])
                    g_f1b_f2a = separator.join([f1_b, f2_a])
                    g_f1b_f2b = separator.join([f1_b, f2_b])

                    if all(g in group_mapping for g in [g_f1a_f2a, g_f1a_f2b, g_f1b_f2a, g_f1b_f2b]):
                        # Interaction: (f1a_f2b - f1a_f2a) - (f1b_f2b - f1b_f2a)
                        # = f1a_f2b - f1a_f2a - f1b_f2b + f1b_f2a
                        formula = (
                            f'({group_mapping[g_f1a_f2b]} - {group_mapping[g_f1a_f2a]}) - '
                            f'({group_mapping[g_f1b_f2b]} - {group_mapping[g_f1b_f2a]})'
                        )
                        contrasts.append({
                            'name': f'interaction_{f1_a}vs{f1_b}_{f2_a}vs{f2_b}',
                            'formula': formula,
                            'comparison': f'Interaction: {f1_a} vs {f1_b} effect differs between {f2_a} and {f2_b}',
                            'type': 'interaction'
                        })
                        n_interactions += 1

            if n_interactions > 0:
                logging.info(f"  Added {n_interactions} interaction contrasts")

        # Trajectory F-tests (if requested and ≥3 timepoints)
        if include_trajectory_tests and len(factor1_levels) >= 2 and len(factor2_levels) >= 3:
            logging.info("Adding trajectory F-tests for overall temporal patterns...")
            n_trajectory = 0

            # Determine reference timepoint
            if time_order is not None:
                if len(time_order) != len(factor2_levels):
                    logging.warning(f"time_order length ({len(time_order)}) doesn't match factor2 levels ({len(factor2_levels)})")
                    logging.warning(f"time_order: {time_order}, factor2_levels: {factor2_levels}")
                    logging.warning("Will auto-detect reference from interaction contrasts")
                    reference_time = None
                    other_times = None
                else:
                    reference_time = time_order[0]
                    other_times = time_order[1:]
                    logging.info(f"Using user-specified time order: {time_order}")
                    logging.info(f"Reference timepoint: {reference_time}")
                    logging.info(f"Other timepoints: {other_times}")
            else:
                reference_time = None
                other_times = None
                logging.info("No time_order specified, will auto-detect from interaction contrasts")

            # For each fiber pair, find all their interaction contrasts
            for f1_a, f1_b in itertools.combinations(factor1_levels, 2):
                # Build list of interaction contrast names for this fiber pair
                # These will be tested jointly in an F-test
                interaction_contrast_names = []

                if reference_time is not None:
                    # User specified time order - look for specific interactions
                    for time_point in other_times:
                        interaction_name = f'interaction_{f1_a}vs{f1_b}_{time_point}vs{reference_time}'
                        if any(c['name'] == interaction_name for c in contrasts):
                            interaction_contrast_names.append(interaction_name)
                        else:
                            logging.warning(f"  Expected interaction {interaction_name} not found")
                else:
                    # Auto-detect - find ALL interactions for this fiber pair
                    fiber_pair_pattern = f'interaction_{f1_a}vs{f1_b}_'
                    for contrast in contrasts:
                        if contrast['name'].startswith(fiber_pair_pattern) and contrast['type'] == 'interaction':
                            interaction_contrast_names.append(contrast['name'])

                logging.info(f"  Found {len(interaction_contrast_names)} interactions for {f1_a} vs {f1_b}: {interaction_contrast_names}")

                # If we have ≥2 interaction contrasts, we can do an F-test
                if len(interaction_contrast_names) >= 2:
                    # Extract timepoint comparisons from interaction names for display
                    time_comparisons = []
                    for int_name in interaction_contrast_names:
                        # Extract the time part: interaction_Type1vsType2A_P1vsR1 -> P1vsR1
                        time_part = int_name.split('_')[-1]
                        time_comparisons.append(time_part)

                    contrasts.append({
                        'name': f'trajectory_{f1_a}vs{f1_b}',
                        'formula': interaction_contrast_names,  # List of contrast names for F-test
                        'comparison': (
                            f'Overall trajectory: {f1_a} vs {f1_b} across timepoints '
                            f'({", ".join(time_comparisons)}) - F-test, {len(interaction_contrast_names)} df'
                        ),
                        'type': 'trajectory',
                        'df': len(interaction_contrast_names)
                    })
                    n_trajectory += 1

            if n_trajectory > 0:
                logging.info(f"  Added {n_trajectory} trajectory F-tests")
                logging.info(f"  Each F-test jointly tests {len(interaction_contrast_names)} interaction contrasts")

    logging.info(f"Created {len(contrasts)} total contrasts")
    print_status('green', f"Contrasts: {len(contrasts)} created")

    return contrasts


def test_contrasts_and_extract(fit, contrasts, adjust_method='BH', p_cutoff=0.05, lfc_cutoff=0,
                                global_fdr=True):
    """
    Test contrasts and extract results.

    STATISTICAL NOTE: By default, applies GLOBAL FDR correction across all contrasts
    to properly control family-wise false discovery rate. Set global_fdr=False for
    per-contrast FDR (less conservative but higher false positives).

    Parameters:
    -----------
    fit : R MArrayLM
    contrasts : list of dict
    adjust_method : str, default='BH'
        Method for FDR correction (BH = Benjamini-Hochberg)
    p_cutoff : float, default=0.05
        Adjusted p-value cutoff
    lfc_cutoff : float, default=0
        Log fold-change cutoff (absolute value)
    global_fdr : bool, default=True
        If True, applies FDR correction globally across all features × contrasts.
        If False, applies per-contrast FDR (NOT recommended for many contrasts).

        IMPORTANT: With global_fdr=True, the adj.P.Val controls FDR across the
        entire family of tests. This is more stringent but statistically appropriate
        when testing many contrasts.

    Returns:
    --------
    results : dict of DataFrames
    """

    # Separate regular contrasts from trajectory F-tests
    regular_contrasts = [c for c in contrasts if isinstance(c['formula'], str)]
    trajectory_contrasts = [c for c in contrasts if isinstance(c['formula'], list)]

    n_regular = len(regular_contrasts)
    n_trajectory = len(trajectory_contrasts)

    logging.info(f"Testing {n_regular} regular contrasts + {n_trajectory} trajectory F-tests")
    logging.info(f"FDR correction mode: {'GLOBAL across all contrasts' if global_fdr else 'PER-CONTRAST (less conservative)'}")
    print_status('green', f"Testing {n_regular} regular contrasts + {n_trajectory} trajectory F-tests (global FDR: {global_fdr})...")

    # FIT ALL REGULAR CONTRASTS SIMULTANEOUSLY (correct approach)
    # Build combined contrast matrix
    all_formulas = [c['formula'] for c in regular_contrasts]
    contrast_names_r = robjects.StrVector([c['name'] for c in regular_contrasts])

    logging.info("Building contrast matrix for all regular contrasts simultaneously...")

    # Create contrast matrix with all formulas at once
    contrast_matrix = limma.makeContrasts(
        contrasts=robjects.StrVector(all_formulas),
        levels=fit.rx2('design')
    )

    # Assign column names to contrast matrix
    contrast_matrix.colnames = contrast_names_r

    logging.info(f"Contrast matrix: {contrast_matrix.nrow} groups × {contrast_matrix.ncol} regular contrasts")

    # Fit all contrasts at once
    fit2 = limma.contrasts_fit(fit, contrast_matrix)

    # Apply empirical Bayes moderation ONCE for all contrasts
    fit2 = limma.eBayes(fit2, trend=True, robust=True)
    logging.info("Empirical Bayes moderation applied to all contrasts simultaneously")

    # Extract results for each regular contrast
    results = {}
    all_pvalues = []  # For global FDR
    contrast_name_to_index = {c['name']: i for i, c in enumerate(regular_contrasts)}

    for i, contrast_info in enumerate(regular_contrasts):
        name = contrast_info['name']

        logging.info(f"Extracting: {name}")

        # Extract results for this contrast (coef = column index + 1 in R)
        # Don't apply adjust.method here if doing global FDR
        if global_fdr:
            toptable = limma.topTable(
                fit2,
                coef=i+1,  # R is 1-indexed
                number=robjects.r('Inf'),
                adjust_method='none',  # Will apply global correction later
                sort_by='none'  # Keep original order for alignment
            )
        else:
            toptable = limma.topTable(
                fit2,
                coef=i+1,
                number=robjects.r('Inf'),
                adjust_method=adjust_method
            )

        # Convert to pandas
        with localconverter(robjects.default_converter + pandas2ri.converter):
            result_df = robjects.conversion.rpy2py(toptable)

        # Store raw p-values for global correction
        if global_fdr:
            result_df['contrast_name'] = name  # Track which contrast
            all_pvalues.append(result_df[['P.Value']].copy())
            all_pvalues[-1]['contrast_name'] = name
            all_pvalues[-1]['feature'] = result_df.index

        results[name] = result_df

    # Extract trajectory F-tests
    if n_trajectory > 0:
        logging.info(f"\\nExtracting {n_trajectory} trajectory F-tests...")

        for traj_contrast in trajectory_contrasts:
            traj_name = traj_contrast['name']
            interaction_names = traj_contrast['formula']  # List of contrast names

            logging.info(f"Extracting F-test: {traj_name}")
            logging.info(f"  Testing {len(interaction_names)} interaction contrasts jointly:")
            for int_name in interaction_names:
                logging.info(f"    - {int_name}")

            # Get coefficient indices for these contrasts
            coef_indices = []
            for int_name in interaction_names:
                if int_name in contrast_name_to_index:
                    # R is 1-indexed
                    coef_indices.append(contrast_name_to_index[int_name] + 1)
                else:
                    logging.warning(f"  WARNING: Interaction {int_name} not found, skipping")

            if len(coef_indices) < 2:
                logging.warning(f"  WARNING: Not enough valid coefficients for F-test, skipping {traj_name}")
                continue

            # Perform F-test using topTable with multiple coefficients
            if global_fdr:
                toptable_ftest = limma.topTable(
                    fit2,
                    coef=robjects.IntVector(coef_indices),  # Multi-coef F-test
                    number=robjects.r('Inf'),
                    adjust_method='none',
                    sort_by='none'
                )
            else:
                toptable_ftest = limma.topTable(
                    fit2,
                    coef=robjects.IntVector(coef_indices),
                    number=robjects.r('Inf'),
                    adjust_method=adjust_method
                )

            # Convert to pandas
            with localconverter(robjects.default_converter + pandas2ri.converter):
                ftest_df = robjects.conversion.rpy2py(toptable_ftest)

            # F-test results have F-statistic and P.Value, but no logFC (multi-dimensional)
            # Add placeholder logFC as NaN for compatibility
            ftest_df['logFC'] = np.nan
            ftest_df['AveExpr'] = ftest_df.get('AveExpr', np.nan)
            ftest_df['t'] = np.nan  # F-test doesn't have t-statistic

            # Store raw p-values for global correction
            if global_fdr:
                ftest_df['contrast_name'] = traj_name
                all_pvalues.append(ftest_df[['P.Value']].copy())
                all_pvalues[-1]['contrast_name'] = traj_name
                all_pvalues[-1]['feature'] = ftest_df.index

            results[traj_name] = ftest_df

            logging.info(f"  F-test completed: {len(ftest_df)} features tested")

    # Apply GLOBAL FDR correction if requested
    if global_fdr:
        logging.info("Applying global FDR correction across all contrasts...")

        # Collect all p-values
        all_p_df = pd.concat([
            pd.DataFrame({
                'feature': results[c['name']].index,
                'contrast': c['name'],
                'P.Value': results[c['name']]['P.Value'].values
            })
            for c in contrasts
        ], ignore_index=True)

        # Apply BH correction to ALL p-values
        from statsmodels.stats.multitest import multipletests
        _, adj_pvals_global, _, _ = multipletests(
            all_p_df['P.Value'].values,
            alpha=p_cutoff,
            method='fdr_bh'
        )
        all_p_df['adj.P.Val_global'] = adj_pvals_global

        # Update each contrast's results with global FDR
        for contrast_info in contrasts:
            name = contrast_info['name']
            contrast_pvals = all_p_df[all_p_df['contrast'] == name].copy()

            # Match by feature index
            results[name] = results[name].copy()
            results[name]['adj.P.Val'] = contrast_pvals.set_index('feature')['adj.P.Val_global']

        logging.info(f"Global FDR applied: {len(all_p_df)} total tests (features × contrasts)")

    # Add significance flags and count
    total_sig = 0
    for contrast_info in contrasts:
        name = contrast_info['name']

        # For trajectory F-tests, only use p-value (no logFC threshold)
        if contrast_info.get('type') == 'trajectory':
            results[name]['significant'] = (results[name]['adj.P.Val'] < p_cutoff)
        else:
            # Regular contrasts: use both p-value and logFC thresholds
            results[name]['significant'] = (
                (results[name]['adj.P.Val'] < p_cutoff) &
                (results[name]['logFC'].abs() > lfc_cutoff)
            )

        n_sig = results[name]['significant'].sum()
        total_sig += n_sig
        logging.info(f"  {name}: {n_sig} significant features")

    # Summary status
    fdr_mode_str = "global FDR" if global_fdr else "per-contrast FDR"
    if total_sig == 0:
        print_status('yellow', f"Results: 0 significant features ({fdr_mode_str})")
    else:
        print_status('green', f"Results: {total_sig} significant features ({fdr_mode_str}, adj.P < {p_cutoff}, |logFC| > {lfc_cutoff})")

    # Warning about multiple testing
    if not global_fdr and len(contrasts) > 20:
        print_status('yellow', f"WARNING: Using per-contrast FDR with {len(contrasts)} contrasts. Consider global_fdr=True.")

    return results


def run_limma_pipeline(expression_data, metadata, group_cols, block_col=None,
                       separator='_', trend=True, robust=True, log_dir='limma_logs',
                       auto_align=True, sample_id_col='sample_id',
                       adjust_method='BH', p_cutoff=0.05, lfc_cutoff=0,
                       min_samples_per_group=4, include_interactions=True,
                       include_trajectory_tests=False, time_order=None, global_fdr=True):
    """
    Complete end-to-end limma pipeline: model fitting → contrasts → results.

    Returns a single consolidated DataFrame with all contrasts and all features.

    Parameters:
    -----------
    expression_data : pd.DataFrame
        Features × samples
    metadata : pd.DataFrame
        Sample metadata
    group_cols : list of str
        Grouping columns (e.g., ['fiber_id', 'time_id'])
    block_col : str, optional
        Blocking variable (e.g., 'subject_id')
    separator : str, default='_'
        Group separator
    trend, robust : bool
        eBayes parameters
    log_dir : str
        Log directory
    auto_align : bool, default=True
        Automatically reorder expression_data to match metadata order
    sample_id_col : str, default='sample_id'
        Column in metadata with sample IDs
    adjust_method : str, default='BH'
        P-value adjustment method
    p_cutoff : float, default=0.05
        Adjusted p-value cutoff
    lfc_cutoff : float, default=0
        Log fold-change cutoff (absolute value)
    min_samples_per_group : int, default=4
        Minimum samples required per group. Groups with fewer samples are dropped.
        Raised from 3 to 4 for more reliable variance estimation, especially
        important for blocking designs where effective N is reduced by correlation.
    include_interactions : bool, default=True
        For 2-factor designs (e.g., fiber × time), include interaction contrasts
        that test whether the effect of one factor differs across levels of the other.
        Example: Does the fiber type effect differ between time points?
    include_trajectory_tests : bool, default=False
        For 2-factor designs with ≥3 timepoints, include F-tests for overall temporal
        trajectory differences. Tests whether the complete temporal pattern (e.g.,
        R1→P1→P24) differs between factor1 levels (e.g., fiber types).

        These are multi-degree-of-freedom F-tests that jointly test multiple interaction
        contrasts. More powerful than looking at pairwise interactions individually.

        Example: Tests if Type1 has a different overall trajectory than Type2A by
        simultaneously testing interaction_Type1vsType2A_P1vsR1 AND
        interaction_Type1vsType2A_P24vsR1.

        NOTE: Requires include_interactions=True. Set to False for standard analysis.
    time_order : list, optional
        Order of time levels for trajectory tests. First element = reference/baseline.
        Only used when include_trajectory_tests=True.

        Example: time_order=['R1', 'P1', 'P24']
        Makes R1 the baseline and tests interactions: P1vsR1, P24vsR1

        If None, auto-detects reference from interaction contrasts (uses all available).
        Specify this if your time levels don't sort naturally.
    global_fdr : bool, default=True
        Apply FDR correction globally across all features × contrasts (recommended).
        If False, applies per-contrast FDR which is less conservative but increases
        false positives when testing many contrasts.

        IMPORTANT FOR PUBLICATION: Global FDR properly controls the family-wise
        false discovery rate and should be reported in methods sections.

    Returns:
    --------
    results_df : pd.DataFrame
        Consolidated results with columns:
        - feature: phosphosite/gene ID (index)
        - contrast: contrast name
        - comparison: human-readable comparison description
        - contrast_type: type of contrast (within_time_id, within_fiber_id, etc.)
        - logFC: log fold-change
        - AveExpr: average expression
        - t: t-statistic
        - P.Value: raw p-value
        - adj.P.Val: adjusted p-value
        - B: B-statistic
        - significant: boolean flag (adj.P.Val < p_cutoff & |logFC| > lfc_cutoff)

    Example:
    --------
    results = run_limma_pipeline(
        phospho_data,
        fiber_metadata,
        group_cols=['fiber_id', 'time_id'],
        block_col='subject_id',
        p_cutoff=0.05,
        lfc_cutoff=0.5
    )

    # Filter for significant results
    sig_results = results[results['significant']]

    # Get results for specific contrast
    fast_vs_slow = results[results['contrast'] == 'Fast-2A_vs_Slow-1_at_P1']

    # Get all within-time comparisons
    within_time = results[results['contrast_type'] == 'within_time_id']

    # Top 10 by p-value across all contrasts
    top10 = results.nsmallest(10, 'adj.P.Val')
    """

    print_status('green', "="*70)
    print_status('green', "COMPLETE LIMMA PIPELINE")
    print_status('green', "="*70)

    # Step 1: Fit limma model
    fit, design, metadata_updated, group_mapping, correlation, log_file = run_limma(
        expression_data, metadata, group_cols, block_col,
        separator, trend, robust, log_dir, auto_align, sample_id_col, min_samples_per_group
    )

    # Step 2: Create all contrasts
    contrasts = create_complete_contrasts(
        metadata_updated, group_cols, group_mapping, separator,
        include_interactions, include_trajectory_tests, time_order
    )

    # Validate contrasts exist
    if len(contrasts) == 0:
        error_msg = (
            "ERROR: No valid contrasts could be created.\n"
            f"After filtering, only {len(group_mapping)} group(s) remained.\n"
            f"Try reducing min_samples_per_group={min_samples_per_group} or using different grouping columns."
        )
        logging.error(error_msg)
        print_status('red', error_msg)
        raise ValueError(error_msg)

    # Step 3: Test contrasts and extract results
    contrast_results = test_contrasts_and_extract(
        fit, contrasts, adjust_method, p_cutoff, lfc_cutoff, global_fdr
    )

    # Step 4: Consolidate all results into single DataFrame
    logging.info("Consolidating results into single DataFrame...")

    all_results = []

    for contrast_info in contrasts:
        contrast_name = contrast_info['name']
        contrast_df = contrast_results[contrast_name].copy()

        # Add contrast metadata
        contrast_df['contrast'] = contrast_name
        contrast_df['comparison'] = contrast_info['comparison']
        contrast_df['contrast_type'] = contrast_info['type']

        # Reset index to make feature ID a column
        contrast_df['feature'] = contrast_df.index

        all_results.append(contrast_df)

    # Concatenate all results
    results_df = pd.concat(all_results, ignore_index=True)

    # Reorder columns for better readability
    col_order = ['feature', 'contrast', 'comparison', 'contrast_type',
                 'logFC', 'AveExpr', 't', 'P.Value', 'adj.P.Val', 'B', 'significant']
    results_df = results_df[col_order]

    # Sort by contrast, then by adjusted p-value
    results_df = results_df.sort_values(['contrast', 'adj.P.Val']).reset_index(drop=True)

    # Summary statistics
    n_features = results_df['feature'].nunique()
    n_contrasts = results_df['contrast'].nunique()
    n_sig_total = results_df['significant'].sum()
    n_sig_features = results_df[results_df['significant']]['feature'].nunique()

    logging.info(f"Consolidated DataFrame: {len(results_df)} rows ({n_features} features × {n_contrasts} contrasts)")
    logging.info(f"Significant results: {n_sig_total} ({n_sig_features} unique features)")

    print_status('green', f"Results: {len(results_df)} rows ({n_features} features × {n_contrasts} contrasts)")
    print_status('green', f"Significant: {n_sig_total} results ({n_sig_features} unique features)")

    print_status('green', "="*70)
    print_status('green', f"PIPELINE COMPLETED - Log: {log_file}")
    print_status('green', "="*70)

    logging.info("="*70)
    logging.info("PIPELINE SUMMARY")
    logging.info("="*70)
    logging.info(f"Groups: {len(group_mapping)}")
    logging.info(f"Contrasts tested: {n_contrasts}")
    logging.info(f"Features: {n_features}")
    logging.info(f"Total rows: {len(results_df)}")
    logging.info(f"Significant results: {n_sig_total}")
    logging.info(f"Correlation: {correlation}")
    logging.info(f"Log file: {log_file}")
    logging.info("="*70)

    return results_df
