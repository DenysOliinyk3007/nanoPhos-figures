import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA




def do_pca(
    df, meta_data=None, condition_to_remove=None, condition_to_keep=None, number_components=2
):
    """
    Perform Principal Component Analysis (PCA) on proteomics data.

    This function performs PCA dimensionality reduction on the input data
    and returns the transformed data, explained variance, and feature loadings.
    PCA is commonly used in proteomics to visualize sample clustering and
    identify the proteins that contribute most to the observed variation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with samples as rows and proteins/features as columns
    meta_data : pandas.DataFrame or None, default=None
        Optional metadata about samples (not used in current implementation)
    condition_to_remove : str or list or None, default=None
        Optional condition(s) to exclude from analysis (not used in current implementation)
    condition_to_keep : str or list or None, default=None
        Optional condition(s) to include in analysis (not used in current implementation)
    number_components : int, default=2
        Number of principal components to calculate

    Returns
    -------
    tuple
        A tuple containing three elements:
        - pc_df : pandas.DataFrame
            Dataframe with samples as rows and principal components as columns,
            preserving the original sample index
        - explained_var : numpy.ndarray
            Array of explained variance percentages for each principal component
        - loadings : pandas.DataFrame
            Dataframe with proteins/features as rows and their loadings (contributions)
            to each principal component as columns

    Notes
    -----
    The function converts all column names to strings to ensure compatibility
    with scikit-learn's PCA implementation. The explained variance is returned
    as percentages rather than proportions (multiplied by 100).

    While meta_data, condition_to_remove, and condition_to_keep parameters are
    included in the function signature, they are not currently used in the
    implementation. They may be intended for future filtering functionality.
    """
    # Initialize PCA with specified number of components
    pca = PCA(n_components=number_components)

    # Convert column names to strings for compatibility
    df.columns = df.columns.astype(str)

    # Fit PCA model and transform the data
    pcas = pca.fit_transform(df)

    # Create dataframe of transformed data with proper column names
    pc_df = pd.DataFrame(pcas, columns=[f"PC{i + 1}" for i in range(number_components)])

    # Preserve the original index from input dataframe
    pc_df = pc_df.set_index(df.index)

    # Calculate loadings (contribution of each feature to the principal components)
    loadings = pd.DataFrame(
        pca.components_.T,  # Transpose to get features as rows
        columns=[f"PC{i + 1}" for i in range(number_components)],
        index=df.columns,
    )

    # Convert explained variance ratio to percentages
    explained_var = pca.explained_variance_ratio_ * 100
    return pc_df, explained_var, loadings


def plot_pca_loadings(loadings, pc1=1, pc2=2, top_n=10, ax=None, return_top_features=False):
    """
    Create a PCA loadings plot highlighting the top contributing features.

    This function visualizes the loadings (contributions) of each feature to two
    principal components in a PCA analysis. It highlights three sets of top contributors:
    those with the highest combined contribution to both PCs, and those with the highest
    individual contributions to each PC. This helps identify which proteins or peptides
    are driving the separation in PCA plots of proteomics data.

    Parameters
    ----------
    loadings : pandas.DataFrame
        DataFrame containing PCA loadings with features as rows and principal
        components as columns. Columns should be named 'PC1', 'PC2', etc.
    pc1 : int, default=1
        The first principal component to plot (x-axis)
    pc2 : int, default=2
        The second principal component to plot (y-axis)
    top_n : int, default=10
        Number of top contributors to highlight in each category
    ax : matplotlib.axes.Axes or None, default=None
        The axes to plot on. If None, creates a new figure and axes.
    return_top_features : bool, default=False
        If True, returns DataFrames containing the top features for each category

    Returns
    -------
    matplotlib.axes.Axes or tuple
        If return_top_features is False, returns the axes object with the plot.
        If return_top_features is True, returns a tuple containing:
        - The axes object with the plot
        - A tuple of three DataFrames (combined, pc1, pc2) with top features

    Notes
    -----
    The function highlights three sets of top contributors with different colors:
    - Purple: Features with highest combined contribution to both PCs
    - Blue: Features with highest contribution to PC1
    - Red: Features with highest contribution to PC2

    The loadings are visualized in a scatter plot where:
    - The position indicates the direction and magnitude of contribution
    - Features far from the origin have stronger influence on the PCs
    - Features are labeled with their names and connected by dashed lines
    - Overlapping labels are automatically adjusted for better readability

    This visualization is particularly useful for interpreting PCA results in
    proteomics data by identifying which proteins drive the most variation.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all loadings with smaller circles and muted color
    ax.scatter(loadings[f"PC{pc1}"], loadings[f"PC{pc2}"], alpha=0.3, s=20, color="darkgray")

    # Calculate different types of contributions
    combined_contributions = np.sqrt(loadings[f"PC{pc1}"] ** 2 + loadings[f"PC{pc2}"] ** 2)
    pc1_contributions = abs(loadings[f"PC{pc1}"])
    pc2_contributions = abs(loadings[f"PC{pc2}"])

    # Get top features for each type
    top_combined = combined_contributions.nlargest(top_n)
    top_pc1 = pc1_contributions.nlargest(top_n)
    top_pc2 = pc2_contributions.nlargest(top_n)

    # Colors for different types of contributors
    colors = {
        "combined": "#9b59b6",  # Purple
        "pc1": "#3498db",  # Blue
        "pc2": "#e74c3c",  # Red
    }

    # Function to create and add labels for a set of features
    def add_labels(features, color, prefix=""):
        labels = []
        for feature in features.index:
            # Handle case where loc returns a Series instead of a single value
            x = (
                loadings.loc[feature, f"PC{pc1}"].iloc[0]
                if isinstance(loadings.loc[feature, f"PC{pc1}"], pd.Series)
                else loadings.loc[feature, f"PC{pc1}"]
            )
            y = (
                loadings.loc[feature, f"PC{pc2}"].iloc[0]
                if isinstance(loadings.loc[feature, f"PC{pc2}"], pd.Series)
                else loadings.loc[feature, f"PC{pc2}"]
            )

            # Add text label
            label = ax.text(x, y, str(feature), fontsize=8, color=color)
            labels.append(label)

            # Add a colored dot to highlight the feature
            ax.scatter(x, y, color=color, s=50, alpha=0.6)
        return labels

    # Add labels for all three sets of top contributors
    labels = []
    labels.extend(add_labels(top_combined, colors["combined"]))
    labels.extend(add_labels(top_pc1, colors["pc1"]))
    labels.extend(add_labels(top_pc2, colors["pc2"]))

    # Adjust text positions to avoid overlap using adjustText library
    adjust_text(labels, arrowprops=dict(arrowstyle="-", ls="dashed", alpha=0.5))

    # Add reference gridlines at origin
    ax.axhline(y=0, color="darkgray", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="darkgray", linestyle="--", alpha=0.3)

    # Add axis labels and title
    ax.set_xlabel(f"PC{pc1} loadings")
    ax.set_ylabel(f"PC{pc2} loadings")
    ax.set_title("PCA Loadings Plot")

    # Add legend with color-coded markers
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["combined"],
            label=f"Top {top_n} Combined",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["pc1"],
            label=f"Top {top_n} PC{pc1}",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["pc2"],
            label=f"Top {top_n} PC{pc2}",
            markersize=8,
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # Optionally return DataFrames with top feature information
    if return_top_features:
        # Create DataFrames for all three sets of top contributors
        combined_df = pd.DataFrame(
            {"Feature": top_combined.index, "Contribution": top_combined.values}
        )

        pc1_df = pd.DataFrame({"Feature": top_pc1.index, "Contribution": top_pc1.values})

        pc2_df = pd.DataFrame({"Feature": top_pc2.index, "Contribution": top_pc2.values})

        return ax, (combined_df, pc1_df, pc2_df)
    return ax


def plot_pca(
    df,
    x="PC1",
    y="PC2",
    hue=None,
    ax=None,
    title=None,
    savepath=None,
    number_components=2,
    number_plots=None,
    alpha=0.2,
    plot_loadings=False,
    top_n=10,
    plot_density=True,
    df_no_imputation=None,
):
    """
    Create advanced Principal Component Analysis (PCA) visualizations for proteomics data.

    This comprehensive PCA plotting function generates publication-quality visualizations
    of PCA results with multiple options for customization. It supports plotting density
    contours to highlight sample clustering, visualizing feature loadings to identify important
    proteins/peptides, and distinguishing imputed values from measured values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe for PCA with samples as rows and proteins/features as columns.
        Should contain pre-processed, imputed values ready for PCA.
    x : str, default='PC1'
        Principal component to plot on x-axis (e.g., 'PC1', 'PC2')
    y : str, default='PC2'
        Principal component to plot on y-axis (e.g., 'PC2', 'PC3')
    hue : str or pandas.Series or None, default=None
        Variable for color-coding points. Can be a column name in the index or a
        separate Series with sample identifiers as the index. If None, points are
        colored by their index.
    ax : matplotlib.axes.Axes or None, default=None
        Axes object to plot on. If None, creates new figure and axes.
    title : str or None, default=None
        Plot title. Also used as the base filename when saving.
    savepath : str or None, default=None
        Directory path where figures should be saved. If None, figures are not saved.
    number_components : int, default=2
        Number of principal components to compute in the PCA analysis
    number_plots : int or None, default=None
        Number of PC combinations to plot. If None, creates plots for all possible
        pairwise combinations of components.
    alpha : float, default=0.2
        Transparency level for density contours
    plot_loadings : bool, default=False
        Whether to create an additional plot showing feature loadings (contributions)
        to the selected principal components
    top_n : int, default=10
        Number of top contributing features to highlight in the loadings plot
    plot_density : bool, default=True
        Whether to plot density contours for categorical data to highlight
        sample clustering
    df_no_imputation : pandas.DataFrame or None, default=None
        Dataframe with the same shape as df but containing NaN values for missing data.
        Points that are NaN in this dataframe will be shown in grey to distinguish
        imputed values from measured values.

    Returns
    -------
    tuple
        The return value depends on whether loadings are plotted:
        - If plot_loadings=False: (fig,)
            - fig : matplotlib.figure.Figure - The main PCA plot figure
        - If plot_loadings=True: (fig, fig_loadings, top_features)
            - fig : matplotlib.figure.Figure - The main PCA plot figure
            - fig_loadings : matplotlib.figure.Figure - The loadings plot figure
            - top_features : tuple - DataFrames containing top features information

    Notes
    -----
    The function:
    - Computes PCA using the do_pca function
    - Creates appropriate number of subplots for displaying multiple PC combinations
    - Handles both continuous and categorical color variables
    - Optimizes color palettes based on the number of categories
    - Adds density contours to highlight sample clustering
    - Can display missing value information to distinguish imputed values
    - Optionally creates a loadings plot to identify key contributing features

    This visualization is particularly useful for exploratory data analysis
    in proteomics, helping to identify patterns, clusters, and outliers in
    complex datasets while also providing insight into which proteins are
    driving the observed variation.
    """
    # Compute PCA
    pca_df, explained_var, loadings = do_pca(df, number_components=number_components)

    # Set up plot grid
    if number_plots is None:
        number_plots = sum(range(number_components))
    ncols = int(np.ceil(np.sqrt(number_plots)))
    nrows = int(np.ceil(number_plots / ncols))

    # Create figure
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows)
    if ncols > 1:
        axs = axs.flatten()

    # Set title if provided
    if title is not None:
        plt.suptitle(title, y=1.02)

    # Set up hue and determine if continuous
    if hue is None:
        hue = pca_df.index
    is_continuous = isinstance(hue, pd.Series) and np.issubdtype(hue.dtype, np.number)

    # Get column name if hue is a Series
    hue_column = None
    if isinstance(hue, pd.Series):
        if hue.name is not None:
            hue_column = hue.name

    # Set marker size based on number of components
    marker_size = 30 if number_components <= 2 else 10

    # Set up color scheme
    unique_conditions = None
    color_dict = None

    if is_continuous:
        scatter_kws = {"c": hue, "cmap": "viridis", "s": marker_size}
    else:
        unique_conditions = pd.Series(hue).unique()
        if len(unique_conditions) < 8 and len(unique_conditions) > 4:
            colors = sns.color_palette("RdBu_r", n_colors=len(unique_conditions))
        elif len(unique_conditions) > 8:
            colors = sns.color_palette("Spectral", n_colors=len(unique_conditions))
        else:
            colors = sns.color_palette("Set2", n_colors=len(unique_conditions))
        color_dict = dict(zip(unique_conditions, colors))
        scatter_kws = {"hue": hue, "palette": color_dict, "s": marker_size}

    # Plot each PC combination
    plot_number = 0
    for i in range(number_components):
        if i == number_plots - 1:
            if plot_number == 0:
                _plot_pc_pair(
                    pca_df,
                    x,
                    y,
                    axs if ncols == 1 else axs[0],
                    scatter_kws,
                    is_continuous,
                    unique_conditions,
                    color_dict,
                    alpha,
                    explained_var,
                    plot_density,
                    df_no_imputation,
                    hue,
                    hue_column,
                )
            break

        for j in range(i + 1, number_components):
            current_ax = axs[plot_number]
            _plot_pc_pair(
                pca_df,
                f"PC{i + 1}",
                f"PC{j + 1}",
                current_ax,
                scatter_kws,
                is_continuous,
                unique_conditions,
                color_dict,
                alpha,
                explained_var,
                plot_density,
                df_no_imputation,
                hue,
                hue_column,
            )
            plot_number += 1

    # Hide unused subplots
    if ncols > 1:
        for i in range(len(axs) - number_plots):
            axs[-i - 1].set_visible(False)

    # Add legend if categorical
    if not is_continuous:
        if ncols > 1:
            first_ax = axs[0]
        else:
            first_ax = axs
        handles, labels = first_ax.get_legend_handles_labels()
        if df_no_imputation is not None:
            # Add "Missing Values" to legend only if there are NaN values
            if df_no_imputation.isna().any().any():
                handles.append(plt.scatter([], [], c="grey", alpha=0.2, s=marker_size))
                labels.append("Missing Values")
        fig.legend(
            handles,
            labels,
            frameon=True,
            fontsize=12,
            markerscale=1,
            bbox_to_anchor=(0.5, -0.02),
            loc="upper center",
            ncol=3,
        )

    fig.tight_layout()

    # Plot loadings if requested
    if plot_loadings:
        fig_loadings, ax_loadings = plt.subplots(figsize=(10, 10))
        loadings_result = plot_pca_loadings(
            loadings,
            pc1=int(x.replace("PC", "")),
            pc2=int(y.replace("PC", "")),
            top_n=top_n,
            ax=ax_loadings,
            return_top_features=True,
        )
        ax_loadings, top_features = loadings_result
        if title:
            ax_loadings.set_title(f"{title} - Loadings")

        if savepath is not None:
            _save_pca_plots(fig, fig_loadings, title, savepath)
        return fig, fig_loadings, top_features

    if savepath is not None:
        _save_pca_plots(fig, None, title, savepath)
    return fig


def _plot_pc_pair(
    pca_df,
    pc1,
    pc2,
    ax,
    scatter_kws,
    is_continuous,
    unique_conditions,
    color_dict,
    alpha,
    explained_var,
    plot_density=True,
    df_no_imputation=None,
    hue=None,
    hue_column=None,
):
    """
    Helper function to plot a single pair of principal components with advanced styling.

    This internal function handles the details of plotting one PC-PC combination in the
    PCA visualization. It supports density contours, highlighting of missing/imputed values,
    and different visualization approaches for continuous vs. categorical data.

    Parameters
    ----------
    pca_df : pandas.DataFrame
        DataFrame containing PCA results with samples as index and PCs as columns
    pc1 : str
        Name of the first PC to plot (e.g., 'PC1')
    pc2 : str
        Name of the second PC to plot (e.g., 'PC2')
    ax : matplotlib.axes.Axes
        Axes object to plot on
    scatter_kws : dict
        Keywords for scatter plot styling (includes color, size, etc.)
    is_continuous : bool
        Whether the hue variable is continuous or categorical
    unique_conditions : array-like or None
        Unique categories when hue is categorical
    color_dict : dict or None
        Mapping from categories to colors
    alpha : float
        Transparency level for density contours
    explained_var : array-like
        Percentage of variance explained by each PC
    plot_density : bool, default=True
        Whether to plot density contours for categorical data
    df_no_imputation : pandas.DataFrame or None, default=None
        DataFrame with NaN values indicating missing/imputed data
    hue : pandas.Series or None, default=None
        Variable used for color coding
    hue_column : str or None, default=None
        Column name if hue is a Series

    Notes
    -----
    This function handles several visualization scenarios:

    1. Missing/Imputed Data Visualization:
       - If df_no_imputation is provided, points with missing values are plotted in grey
       - This helps distinguish between measured and imputed values

    2. Categorical vs. Continuous Data:
       - For categorical data: uses sns.scatterplot with discrete color palette
       - For continuous data: uses plt.scatter with a continuous colormap

    3. Density Contours:
       - For categorical data: adds contours to highlight sample clusters
       - Only adds contours when there are 3+ points in a category
       - Uses get_density_contour to compute the 85% confidence region

    4. Axis Labels:
       - Adds variance explained percentages to axis labels

    The function uses zorder to control the visibility of overlapping elements:
    - Grey points for missing data: zorder=1
    - Density contours: zorder=2
    - Colored data points: zorder=3
    """

    if df_no_imputation is not None:
        # Create mask for points that have NaN values in the hue column only
        if hue_column is not None and hue_column in df_no_imputation.columns:
            nan_mask = df_no_imputation[hue_column].isna()
        else:
            # If we can't find the exact column, don't grey out any points
            nan_mask = pd.Series(False, index=pca_df.index)

        # Plot points with NaN values in grey
        if nan_mask.any():
            ax.scatter(
                pca_df[nan_mask][pc1],
                pca_df[nan_mask][pc2],
                c="grey",
                alpha=0.2,
                s=scatter_kws["s"],
                zorder=1,
                label="Missing Values",
            )

        # Plot non-NaN points with colors
        non_nan_mask = ~nan_mask

        # Plot density contours for categorical data
        if (
            plot_density
            and not is_continuous
            and unique_conditions is not None
            and color_dict is not None
        ):
            for condition in unique_conditions:
                mask = (
                    pd.Series(scatter_kws["hue"], index=pca_df.index) == condition
                ) & non_nan_mask
                points_x = pca_df.loc[mask, pc1]
                points_y = pca_df.loc[mask, pc2]

                if len(points_x) >= 3:
                    contour_path = get_density_contour(points_x, points_y, confidence=0.85)
                    if contour_path is not None:
                        patch = mpatches.PathPatch(
                            contour_path,
                            facecolor=color_dict[condition],
                            alpha=alpha,
                            edgecolor="none",
                            zorder=2,
                        )
                        ax.add_patch(patch)

        # Plot non-NaN points
        if is_continuous:
            # For continuous variables (like expression levels), use colormap
            scatter_kws_non_nan = scatter_kws.copy()
            if isinstance(scatter_kws["c"], pd.Series):
                scatter_kws_non_nan["c"] = scatter_kws["c"][non_nan_mask]
            plot = ax.scatter(
                pca_df[non_nan_mask][pc1],
                pca_df[non_nan_mask][pc2],
                zorder=3,
                **scatter_kws_non_nan,
            )
            cbar = plt.colorbar(plot, ax=ax)
            cbar.set_label("Expression Level", rotation=270, labelpad=15)
        else:
            # For categorical variables, use seaborn scatterplot
            scatter_kws_non_nan = scatter_kws.copy()
            if isinstance(scatter_kws["hue"], pd.Series):
                scatter_kws_non_nan["hue"] = scatter_kws["hue"][non_nan_mask]
            sns.scatterplot(
                data=pca_df[non_nan_mask], x=pc1, y=pc2, ax=ax, zorder=3, **scatter_kws_non_nan
            )
            ax.legend([], [], frameon=False)  # Remove individual subplot legends

    else:
        # Original plotting logic when no df_no_imputation is provided
        if (
            plot_density
            and not is_continuous
            and unique_conditions is not None
            and color_dict is not None
        ):
            for condition in unique_conditions:
                mask = pd.Series(scatter_kws["hue"], index=pca_df.index) == condition
                points_x = pca_df.loc[mask, pc1]
                points_y = pca_df.loc[mask, pc2]

                if len(points_x) >= 3:
                    contour_path = get_density_contour(points_x, points_y, confidence=0.85)
                    if contour_path is not None:
                        patch = mpatches.PathPatch(
                            contour_path,
                            facecolor=color_dict[condition],
                            alpha=alpha,
                            edgecolor="none",
                        )
                        ax.add_patch(patch)

        if is_continuous:
            # Plot with colormap for continuous variables
            plot = ax.scatter(pca_df[pc1], pca_df[pc2], zorder=3, **scatter_kws)
            cbar = plt.colorbar(plot, ax=ax)
            cbar.set_label("Expression Level", rotation=270, labelpad=15)
        else:
            # Plot with discrete colors for categorical variables
            sns.scatterplot(data=pca_df, x=pc1, y=pc2, ax=ax, zorder=3, **scatter_kws)
            ax.legend([], [], frameon=False)  # Remove individual subplot legends

    # Set labels with variance explained
    pc1_idx = int(pc1.replace("PC", "")) - 1
    pc2_idx = int(pc2.replace("PC", "")) - 1
    ax.set_xlabel(f"{pc1} ({explained_var[pc1_idx]:.1f}%)")
    ax.set_ylabel(f"{pc2} ({explained_var[pc2_idx]:.1f}%)")


def _save_pca_plots(fig, fig_loadings, title, savepath):
    """
    Helper function to save PCA plots and optional loadings plots to disk.

    This internal function handles the saving of PCA visualization figures,
    including both the main PCA plot and optional loadings plot. It uses the
    save_plot utility function to apply consistent formatting and naming
    conventions.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The main PCA plot figure to save
    fig_loadings : matplotlib.figure.Figure or None
        The loadings plot figure to save, if available
    title : str or None
        Title used as the base filename for saved files, after cleaning
    savepath : str
        Directory path where figures should be saved

    Notes
    -----
    The function:
    - Cleans the title to create a suitable filename using clean_filename
    - Saves the main PCA plot as both SVG and PNG formats
    - If loadings plot exists, saves it with '_loadings' suffix in both formats
    - Uses the save_plot utility for standardized file naming with date prefixes

    This ensures consistent file naming and format conventions across all
    saved PCA visualizations.
    """
    if title is not None:
        savename = clean_filename(title)
        save_plot(fig, savepath, savename, svg=True, png=True)
        if fig_loadings is not None:
            save_plot(fig_loadings, savepath, savename + "_loadings", svg=True, png=True)
