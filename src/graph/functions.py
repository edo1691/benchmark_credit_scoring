

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_imbalance_boxplot_grid(
    df: pd.DataFrame,
    dataset: str,
    models: list[str],
    metric: str = "rankstab_local",
    ylabel: str = "Global stability",
    xlabel: str = "Default rate",
    imbalance_order: list | None = None,
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (5.0, 3.8),
    show_means: bool = True,
    sharey: bool = True,
    suptitle: str | None = None,
):
    """
    Plot boxplots by imbalance level for multiple models in a subplot grid.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing columns:
        ['dataset', 'imbalance', 'model', metric]
    dataset : str
        Dataset to filter.
    models : list[str]
        Models to show, one subplot per model.
    metric : str
        Metric column to plot.
    ylabel : str
        Y-axis label.
    xlabel : str
        X-axis label.
    imbalance_order : list | None
        Explicit order of imbalance levels, e.g.
        [1, 2.5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ncols : int
        Number of subplot columns.
    figsize_per_panel : tuple[float, float]
        Width, height per subplot panel.
    show_means : bool
        Whether to plot mean as red dots.
    sharey : bool
        Whether subplots share y-axis.
    suptitle : str | None
        Global title for the full figure.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes array.
    """
    required_cols = {"dataset", "imbalance", "model", metric}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    dff = df[df["dataset"] == dataset].copy()
    if dff.empty:
        raise ValueError(f"No rows found for dataset={dataset!r}")

    dff = dff.dropna(subset=[metric, "imbalance", "model"])

    # Keep only models present in data
    models_present = [m for m in models if m in dff["model"].unique()]
    if not models_present:
        raise ValueError("None of the requested models are present in the filtered data.")

    # Imbalance order
    if imbalance_order is None:
        unique_vals = dff["imbalance"].dropna().unique()
        try:
            imbalance_order = sorted(unique_vals, key=float)
        except Exception:
            imbalance_order = sorted(unique_vals)

    imbalance_order = [x for x in imbalance_order if x in dff["imbalance"].unique()]
    if not imbalance_order:
        raise ValueError("No valid imbalance levels found.")

    n_models = len(models_present)
    nrows = math.ceil(n_models / ncols)

    fig_width = figsize_per_panel[0] * ncols
    fig_height = figsize_per_panel[1] * nrows

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        sharey=sharey,
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for ax, model in zip(axes_flat, models_present):
        sub = dff[dff["model"] == model].copy()

        if sub.empty:
            ax.set_visible(False)
            continue

        stats = (
            sub.groupby("imbalance")[metric]
            .agg(
                median="median",
                q1=lambda x: x.quantile(0.25),
                q3=lambda x: x.quantile(0.75),
                mean="mean",
            )
            .reindex(imbalance_order)
        )
        stats["iqr"] = stats["q3"] - stats["q1"]

        values = [
            sub.loc[sub["imbalance"] == imb, metric].values
            for imb in imbalance_order
        ]

        labels = [f"{imb}%" for imb in imbalance_order]

        bp = ax.boxplot(
            values,
            patch_artist=True,
            labels=labels,
            widths=0.75,
            showfliers=True,
        )

        colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(imbalance_order)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("#4d4d4d")
            patch.set_linewidth(0.8)

        for whisker in bp["whiskers"]:
            whisker.set_color("#7f7f7f")
            whisker.set_linewidth(0.8)

        for cap in bp["caps"]:
            cap.set_color("#7f7f7f")
            cap.set_linewidth(0.8)

        for median in bp["medians"]:
            median.set_color("#444444")
            median.set_linewidth(1.2)

        if show_means:
            means = stats["mean"].values
            ax.scatter(
                range(1, len(imbalance_order) + 1),
                means,
                color="red",
                s=10,
                zorder=3,
            )

        ax.set_title(model)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", linestyle="-", alpha=0.15)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    # Hide extra empty panels
    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    # Y-label only on leftmost visible axes
    for i, ax in enumerate(axes_flat[:n_models]):
        if i % ncols == 0:
            ax.set_ylabel(ylabel)

    if suptitle is not None:
        fig.suptitle(suptitle, y=0.995)

    plt.tight_layout()
    return fig, axes


def plot_model_boxplot(
    df: pd.DataFrame,
    dataset: str,
    imbalance: int | float | str,
    metric: str = "rankstab_local",
    ylabel: str = "Global stability",
    title: str | None = None,
    sort_desc: bool = True,
    figsize: tuple[float, float] = (8, 6),
):
    """
    Plot a boxplot by model for a given dataset and imbalance.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        ['feature', 'index', 'feature_importance', 'rankstab_local',
         'rankstab_local_weight', 'dataset', 'imbalance', 'model']
    dataset : str
        Dataset to filter, e.g. "german"
    imbalance : int | float | str
        Imbalance value to filter, e.g. 5
    metric : str
        Metric column to plot. Example:
        - "rankstab_local"
        - "rankstab_local_weight"
        - "feature_importance"
    ylabel : str
        Label for the y-axis.
    title : str | None
        Plot title. If None, uses dataset name.
    sort_desc : bool
        Sort models by descending median of the metric.
    figsize : tuple
        Figure size.
    """
    required_cols = {
        "dataset", "imbalance", "model", metric
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Filter
    dff = df[(df["dataset"] == dataset) & (df["imbalance"] == imbalance)].copy()

    if dff.empty:
        raise ValueError(
            f"No rows found for dataset={dataset!r} and imbalance={imbalance!r}"
        )

    dff = dff.dropna(subset=[metric, "model"])

    # Order models by median
    stats = (
        dff.groupby("model")[metric]
        .agg(
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
        )
        .sort_values("median", ascending=not sort_desc)
    )
    stats["iqr"] = stats["q3"] - stats["q1"]

    order = stats.index.tolist()

    # Data for boxplot
    values = [dff.loc[dff["model"] == m, metric].values for m in order]

    # Labels with summary stats
    labels = [
        f"{m}\nμ={stats.loc[m, 'median']:.2f}\nIQR={stats.loc[m, 'iqr']:.2f}"
        for m in order
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        values,
        patch_artist=True,
        labels=labels,
        widths=0.8,
        showfliers=True,
    )

    # Blue gradient
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(order)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    for median in bp["medians"]:
        median.set_color("#444444")
        median.set_linewidth(1.5)

    ax.set_title(title if title is not None else dataset)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()

    return fig, ax
