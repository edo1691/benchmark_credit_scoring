

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
