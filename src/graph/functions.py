from __future__ import annotations

import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path
import math
import seaborn as sns


import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_imbalance_boxplot_grid(
    df: pd.DataFrame,
    datasets: list[str] | str,
    models: list[str],
    metric: str = "rankstab_local",
    ylabel: str = "Local stability",
    xlabel: str = "Default rate",
    imbalance_order: list | None = None,
    figsize_per_panel: tuple[float, float] = (5.0, 4.8),
    show_means: bool = True,
    sharey: bool = True,
    suptitle: str | None = None,
    ylim: tuple[float, float] | None = None,
):
    """
    Plot metric distributions across imbalance levels for multiple datasets.

    Rows = datasets
    Columns = models
    Each panel = one dataset-model combination
    """

    if isinstance(datasets, str):
        datasets = [datasets]

    required_cols = {"dataset", "imbalance", "model", metric}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    dff = df[
        (df["dataset"].isin(datasets)) &
        (df["model"].isin(models))
    ].copy()

    if dff.empty:
        raise ValueError("No rows found for the selected datasets and models.")

    dff = dff.dropna(subset=["dataset", "model", "imbalance", metric])

    # Avoid int/string mismatch in imbalance values
    dff["imbalance"] = dff["imbalance"].astype(str)

    if imbalance_order is None:
        unique_vals = dff["imbalance"].dropna().unique()
        try:
            imbalance_order = sorted(unique_vals, key=lambda x: float(x))
        except Exception:
            imbalance_order = sorted(unique_vals)
    else:
        imbalance_order = [str(x) for x in imbalance_order]

    imbalance_order = [
        x for x in imbalance_order
        if x in dff["imbalance"].unique()
    ]

    if not imbalance_order:
        raise ValueError("No valid imbalance levels found.")

    models_present = [m for m in models if m in dff["model"].unique()]
    if not models_present:
        raise ValueError("None of the requested models are present in the filtered data.")

    nrows = len(datasets)
    ncols = len(models_present)

    fig_width = figsize_per_panel[0] * ncols
    fig_height = figsize_per_panel[1] * nrows

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        sharey=sharey,
        squeeze=False,
    )

    palette_full = sns.color_palette("Greys", n_colors=len(imbalance_order) + 2)[2:]

    for row_idx, dataset in enumerate(datasets):
        for col_idx, model in enumerate(models_present):

            ax = axes[row_idx, col_idx]

            sub = dff[
                (dff["dataset"] == dataset) &
                (dff["model"] == model)
            ].copy()

            sub = sub.dropna(subset=[metric, "imbalance"])

            valid_levels = [
                x for x in imbalance_order
                if x in sub["imbalance"].unique()
            ]

            if sub.empty or len(valid_levels) == 0:
                ax.set_visible(False)
                continue

            palette = palette_full[:len(valid_levels)]

            sns.boxplot(
                data=sub,
                x="imbalance",
                y=metric,
                order=valid_levels,
                palette=palette,
                linewidth=1.0,
                fliersize=2,
                ax=ax,
            )

            stats = (
                sub.groupby("imbalance")[metric]
                .agg(
                    median="median",
                    q1=lambda x: x.quantile(0.25),
                    q3=lambda x: x.quantile(0.75),
                    mean="mean",
                )
                .reindex(valid_levels)
            )
            stats["iqr"] = stats["q3"] - stats["q1"]

            if show_means:
                ax.scatter(
                    range(len(valid_levels)),
                    stats["mean"].values,
                    color="black",
                    s=12,
                    zorder=3,
                    marker="o",
                )

            trans = ax.get_xaxis_transform()

            for x_pos, imbalance in enumerate(valid_levels):
                row = stats.loc[imbalance]

                if pd.notna(row["median"]):
                    ax.text(
                        x_pos,
                        -0.22,
                        f"Med={row['median']:.3f}",
                        transform=trans,
                        ha="center",
                        va="top",
                        fontsize=6,
                    )

                    ax.text(
                        x_pos,
                        -0.29,
                        f"IQR={row['iqr']:.3f}",
                        transform=trans,
                        ha="center",
                        va="top",
                        fontsize=6,
                    )

                    ax.text(
                        x_pos,
                        -0.36,
                        f"Avg={row['mean']:.3f}",
                        transform=trans,
                        ha="center",
                        va="top",
                        fontsize=6,
                    )

            if row_idx == 0:
                ax.set_title(model, fontsize=9)

            if col_idx == 0:
                ax.set_ylabel(f"{dataset}\n{ylabel}", fontsize=9)
            else:
                ax.set_ylabel("")

            ax.set_xlabel(xlabel, fontsize=9, labelpad=34)

            ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
            ax.tick_params(axis="x", labelrotation=0, labelsize=7, pad=2)
            ax.tick_params(axis="y", labelsize=8)

            if ylim is not None:
                ax.set_ylim(ylim)

            sns.despine(ax=ax)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30, hspace=0.70, wspace=0.25)

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


from pathlib import Path
import pandas as pd


def load_xai_file(
    data_dir,
    dataset_name,
    dataset_variant=None,
    model_choice=None,
    file_type="summary",
    normalized=False,
):
    """
    Load one or multiple saved XAI result files.

    Parameters
    ----------
    dataset_name : str | list[str]
        Single dataset name or list of datasets.
        Example:
            "german"
            ["german", "australian", "taiwan"]

    Returns
    -------
    pd.DataFrame
        If multiple datasets are provided, all files are appended.
    """

    data_dir = Path(data_dir)

    # ---------------------------------------------------
    # If list of datasets -> load each one and append
    # ---------------------------------------------------
    if isinstance(dataset_name, (list, tuple, set)):

        all_dfs = []

        for ds in dataset_name:
            df = load_xai_file(
                data_dir=data_dir,
                dataset_name=ds,
                dataset_variant=dataset_variant,
                model_choice=model_choice,
                file_type=file_type,
                normalized=normalized,
            )

            df["dataset"] = ds
            all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True)

    # ===================================================
    # SINGLE DATASET LOGIC
    # ===================================================

    merged_prefix = "all_models_"

    if file_type.startswith(merged_prefix):

        base_type = file_type.replace(merged_prefix, "")
        suffix = "_normalized" if normalized else ""

        file_path = (
            data_dir
            / "output"
            / dataset_name
            / "xai"
            / f"{dataset_name}_xai_all_models_{base_type}{suffix}.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return pd.read_csv(file_path)

    # ---------------------------------------------------
    # Standard model files
    # ---------------------------------------------------
    if dataset_variant is None or model_choice is None:
        raise ValueError(
            "dataset_variant and model_choice are required "
            "for model-level files."
        )

    result_dir = (
        data_dir
        / "output"
        / dataset_name
        / "xai"
        / model_choice
        / dataset_variant
        / "metrics"
    )

    file_path = result_dir / f"{dataset_variant}_{model_choice}_{file_type}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


import math
import matplotlib.pyplot as plt
import seaborn as sns


def plot_models_by_imbalance(
    df,
    datasets,
    models,
    imbalance_levels,
    metric="rankstab_local",
    ylabel="Local stability",
    xlabel="Model",
    suptitle=None,
    ylim=None,
):
    """
    Plot model stability by imbalance level for multiple datasets.

    Rows represent datasets.
    Columns represent imbalance levels.
    """

    if isinstance(datasets, str):
        datasets = [datasets]

    nrows = len(datasets)
    ncols = len(imbalance_levels)

    df_plot = df[
        (df["dataset"].isin(datasets)) &
        (df["model"].isin(models)) &
        (df["imbalance"].isin(imbalance_levels))
    ].copy()

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 4.8 * nrows),
        squeeze=False,
    )

    palette = sns.color_palette("Greys", n_colors=len(models) + 2)[2:]

    for row_idx, dataset in enumerate(datasets):
        for col_idx, imbalance in enumerate(imbalance_levels):

            ax = axes[row_idx, col_idx]

            df_temp = df_plot[
                (df_plot["dataset"] == dataset) &
                (df_plot["imbalance"] == imbalance)
            ].copy()

            sns.boxplot(
                data=df_temp,
                x="model",
                y=metric,
                order=models,
                palette=palette,
                linewidth=1.0,
                fliersize=2,
                ax=ax,
            )

            ax.set_title(
                f"{dataset} | Default rate = {imbalance}",
                fontsize=9,
            )

            ax.set_xlabel(xlabel, fontsize=9)

            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            else:
                ax.set_ylabel("")

            if ylim is not None:
                ax.set_ylim(ylim)

            ax.tick_params(axis="x", labelrotation=0, labelsize=7, pad=20)
            ax.tick_params(axis="y", labelsize=8)

            trans = ax.get_xaxis_transform()

            for x_pos, model in enumerate(models):
                values = df_temp.loc[df_temp["model"] == model, metric].dropna()

                if len(values) > 0:
                    q1 = values.quantile(0.25)
                    q3 = values.quantile(0.75)
                    iqr = q3 - q1
                    median = values.median()

                    ax.text(
                        x_pos,
                        -0.16,
                        f"Med={median:.3f}",
                        transform=trans,
                        ha="center",
                        va="top",
                        fontsize=6,
                    )

                    ax.text(
                        x_pos,
                        -0.22,
                        f"IQR={iqr:.3f}",
                        transform=trans,
                        ha="center",
                        va="top",
                        fontsize=6,
                    )

            ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
            sns.despine(ax=ax)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.55, wspace=0.25)

    return fig, axes