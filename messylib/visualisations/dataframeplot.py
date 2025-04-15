import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional


def plot_df(
    # --- Required data ---
    df: pd.DataFrame,
    x_col: str,
    color_col: Optional[str] = None,
    order: dict = None,
    # --- Plot aesthetics ---
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (8, 5),
    # --- Style options ---
    palette: Optional[list | str] = "viridis",
    style: Optional[str] = "darkgrid",
    font_scale: float = 0.9,
    despine: bool = True,
    # --- Legend options ---
    show_legend: bool = True,
    legend_outside: bool = False,
    # --- Output options ---
    show_plot: bool = True,
    fname: Optional[str] = None,
    savedpi: Optional[int] = 300,
):
    """
    Plots a countplot (bar plot) for categorical distributions.

    Args:
        df (pd.DataFrame): DataFrame containing data to plot.
        x_col (str): Column for x-axis categories.
        color_col (str, optional): Column for hue grouping.
        title (str): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        figsize (tuple): Size of the figure.
        palette (list or str): Seaborn color palette or list of colours.
        style (str): Seaborn style (e.g., "whitegrid", "darkgrid").
        font_scale (float): Font scaling factor.
        despine (bool): Whether to remove top/right spines.
        show_legend (bool): Show legend.
        legend_outside (bool): Move legend outside plot.
        show_plot (bool): Show the plot.
        fname (str): If given, save the plot to this file path.
        savedpi (int): DPI for saving the plot.
    """
    sns.set_theme(style=style, font_scale=font_scale)

    if isinstance(palette, str):
        n_colors = df[color_col].nunique() if color_col else df[x_col].nunique()
        palette = sns.color_palette(palette, n_colors=n_colors)

    plt.figure(figsize=figsize)

    ax = sns.countplot(data=df, x=x_col, hue=color_col, palette=palette, order=order)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if despine:
        sns.despine(top=True, right=True)

    if show_legend and color_col:
        if legend_outside:
            plt.legend(
                title=color_col,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
            )
        else:
            plt.legend(title=color_col)
    else:
        ax.get_legend().remove()

    if fname:
        plt.savefig(fname, bbox_inches="tight", dpi=savedpi)

    if show_plot:
        plt.show()

    plt.close()
