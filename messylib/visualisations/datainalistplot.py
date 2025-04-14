import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import seaborn as sns


def plot_data_inalist(
    # --- Required data ---
    mean_np: List[np.ndarray],
    # --- Optional data ---
    x: Optional[List[np.ndarray]] = None,
    std_np: Optional[List[np.ndarray]] = None,
    labels: Optional[List[str]] = None,
    # --- Plot aesthetics ---
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (8, 5),
    linewidth: float | int = 1.5,
    linealpha: float = 0.9,
    fill_between: bool = True,
    # --- Style options ---
    palette: Optional[str] = "viridis",
    style: Optional[str] = "darkgrid",
    font_scale: float = 0.9,
    despine: bool = True,
    # --- Output options ---
    show_plot: bool = True,
    fname: Optional[str] = None,
    savedpi: Optional[int] = 300,  # <-- Added here
):
    """
    Plots mean curves with optional standard deviation shading.

    Args:
        x (list of np.ndarray, optional): X values. If None, uses np.arange for each mean_np[i].
        mean_np (list of np.ndarray): Mean values to plot.
        std_np (list of np.ndarray, optional): Standard deviation values for shading.
        labels (list of str, optional): Labels for each line.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        fill_between (bool): Whether to fill between (mean ± std).
        show_plot (bool): Whether to call plt.show().
        fname (str, optional): If provided, saves the plot to this file.
        savedpi (int, optinal): Control the DPI for saved plot.
    """

    sns.set_theme(style=style, font_scale=font_scale)
    palette = sns.color_palette(palette, n_colors=len(mean_np))

    if x is None:
        x = [np.arange(len(mean)) for mean in mean_np]

    plt.figure(figsize=figsize)

    for i, mean in enumerate(mean_np):
        x_vals = x[i]
        color = palette[i % len(palette)]
        label = labels[i] if labels and i < len(labels) else f"Line {i+1}"
        plt.plot(
            x_vals, mean, label=label, alpha=linealpha, color=color, linewidth=linewidth
        )

        if fill_between and std_np is not None and i < len(std_np):
            std = std_np[i]
            plt.fill_between(x_vals, mean - std, mean + std, alpha=0.3)

    if despine:
        sns.despine(top=True, right=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels:
        plt.legend()

    if fname:
        plt.savefig(fname, bbox_inches="tight", dpi=savedpi)

    if show_plot:
        plt.show()

    plt.close()
