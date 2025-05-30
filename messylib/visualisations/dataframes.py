import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Sequence
import numpy as np
import matplotlib.ticker as ticker


def plot_df_cols(
    df: pd.DataFrame,
    x: str | None = None,
    y_cols: Sequence[str] = ("train_loss", "val_loss"),
    colors: list[str] | str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    legend_outside: bool = False,
    show_plot: bool = True,
    seaborn_style: str | None = "whitegrid",
    savedir: str | None = None,
    title_fontsize: int = 14,
    xlabel_fontsize: int = 12,
    ylabel_fontsize: int = 12,
    remove_spines: Sequence[str] | None = None,
    y_lim: float | tuple[float, float] | None = None,
    line_width: float = 2.0,
    line_styles: Sequence[str] | None = None,
    markers: Sequence[str] | None = None,
    marker_size: float = 5.0,
    show_grid: bool = True,
    highlight_spines: Sequence[str] = (),  # e.g., ["left", "bottom"]
    highlight_spines_dict: dict = {"linewidth": 1, "color": "black"},
    show_legend: bool = True,
    save_dpi: int = 150,
) -> None:
    """
    Plots training and validation metrics from a pandas DataFrame using seaborn.

    Args:
        df : pd.DataFrame
            DataFrame containing the training log with metric columns.
        x : str | None
            Column to use as the x-axis. If None, uses the DataFrame index.
        y_cols : Sequence[str]
            List of column names to plot as y-values.
        colors : list[str] | str | None
            Specific color names or seaborn-compatible colormap name.
        xlabel : str | None
            Label for the x-axis. If None, uses `x`.
        ylabel : str | None
            Label for the y-axis. If None, defaults to "Value".
        title : str | None
            Plot title. If None, no title is shown.
        figsize : tuple[int, int]
            Size of the figure in inches.
        legend_outside : bool
            Whether to place the legend outside the plot area.
        show_plot : bool
            Whether to display the plot with plt.show().
        seaborn_style : str | None
            Style to pass to seaborn.set(). If None, style is not changed.
        savedir : str | None
            If not None, saves the figure to this path.
        title_fontsize : int
            Font size for the plot title.
        xlabel_fontsize : int
            Font size for the x-axis label.
        ylabel_fontsize : int
            Font size for the y-axis label.
        remove_spines : Sequence[str]
            List of spines to remove, e.g., ("top", "right").
        y_lim : float | tuple[float, float] | None
            Limits for y-axis. If float, sets upper limit with lower = 0.
        line_width : float
            Line thickness.
        line_styles : Sequence[str] | None
            List of line styles for each metric. If None, defaults to solid lines.
            Each entry should be a valid matplotlib linestyle, such as:
            Example: line_styles = ["-", "--", "-.", ":"]
        markers : Sequence[str] | None
            List of markers for each metric. If None, no markers are used.
            Each entry should be a valid matplotlib marker, such as:
            - "o" (circle)
            - "s" (square)
            - "^" (triangle up)
            - "D" (diamond)
            - "*" (star)
            - "." (point)
            Example: markers = ["o", "s", "^", "D"]
        marker_size : float
            Size of the markers used in the plot (if `markers` are specified).
        show_grid : bool
            Whether to display the grid.
        highlight_spines : Sequence[str]
            List of spines to make thick and black, e.g., ["left", "bottom"].
        highlight_spines_dict : dict
            Highlighted spine line parameters.
        show_legend : bool
            Whether to display the legend. Defaults to True.
        save_dpi : int
            DPI (dots per inch) for saving the figure. Defaults to 150.

    Returns:
        None
    """

    # Create a copy of the DataFrame to avoid modifying the original
    plot_df = df.copy()
    if x is None:
        # If x is not specified, use the DataFrame's index
        plot_df = plot_df.reset_index()
        x = "index"

    # Melt the DataFrame to long format for seaborn plotting
    melted_df = plot_df.melt(
        id_vars=x, value_vars=y_cols, var_name="Metric", value_name="Value"
    )

    # Create the plot figure
    plt.figure(figsize=figsize)
    if seaborn_style:
        # Set the seaborn style if specified
        sns.set(style=seaborn_style)

    # Determine the color palette
    palette = None
    if isinstance(colors, list):
        palette = colors
    elif isinstance(colors, str):
        palette = sns.color_palette(colors, n_colors=len(y_cols))

    # Get the current axes
    ax = plt.gca()
    unique_metrics = melted_df["Metric"].unique()

    # Plot each metric
    for i, metric in enumerate(unique_metrics):
        subset = melted_df[melted_df["Metric"] == metric]
        style = line_styles[i] if line_styles and i < len(line_styles) else None
        marker = markers[i] if markers and i < len(markers) else None
        sns.lineplot(
            data=subset,
            x=x,
            y="Value",
            label=metric,
            color=palette[i] if palette and i < len(palette) else None, # Ensure palette index is within bounds
            linewidth=line_width,
            linestyle=style,
            marker=marker,
            markersize=marker_size,
            ax=ax,
        )

    # Set axis labels and title
    ax.set_xlabel(xlabel if xlabel else x, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel if ylabel else "Value", fontsize=ylabel_fontsize)

    if title:
        ax.set_title(title, fontsize=title_fontsize)

    # Set y-axis limits if specified
    if y_lim is not None:
        if isinstance(y_lim, tuple):
            ax.set_ylim(y_lim)
        else:
            ax.set_ylim((0, y_lim))

    # Control grid visibility
    if not show_grid:
        ax.grid(False)
    else:
        # Ensure grid is shown if show_grid is True (it's enabled by default with some styles)
        ax.grid(True, which="major", axis="both", linestyle="-", linewidth=0.5)
        ax.grid(True, which="minor", axis="both", linestyle=":", linewidth=0.5)


    # Remove specified spines
    if remove_spines:
        for spine in remove_spines:
            if spine in ax.spines: # Check if spine exists before trying to remove
                ax.spines[spine].set_visible(False)

    # Highlight specified spines
    for spine_name in highlight_spines: # Renamed variable for clarity
        if spine_name in ax.spines:
            ax.spines[spine_name].set_linewidth(highlight_spines_dict["linewidth"])
            ax.spines[spine_name].set_color(highlight_spines_dict["color"])

    # Configure tick parameters
    ax.tick_params(
        bottom="on", # Ensure bottom ticks are on
        left="on",   # Ensure left ticks are on
        axis="both",
        direction="out",
        length=5,
        width=1,
        colors="black", # Explicitly set tick color
    )
    ax.tick_params(axis="x", labelsize=xlabel_fontsize)
    ax.tick_params(axis="y", labelsize=ylabel_fontsize)


    # Adjust the frequency of minor ticks
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Handle legend display
    if show_legend:
        if legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            # Adjust layout to make space for the outside legend
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            plt.legend()
            plt.tight_layout() # Apply tight layout if legend is inside
    else:
        if ax.get_legend() is not None: # Remove legend if it exists and show_legend is False
            ax.get_legend().remove()
        plt.tight_layout() # Apply tight layout even if there's no legend


    # Save the figure if a directory is specified
    if savedir:
        plt.savefig(savedir, bbox_inches="tight", dpi=save_dpi) # Use save_dpi

    # Show or close the plot
    if show_plot:
        plt.show()
    else:
        plt.close()


# ==============================================================================


def boxplot_df_cols(
    df: pd.DataFrame,
    x: str,
    y_cols: Sequence[str],
    hue: str | None = None,
    colors: list[str] | str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    legend_outside: bool = False,
    show_plot: bool = True,
    seaborn_style: str | None = "whitegrid",
    savedir: str | None = None,
    title_fontsize: int = 14,
    xlabel_fontsize: int = 12,
    ylabel_fontsize: int = 12,
    remove_spines: Sequence[str] | None = None,
    y_lim: float | tuple[float, float] | None = None,
    show_grid: bool = True,
    highlight_spines: Sequence[str] = (),
    highlight_spines_dict: dict = {"linewidth": 1, "color": "black"},
    showfliers: bool = True,
    show_legend: bool = True, # Added show_legend
    save_dpi: int = 150, # Added save_dpi
) -> None:
    """
    Plots boxplots (with optional subgroups) of specified DataFrame columns.

    Args:
        df : pd.DataFrame
            DataFrame containing the data.
        x : str
            Main categorical variable on the x-axis.
        y_cols : Sequence[str]
            Columns to plot as boxplots (melted and grouped by 'Metric').
        hue : str | None
            Optional subgroup within each x-category (e.g., for subgroups).
        colors : list[str] | str | None
            Specific color names or seaborn-compatible colormap name.
        xlabel : str | None
            Label for the x-axis. If None, uses `x`.
        ylabel : str | None
            Label for the y-axis. If None, defaults to "Value".
        title : str | None
            Plot title. If None, no title is shown.
        figsize : tuple[int, int]
            Size of the figure in inches.
        legend_outside : bool
            Whether to place the legend outside the plot area.
        show_plot : bool
            Whether to display the plot with plt.show().
        seaborn_style : str | None
            Style to pass to seaborn.set(). If None, style is not changed.
        savedir : str | None
            If not None, saves the figure to this path.
        title_fontsize : int
            Font size for the plot title.
        xlabel_fontsize : int
            Font size for the x-axis label.
        ylabel_fontsize : int
            Font size for the y-axis label.
        remove_spines : Sequence[str]
            List of spines to remove, e.g., ("top", "right").
        y_lim : float | tuple[float, float] | None
            Limits for y-axis. If float, sets upper limit with lower = 0.
        show_grid : bool
            Whether to display the grid.
        highlight_spines : Sequence[str]
            List of spines to make thick and black, e.g., ["left", "bottom"].
        highlight_spines_dict : dict
            Highlighted spine line parameters.
        showfliers : bool
            Whether to show outlier dots.
        show_legend : bool
            Whether to display the legend. Defaults to True.
        save_dpi : int
            DPI (dots per inch) for saving the figure. Defaults to 150.

    Returns:
        None
    """
    # Create a copy of the DataFrame to avoid modifying the original
    plot_df = df.copy()

    # Melt data so all metrics go into a single 'Metric' column
    id_vars = [x]
    if hue:
        id_vars.append(hue)

    melted_df = plot_df.melt(
        id_vars=id_vars, value_vars=y_cols, var_name="Metric", value_name="Value"
    )

    # Create the plot figure
    plt.figure(figsize=figsize)
    if seaborn_style:
        # Set the seaborn style if specified
        sns.set(style=seaborn_style)

    # Determine the color palette
    palette = None
    if isinstance(colors, list):
        palette = colors
    elif isinstance(colors, str):
        # For boxplot, if hue is 'Metric' (default when hue is None),
        # we need colors for each metric in y_cols.
        # If hue is another column, we need colors for unique values in that hue column.
        if hue:
            num_colors = len(plot_df[hue].unique()) if hue in plot_df else len(y_cols)
        else:
            num_colors = len(y_cols)
        palette = sns.color_palette(colors, n_colors=num_colors)


    # Create the boxplot
    ax = sns.boxplot(
        data=melted_df,
        x=x,
        y="Value",
        hue=hue if hue else "Metric", # Use 'Metric' as hue if no specific hue is provided
        palette=palette,
        showfliers=showfliers,
    )

    # Set axis labels and title
    ax.set_xlabel(xlabel if xlabel else x, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel if ylabel else "Value", fontsize=ylabel_fontsize)

    if title:
        ax.set_title(title, fontsize=title_fontsize)

    # Set y-axis limits if specified
    if y_lim is not None:
        if isinstance(y_lim, tuple):
            ax.set_ylim(y_lim)
        else:
            ax.set_ylim((0, y_lim))

    # Control grid visibility
    if not show_grid:
        ax.grid(False)
    else:
        ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.5) # Boxplots usually have y-grid
        ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5)


    # Remove specified spines
    if remove_spines:
        for spine in remove_spines:
            if spine in ax.spines:
                ax.spines[spine].set_visible(False)

    # Highlight specified spines
    for spine_name in highlight_spines:
        if spine_name in ax.spines:
            ax.spines[spine_name].set_linewidth(highlight_spines_dict["linewidth"])
            ax.spines[spine_name].set_color(highlight_spines_dict["color"])

    # Configure tick parameters
    ax.tick_params(
        bottom="on",
        left="on",
        axis="both",
        direction="out",
        length=5,
        width=1,
        colors="black",
    )
    ax.tick_params(axis="x", labelsize=xlabel_fontsize)
    ax.tick_params(axis="y", labelsize=ylabel_fontsize)

    # Adjust the frequency of minor ticks (usually more relevant for y-axis in boxplots)
    # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator()) # Often not needed for categorical x
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Handle legend display
    if show_legend:
        legend_title = None
        if hue:
            legend_title = hue
        elif len(y_cols) > 1 : # Only show legend if there are multiple metrics when hue is not used
             legend_title = "Metric"

        if legend_title: # Proceed only if there's something to show in legend
            if legend_outside:
                plt.legend(
                    bbox_to_anchor=(1.05, 1), loc="upper left", title=legend_title
                )
                plt.tight_layout(rect=[0, 0, 0.85, 1])
            else:
                plt.legend(title=legend_title)
                plt.tight_layout()
        else: # No legend needed (e.g. single y_col and no hue)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            plt.tight_layout()
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        plt.tight_layout()


    # Save the figure if a directory is specified
    if savedir:
        plt.savefig(savedir, bbox_inches="tight", dpi=save_dpi)

    # Show or close the plot
    if show_plot:
        plt.show()
    else:
        plt.close()
