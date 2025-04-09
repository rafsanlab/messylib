import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_bubble(
    df: pd.DataFrame,
    groupby: list,
    x_col,
    y_col,
    hue_col = None,
    figsize: tuple = (15,3),
    sns_style: str = "darkgrid",
    palette = "rocket_r",
    sizes: tuple = (50, 500),
    alpha: float = 0.8,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    xrotate: int = 90,
    tight_layout: bool = False,
    savepath = None,
    showplot: bool = True,

    ):

    bubble_df = df.groupby(groupby).size().reset_index(name="Count")
    plt.figure(figsize=figsize)
    sns.set_theme()
    sns.set_style(sns_style)
    sns.despine()

    bubble_plot = sns.scatterplot(
        data=bubble_df,
        x=x_col,
        y=y_col,
        size="Count",
        hue=hue_col,
        sizes=sizes,
        palette=palette,
        alpha=alpha,
        edgecolor="black",
        clip_on=False,
    )

    # Add some margin (adjust as needed)
    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(x_min + 0, x_max + 0)
    ax.set_ylim(y_min + 0.2, y_max - 0.2)

    # Improve appearance
    plt.xticks(rotation=xrotate)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    if tight_layout:
        plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    if showplot:
        plt.show()
    plt.close()