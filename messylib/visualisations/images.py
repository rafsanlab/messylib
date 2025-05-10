import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import List, Optional, Tuple


def plot_image(
    image: np.ndarray,
    cmap: str | None = "viridis",
    title: str | None = None,
    show_plot: bool = True,
    fname: str | None = None,
    show_axes: bool = False,
) -> None:
    """
    Plots an image with optional saving and display control.

    :param image: 2D (or 3D?) numpy array representing the image to plot.
    :param cmap: Colormap for the image. Defaults to 'viridis'.
    :param title: Optional title for the plot.
    :param show_plot: Whether to display the plot. Defaults to True.
    :param fname: Filename to save the plot. If None, the plot is not saved.
    :param show_axes: Whether to show axes. Defaults to False.
    """
    plt.imshow(image, cmap=cmap)
    if not show_axes:
        plt.axis("off")
    if title is not None:
        plt.title(title, fontsize="medium")
    plt.tight_layout()
    if fname is not None:
        if title is not None:
            plt.savefig(fname, bbox_inches="tight")
        else:
            plt.savefig(fname, bbox_inches="tight", pad_inches=0)
    if show_plot:
        plt.show()
    else:
        plt.close()


# ====================================================================
# ====================================================================


def plot_images_from_folder(
    folder_path: str,
    rows: int = 1,
    cols: int = 1,
    img_format: str = ".png",
    figsize: tuple = (10, 10),
    title_size: int = 10,
    cmap: str = "binary",
    cmap_reverse: bool = False,
    fname: Optional[str] = None,
    save_dpi: int = 200,
    show_plot: bool = True,
):
    """
    Plot images from a folder in a grid layout.

    Args:
        folder_path (str): Path to the folder containing images.
        rows (int): Number of rows in the subplot grid (default: 1).
        cols (int): Number of columns in the subplot grid (default: 1).
        img_format (str): File extension format to filter images (default: '.png').
        figsize (tuple): Size of the figure in inches (default: (10, 10)).
        title_size (int): Font size for image titles (default: 10).
        cmap (str): Colormap to use for displaying images (default: 'binary').
        cmap_reverse (bool): Whether to use the reversed colormap (default: False).
        fname (Optional[str]): File path to save the figure (default: None).
        save_dpi (int): Dots per inch (DPI) for saving the figure (default: 200).
        show_plot (bool): Whether to display the plot (default: True).
    """
    # Validate parameters
    if rows <= 0 or cols <= 0:
        raise ValueError("`rows` and `cols` must be positive integers.")
    if (
        not isinstance(figsize, tuple)
        or len(figsize) != 2
        or not all(isinstance(x, (int, float)) for x in figsize)
    ):
        raise ValueError("`figsize` must be a tuple of two integers or floats.")
    if title_size <= 0:
        raise ValueError("`title_size` must be a positive integer.")
    if fname is not None and not isinstance(fname, str):
        raise ValueError("`fname` must be a string or None.")
    if save_dpi <= 0:
        raise ValueError("`save_dpi` must be a positive integer.")
    if not isinstance(show_plot, bool):
        raise ValueError("`show_plot` must be a boolean value.")

    # Prepare the plot
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    # Fetch and sort images
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(img_format)])

    # Plot each image
    for i, ax in enumerate(axes):
        if i < len(image_files):
            img_path = os.path.join(folder_path, image_files[i])
            img = mpimg.imread(img_path)

            cmap_obj = plt.colormaps.get_cmap(cmap)
            if cmap_reverse:
                cmap_obj = cmap_obj.reversed()

            ax.imshow(img, cmap=cmap_obj)
            ax.set_title(image_files[i], fontsize=title_size)
            ax.axis("off")
        else:
            ax.axis("off")  # Hide extra subplots

    plt.tight_layout()

    # Handle display and saving
    if fname:
        plt.savefig(fname, dpi=save_dpi)
    if show_plot:
        plt.show()

    plt.close(fig)


# ====================================================================
# ====================================================================


def plot_images_from_list(
    pathlist: List[str | np.ndarray],
    array_mode: bool = False,
    stepsize: int = 25,
    rows: int = 5,
    cols: int = 5,
    figsize: Tuple[int, int] = (15, 15),
    set_title: bool = True,
    title_list: list | None = None,
    title_size: int = 10,
    cmap: str | None = None,
    cmap_reverse: bool = False,
    savedir: Optional[str] = None,
    save_dpi: int = 200,
    show_plot: bool = True,
    verbose: bool = True,
):
    """
    Plot images from a list of file paths in batches, and optionally save the plots.

    :param pathlist: List of file paths to images or list of image arrays.
    :param array_mode: Set to True if using list of arrays.
    :param stepsize: Number of images to process per batch. Default is 25.
    :param rows: Number of rows in the subplot grid. Default is 5.
    :param cols: Number of columns in the subplot grid. Default is 5.
    :param figsize: Size of the figure in inches (width, height). Default is (15, 15).
    :param title_size: Font size for image titles. Default is 10.
    :param set_title: Option to set title on each images.
    :param title_list: List of title.
    :param cmap: Colormap to use for displaying images. Default is 'binary'.
    :param cmap_reverse: Whether to use the reversed colormap. Default is False.
    :param savedir: Directory to save the plotted images. If None, images are not saved. Default is None.
    :param save_dpi: Dots per inch (DPI) for saving the figure. Default is 200.
    :param show_plot: Whether to display the plots. Default is True.
    """
    total_imgs = len(pathlist)

    # Iterate through the list in batches
    for i in range(0, total_imgs, stepsize):
        # Slice the current batch
        to_plots = pathlist[i : i + stepsize]
        if verbose:
            print(f"Plotting batch of {len(to_plots)} images...")

        # Create subplot grid
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

        # Plot images
        for idx, ax in enumerate(axes):
            if idx < len(to_plots):
                if not array_mode:
                    img_path = to_plots[idx]
                    img = mpimg.imread(img_path)
                else:
                    img = to_plots[idx]
                    # ax.set_title(os.path.basename(img_path), fontsize=title_size)
                if cmap is not None:
                    cmap_obj = plt.colormaps.get_cmap(cmap)
                    if cmap_reverse:
                        cmap_obj = cmap_obj.reversed()
                    ax.imshow(img, cmap=cmap_obj)
                else:
                    ax.imshow(img)
                ax.axis("off")
            else:
                ax.axis("off")  # Hide empty subplots

            # set title on each axes (images)
            if set_title:
                if title_list:
                    ax.set_title(title_list[idx], fontsize=title_size)
                else:
                    if not array_mode:
                        ax.set_title(os.path.basename(img_path), fontsize=title_size)
                    else:
                        ax.set_title(str(idx), fontsize=title_size)

        plt.tight_layout()

        # Generate filename for saving
        j = str(i).zfill(3)
        k = str(min(i + stepsize, total_imgs)).zfill(3)
        fname = os.path.join(savedir, f"summary-img_{j}-{k}.png") if savedir else None

        # Save or show the plot
        if fname:
            plt.savefig(fname, dpi=save_dpi)
            if verbose:
                print(f"Saved plot to {fname}")
        if show_plot:
            plt.show()

        plt.close(fig)
