import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm

def plot_hpo_grid(df, 
                 x_param='hidden_size', y_param='num_blocks', z_metric='val. f1', 
                 col_param=None, row_param=None, 
                 fixed_params=None, 
                 share_z=True, use_cubic=True,
                 show_peak=True, peak_color='red', peak_fontsize=10,
                 label_fontsize=9, tick_count=4,
                 super_title=None, title_pos='top',
                 show_contour=True,
                 fname=None,
                 show_plot=True,
                 ): # New: Show contour at the floor
    
    df_plot = df.copy()
    if fixed_params:
        for k, v in fixed_params.items():
            df_plot = df_plot[df_plot[k].astype(str) == str(v)]

    cols = [None] if col_param is None else sorted(df_plot[col_param].unique())
    rows = [None] if row_param is None else sorted(df_plot[row_param].unique())
    n_cols, n_rows = len(cols), len(rows)

    if share_z:
        agg = df_plot.groupby([x_param, y_param] + ([col_param] if col_param else []) + ([row_param] if row_param else []))[z_metric].mean()
        z_min, z_max = agg.min(), agg.max()
        margin = (z_max - z_min) * 0.1
        z_min, z_max = z_min - margin, z_max + margin

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    surfaces = []

    plot_idx = 1
    for r_idx, r_val in enumerate(rows):
        for c_idx, c_val in enumerate(cols):
            subset = df_plot.copy()
            if r_val is not None: subset = subset[subset[row_param] == r_val]
            if c_val is not None: subset = subset[subset[col_param] == c_val]
            
            if subset.empty:
                plot_idx += 1
                continue

            pivot = subset.pivot_table(index=y_param, columns=x_param, values=z_metric, aggfunc='mean')
            x_raw, y_raw = pivot.columns.values.astype(float), pivot.index.values.astype(float)
            
            xi = np.linspace(x_raw.min(), x_raw.max(), 40)
            yi = np.linspace(y_raw.min(), y_raw.max(), 40)
            Xi, Yi = np.meshgrid(xi, yi)
            
            points = np.array([(x, y) for y in y_raw for x in x_raw])
            values = pivot.values.flatten()
            Zi = griddata(points, values, (Xi, Yi), method='cubic' if use_cubic else 'linear')

            ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
            ax.set_box_aspect(None, zoom=1)
            
            # --- Surface Plot ---
            surf = ax.plot_surface(Xi, Yi, Zi, cmap='viridis', alpha=0.8, edgecolor='none',
                                   vmin=z_min if share_z else None, vmax=z_max if share_z else None)
            surfaces.append(surf)

            # --- Optional: Contour at the bottom ---
            if show_contour:
                offset = z_min if share_z else np.nanmin(Zi)
                ax.contour(Xi, Yi, Zi, zdir='z', offset=offset, cmap='viridis', alpha=0.5)

            # --- Peak Annotation ---
            if show_peak:
                max_idx = np.unravel_index(np.nanargmax(Zi), Zi.shape)
                px, py, pz = Xi[max_idx], Yi[max_idx], Zi[max_idx]
                ax.scatter([px], [py], [pz], color=peak_color, s=40, edgecolors='white', zorder=100)
                ax.text(px, py, pz + (pz*0.02), f"{pz:.3f}", color=peak_color, 
                        fontsize=peak_fontsize, fontweight='bold', ha='center')

            # --- Labels & Ticks ---
            ax.set_xlabel(x_param, fontsize=label_fontsize)
            ax.set_ylabel(y_param, fontsize=label_fontsize)
            ax.set_zlabel(z_metric, fontsize=label_fontsize)
            ax.xaxis.set_major_locator(plt.MaxNLocator(tick_count))
            ax.yaxis.set_major_locator(plt.MaxNLocator(tick_count))
            ax.zaxis.set_major_locator(plt.MaxNLocator(tick_count))
            ax.tick_params(labelsize=label_fontsize-2)

            if share_z: ax.set_zlim(z_min, z_max)
            
            title_str = ""
            if row_param: title_str += f"{row_param}:{r_val}"
            if col_param: title_str += f" | {col_param}:{c_val}"            
            ax.set_title(title_str, fontsize=label_fontsize)

            if not share_z:
                fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10)
            
            plot_idx += 1

    # --- Super Title Positioning ---
    if super_title:
        title_font = {'fontsize': label_fontsize + 6, 'fontweight': 'regular'}
        if title_pos == 'top':
            fig.suptitle(super_title, **title_font, y=0.98)
        elif title_pos == 'right':
            # x=0.98 is the far right
            fig.text(0.99, 0.5, super_title, **title_font, rotation=270, va='center', ha='right')
        elif title_pos == 'left':
            # x=0.02 is the far left
            fig.text(0.01, 0.5, super_title, **title_font, rotation=90, va='center', ha='left')

    left_m = 0.12 if title_pos == 'left' else 0.07
    right_m = 0.99 if title_pos == 'right' else 0.7
    top_m = 0.90 if title_pos == 'top' else 0.95
    bottom_m = 0.18 if share_z else 0.10
    
    fig.subplots_adjust(left=left_m, right=right_m, top=top_m, bottom=bottom_m, wspace=0.005, hspace=0.2)

    if share_z:
            # Calculate dynamic dimensions
            plot_width = right_m - left_m
            cbar_width = plot_width * 0.25  # Colourbar takes up 50% of the total grid width
            cbar_left = left_m + (plot_width - cbar_width) / 2
            cbar_bottom = bottom_m * 0.6
            cbar_height = 0.015            # Keep a fixed slim height

            cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            fig.colorbar(surfaces[0], cax=cbar_ax, orientation='horizontal', label=z_metric)
    # plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=500, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

