# ============================================================
# Scientific Plotting Utilities
# Color palettes, canvas setup, axis helpers, scatter/curve
# plotting tools, and matplotlib environment configuration.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates  as mdates
import matplotlib.patches
import matplotlib.collections
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit

def show_colors(
        colors: list,
        width: float = 1, height: float = 1,
        hspace: float = 0.05, wspace: float = 0.05,
        fontsize: int = 10,
):
    """
    Render a grid of colored patches labelled with their hex values.

    Args:
        colors   (list)  : List of color strings to display.
        width    (float) : Width of each patch.
        height   (float) : Height of each patch.
        hspace   (float) : Vertical gap between patches.
        wspace   (float) : Horizontal gap between patches.
        fontsize (int)   : Font size for hex labels.

    Returns:
        tuple: (fig, ax)
    """
    ncolors = len(colors)

    # Determine grid dimensions — closest square that fits all colors
    ncol = int(np.floor(np.sqrt(ncolors)))
    nrow = ncol
    while nrow * ncol < ncolors:
        nrow += 1

    xx = np.arange(0, ncol, width  + wspace)
    yy = np.arange(0, nrow, height + hspace)

    fig, ax = setup_canvas(1, 1, figsize=(6, 6))
    patches = []

    for i, xi in enumerate(xx):
        for j, yi in enumerate(yy):
            cnt   = i * nrow + j
            text  = colors[cnt] if cnt < ncolors else None
            color = colors[cnt] if cnt < ncolors else 'None'

            patch = matplotlib.patches.Rectangle(
                (xi, yi), width, height, fill=True, color=color
            )
            ax.add_patch(patch)

            if text:
                ax.text(
                    xi + width / 2, yi + height / 2, text,
                    fontsize=fontsize,
                    horizontalalignment='center',
                    verticalalignment='center',
                )

    ax.add_collection(matplotlib.collections.PatchCollection(patches))
    ax.relim()
    ax.autoscale_view()
    ax.axis('off')
    return fig, ax


# ===========================================================================
# Environment Setup
# ===========================================================================

def init_env(
        fontsize: int            = 14,
        linemarkersize: int      = 2,
        legendtitle_fontsize: int= 14,
        figuresize: tuple        = (10, 6),
        pandas_max_columns: int  = None,
) -> list:
    """
    Configure matplotlib and pandas display settings for scientific work.

    Args:
        fontsize             (int)   : Base font size for all plot text.
        linemarkersize       (int)   : Default marker size for line plots.
        legendtitle_fontsize (int)   : Font size for legend titles.
        figuresize           (tuple) : Default figure size (width, height) in inches.
        pandas_max_columns   (int)   : Max columns shown by pandas; None = unlimited.

    Returns:
        list: Default matplotlib color cycle (10 colors).
    """
    pd.set_option('display.max_columns', pandas_max_columns)
    plt.rcParams.update({
        'font.size'             : fontsize,
        'lines.markersize'      : linemarkersize,
        'legend.title_fontsize' : legendtitle_fontsize,
        'figure.figsize'        : figuresize,
    })


def reset_env() -> None:
    """Reset all pandas and matplotlib settings to their defaults."""
    pd.reset_option('all')
    plt.rcParams.update(plt.rcParamsDefault)


# ===========================================================================
# Canvas & Axes Setup
# ===========================================================================

def setup_canvas(
        nx: int, ny: int,
        figsize: tuple   = (8, 6),
        sharex: bool     = True,
        sharey: bool     = True,
        fontsize: int    = 10,
        labelsize: int   = 10,
        markersize: int  = 2,
        flatten: bool    = True,
        wspace: float    = 0,
        hspace: float    = 0,
        panels: bool     = False,
):
    """
    Create a figure with a grid of subplots, pre-styled for publication.

    Args:
        nx, ny      (int)   : Grid dimensions (rows × columns).
        figsize     (tuple) : Figure size in inches.
        sharex/y    (bool)  : Share x/y axes across subplots.
        fontsize    (int)   : Global font size.
        labelsize   (int)   : Tick label size.
        markersize  (int)   : Default marker size.
        flatten     (bool)  : Return axes as a 1-D array.
        wspace/hspace(float): Horizontal / vertical spacing between subplots.
        panels      (bool)  : Annotate subplots with (a), (b), (c) … labels.

    Returns:
        tuple: (fig, axes) — axes is a single Axes if nx*ny == 1, else an array.
    """
    plt.rcParams.update({'lines.markersize': markersize, 'font.size': fontsize})
    fig, axes = plt.subplots(nx, ny, figsize=figsize, sharex=sharex, sharey=sharey)

    # Normalise to array so downstream code can always iterate
    if nx * ny == 1:
        axes = np.array([axes])
    if flatten:
        axes = axes.flatten()

    # Inward ticks on all subplots
    for ax in axes.flatten():
        ax.tick_params(direction='in', which='both', labelsize=labelsize)

    # Optional panel labels: (a), (b), (c) …
    if panels:
        for i, ax in enumerate(axes):
            ax.text(0.05, 0.8, f'({chr(97 + i)})', transform=ax.transAxes)

    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    return (fig, axes[0]) if len(axes) == 1 else (fig, axes)


def nrow_x_ncols(acnt: int) -> tuple[int, int]:
    """
    Compute the most square grid dimensions that fit `acnt` subplots.

    Args:
        acnt (int): Number of subplots required.

    Returns:
        tuple: (nrows, ncols)
    """
    nc = int(np.ceil(np.sqrt(acnt)))
    nr = nc - 1 if nc * (nc - 1) >= acnt else nc
    return nr, nc


def savefig(fig, savefile: str, dpi: int = 600, bbox_inches: str = 'tight', transparent: bool = False, **kwargs) -> None:
    """
    Save a figure with publication-quality defaults.

    Args:
        fig         : matplotlib Figure object.
        savefile    (str)  : Output file path (extension determines format).
        dpi         (int)  : Resolution in dots per inch.
        bbox_inches (str)  : Bounding box handling ('tight' removes whitespace).
        transparent (bool) : Save with a transparent background.
        **kwargs           : Additional arguments forwarded to fig.savefig.
    """
    fig.savefig(savefile, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent, **kwargs)


# ===========================================================================
# Axis Helpers
# ===========================================================================

def upper_legend(
        ax,
        xloc: float       = 0.5,
        yloc: float       = 1.1,
        ncols: int        = None,
        nrows: int        = None,
        user_labels: list = [],
        user_labels_order: list = [],
        loc: str          = 'upper center',
        framealpha: float = 0.,
        frameon: bool     = False,
):
    """
    Place a de-duplicated legend above (or outside) an axes.

    Supports custom label lists, custom ordering, and row-major reflow
    so that multi-column legends read top-to-bottom before left-to-right.

    Args:
        ax                (Axes)  : Target axes.
        xloc, yloc        (float) : bbox_to_anchor coordinates.
        ncols             (int)   : Number of legend columns.
        nrows             (int)   : If set, reflows labels into this many rows.
        user_labels       (list)  : Override auto-detected labels.
        user_labels_order (list)  : Reorder labels to this sequence.
        loc               (str)   : Legend anchor point string.
        framealpha        (float) : Legend frame transparency.
        frameon           (bool)  : Show legend frame border.

    Returns:
        Axes: The modified axes.
    """
    def _reorder_by_rows(lst, nrows):
        """Reflow a flat list so it reads top-to-bottom in each column."""
        ncols = len(lst) // nrows
        if nrows * ncols != len(lst):
            ncols += 1
        out = []
        for c in range(ncols):
            for r in range(nrows):
                idx = r * ncols + c
                if idx < len(lst):
                    out.append(lst[idx])
        assert len(lst) == len(out), 'ERROR: reorder length mismatch'
        return out, ncols

    handles, labels = ax.get_legend_handles_labels()

    if user_labels:
        labels = user_labels
    if user_labels_order:
        label_map = dict(zip(labels, handles))
        handles   = [label_map[l] for l in user_labels_order]
        labels    = user_labels_order

    if len(handles) != len(labels):
        print('WARNING: handles and labels have unequal lengths.')

    if nrows:
        labels,  ncols = _reorder_by_rows(labels,  nrows)
        handles, ncols = _reorder_by_rows(handles, nrows)

    by_label = dict(zip(labels, handles))
    if not ncols:
        ncols = len(labels)

    if xloc and yloc:
        ax.legend(
            by_label.values(), by_label.keys(),
            loc=loc, framealpha=framealpha, frameon=frameon,
            bbox_to_anchor=(xloc, yloc), ncol=ncols,
        )
    else:
        ax.legend(
            by_label.values(), by_label.keys(),
            loc=loc, framealpha=framealpha, frameon=frameon,
            ncol=ncols, bbox_to_anchor=(0., -0.05, 1., 0.),
            borderaxespad=0, mode='expand',
        )
    return ax


def nticks_prune(ax, which: str = 'x', nbins: int = None, prune: str = None):
    """
    Limit and optionally prune tick marks on an axis.

    Args:
        ax    : Target axes.
        which (str) : 'x' or 'y'.
        nbins (int) : Maximum number of tick intervals; defaults to current count.
        prune (str) : Remove a tick at 'upper', 'lower', 'both', or None.

    Returns:
        Axes: The modified axes.
    """
    if which == 'x':
        if not nbins:
            nbins = len(ax.get_xticklabels())
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=nbins, prune=prune))
    else:
        if not nbins:
            nbins = len(ax.get_yticklabels())
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=nbins, prune=prune))
    return ax


def rotate_ticks(ax, which: str, degree: float) -> None:
    """
    Rotate tick labels on the specified axis.

    Args:
        ax     : Target axes.
        which  (str)  : 'x' or 'y'.
        degree (float): Rotation angle in degrees.
    """
    ax.tick_params(axis=which, rotation=degree)


def shift_axis_label(ax, which: str, x_shift: float, y_shift: float) -> None:
    """
    Manually reposition an axis label using axes-fraction coordinates.

    Args:
        ax              : Target axes.
        which   (str)   : 'x' or 'y'.
        x_shift (float) : Horizontal position in axes fraction.
        y_shift (float) : Vertical position in axes fraction.

    Raises:
        Exception: If `which` is not 'x' or 'y'.
    """
    if which == 'x':
        ax.xaxis.set_label_coords(x_shift, y_shift, transform=ax.transAxes)
    elif which == 'y':
        ax.yaxis.set_label_coords(x_shift, y_shift, transform=ax.transAxes)
    else:
        raise ValueError("which must be 'x' or 'y'.")


def format_axis_datetime(ax, fmt: str = '%m/%Y', which: str = 'x') -> None:
    """
    Apply a strftime format string to datetime tick labels.

    Args:
        ax    : Target axes.
        fmt   (str) : strftime format, e.g. '%Y-%m' or '%d/%m/%Y'.
        which (str) : 'x' or 'y'.
    """
    formatter = mdates.DateFormatter(fmt)
    if which == 'x':
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)


def unify_xylim(ax) -> tuple[float, float]:
    """
    Set identical limits on both axes (useful for 1:1 scatter plots).

    Args:
        ax: Target axes.

    Returns:
        tuple: (vmin, vmax) — the unified limits applied to both axes.
    """
    xylim = np.vstack([ax.get_xlim(), ax.get_ylim()])
    vmin  = xylim[:, 0].min()
    vmax  = xylim[:, 1].max()
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    return vmin, vmax


def add_text(
        ax, x: float, y: float, text: str,
        fontsize: int        = None,
        color: str           = 'k',
        horizontalalignment  = 'left',
        verticalalignment    = 'center',
        if_background: bool  = False,
        bg_facecolor: str    = 'white',
        bg_alpha: float      = 0.2,
        bg_edgecolor: str    = 'None',
):
    """
    Annotate an axes with text in axes-fraction coordinates.

    Args:
        ax, x, y            : Target axes and position (axes fraction 0–1).
        text                (str)   : Annotation string.
        fontsize            (int)   : Font size; None inherits rcParams.
        color               (str)   : Text color.
        horizontalalignment (str)   : 'left', 'center', or 'right'.
        verticalalignment   (str)   : 'top', 'center', or 'bottom'.
        if_background       (bool)  : Draw a filled box behind the text.
        bg_facecolor        (str)   : Background box fill color.
        bg_alpha            (float) : Background box opacity.
        bg_edgecolor        (str)   : Background box border color.

    Returns:
        matplotlib.text.Text: The created text object.
    """
    t = ax.text(
        x, y, text,
        transform=ax.transAxes, color=color, fontsize=fontsize,
        horizontalalignment=horizontalalignment,
        verticalalignment=verticalalignment,
    )
    if if_background:
        t.set_bbox(dict(facecolor=bg_facecolor, alpha=bg_alpha, edgecolor=bg_edgecolor))
    return t


def add_line(
        ax,
        loc: float,
        linestyle: str = '--',
        color: str     = 'k',
        alpha: float   = 0.5,
        direction: str = 'h',
        bmin: float    = 0,
        bmax: float    = 1,
) -> None:
    """
    Draw a horizontal or vertical reference line across an axes.

    Args:
        ax        : Target axes.
        loc       (float) : Position along the relevant axis.
        linestyle (str)   : Line style, e.g. '--', '-.', ':'.
        color     (str)   : Line color.
        alpha     (float) : Line opacity.
        direction (str)   : 'h' / 'horizontal' or 'v' / 'vertical'.
        bmin/bmax (float) : Fractional start/end of the line (0–1).

    Raises:
        Exception: If direction is not recognised.
    """
    if direction.lower() in ('h', 'horizontal'):
        ax.axhline(loc, linestyle=linestyle, color=color, alpha=alpha, xmin=bmin, xmax=bmax)
    elif direction.lower() in ('v', 'vertical'):
        ax.axvline(loc, linestyle=linestyle, color=color, alpha=alpha, ymin=bmin, ymax=bmax)
    else:
        raise ValueError("direction must be 'h', 'horizontal', 'v', or 'vertical'.")


# ===========================================================================
# Legend Helpers
# ===========================================================================

def get_handles_labels(ax) -> tuple:
    """
    Extract legend handles and labels from an axes.

    Args:
        ax: Source axes.

    Returns:
        tuple: (handles, labels)
    """
    return ax.get_legend_handles_labels()


def reorder_labels(handles: list, labels: list, ncol: int) -> tuple:
    """
    Reorder legend handles/labels so they read top-to-bottom per column.

    Pads to a full nrow × ncol grid, transposes, then strips padding.

    Args:
        handles (list) : List of legend handles.
        labels  (list) : Corresponding label strings.
        ncol    (int)  : Desired number of columns.

    Returns:
        tuple: (handles_new, labels_new) — reordered arrays.
    """
    leng = len(labels)
    nrow = int(np.ceil(leng / ncol))
    pad  = nrow * ncol - leng

    # Pad, reshape to (nrow, ncol), transpose → (ncol, nrow), then flatten
    labels_padded  = labels  + ['0'] * pad
    handles_padded = handles + [0]   * pad

    labels_new  = np.array(labels_padded ).reshape(nrow, ncol).T.ravel()
    handles_new = np.array(handles_padded).reshape(nrow, ncol).T.ravel()

    # Strip padding sentinels
    labels_new  = labels_new [np.where(labels_new  != '0')]
    handles_new = handles_new[np.where(handles_new != 0  )]

    return handles_new, labels_new


# ===========================================================================
# List Utilities
# ===========================================================================

def sort_list_by(lista: list, listb: list) -> list:
    """
    Sort `lista` according to the sort order of `listb`.

    Args:
        lista (list): The list to be reordered.
        listb (list): The list whose sorted order drives the reordering.

    Returns:
        list: `lista` sorted by the values of `listb`.
    """
    return [item for _, item in sorted(zip(listb, lista))]


# ===========================================================================
# Scatter Plots
# ===========================================================================

def kde_scatter(
        ax, dfp: pd.DataFrame,
        x_name: str, y_name: str,
        frac: float    = 0.3,
        v_scale: float = 0.1,
        vmin: float    = None,
        vmax: float    = None,
        cmap: str      = 'RdYlBu_r',
) -> None:
    """
    Scatter plot with points coloured by kernel density estimate (KDE).

    Points are sorted so that the densest points are drawn last (on top),
    and a 1:1 reference line is added automatically.

    Args:
        ax              : Target axes.
        dfp  (DataFrame): Source data; must contain `x_name` and `y_name`.
        x_name, y_name  : Column names for x and y variables.
        frac  (float)   : Fraction of rows to sample (speeds up KDE for large data).
        v_scale (float) : Fractional padding added to unified x/y limits.
        vmin, vmax      : Color scale limits for the density colormap.
        cmap  (str)     : Colormap name for density encoding.
    """
    dfp = dfp[[x_name, y_name]].dropna().sample(frac=frac).reset_index(drop=True)
    x   = dfp[x_name]
    y   = dfp[y_name]

    # Compute KDE density at each point
    z   = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))

    # Sort by density so densest points render on top
    idx = z.argsort()
    x, y, z = x.iloc[idx], y.iloc[idx], z[idx]

    ax.scatter(x, y, c=z, s=50, vmin=vmin, vmax=vmax, cmap=cmap)

    # 1:1 reference line spanning the data range
    xl = np.linspace(np.floor(x.min()), np.ceil(x.max()), 1_000)
    ax.plot(xl, xl, ls='-.', color='k')

    # Unify and pad x/y limits symmetrically
    v_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
    v_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
    v_ran = v_max - v_min
    ax.set_xlim(v_min - v_ran * v_scale, v_max + v_ran * v_scale)
    ax.set_ylim(v_min - v_ran * v_scale, v_max + v_ran * v_scale)

