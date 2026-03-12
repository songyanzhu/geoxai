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


# ===========================================================================
# Color Palettes
# ===========================================================================

# Default matplotlib color cycle
default_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

# Nature journal palette — multiple alpha levels encoded as hex (#RRGGBBAA)
nature_colors_base = {
    'Cinnabar'      : '#E64B35',
    'Shakespeare'   : '#4DBBD5',
    'PersianGreen'  : '#00A087',
    'Chambray'      : '#3C5488',
    'Apricot'       : '#F39B7F',
    'WildBlueYonder': '#8491B4',
    'MonteCarlo'    : '#91D1C2',
    'Monza'         : '#DC0000',
    'RomanCoffee'   : '#7E6148',
    'Sandrift'      : '#B09C85',
}

nature_colors_01 = ['#E64B3519', '#4DBBD519', '#00A08719', '#3C548819', '#F39B7F19',   # alpha = 0.1
                    '#8491B419', '#91D1C219', '#DC000019', '#7E614819', '#B09C8519']
nature_colors_03 = ['#E64B354C', '#4DBBD54C', '#00A0874C', '#3C54884C', '#F39B7F4C',   # alpha = 0.3
                    '#8491B44C', '#91D1C24C', '#DC00004C', '#7E61484C', '#B09C854C']
nature_colors_05 = ['#E64B357F', '#4DBBD57F', '#00A0877F', '#3C54887F', '#F39B7F7F',   # alpha = 0.5
                    '#8491B47F', '#91D1C27F', '#DC00007F', '#7E61487F', '#B09C857F']
nature_colors_07 = ['#E64B35B2', '#4DBBD5B2', '#00A087B2', '#3C5488B2', '#F39B7FB2',   # alpha = 0.7
                    '#8491B4B2', '#91D1C2B2', '#DC0000B2', '#7E6148B2', '#B09C85B2']
nature_colors_09 = ['#E64B35E5', '#4DBBD5E5', '#00A087E5', '#3C5488E5', '#F39B7FE5',   # alpha = 0.9
                    '#8491B4E5', '#91D1C2E5', '#DC0000E5', '#7E6148E5', '#B09C85E5']
nature_colors_10 = ['#E64B35FF', '#4DBBD5FF', '#00A087FF', '#3C5488FF', '#F39B7FFF',   # alpha = 1.0
                    '#8491B4FF', '#91D1C2FF', '#DC0000FF', '#7E6148FF', '#B09C85FF']

# Convenience alias for the default 0.7-alpha Nature palette
nature_colors = nature_colors_07

# Paul Tol 7-color colorblind-safe palette
colors_blindfriendly = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']


# ===========================================================================
# Colorblind-Friendly Colormaps
# ===========================================================================

def cmap_colorblind_friendliness(option: str = 'paul_tol') -> mcolors.ListedColormap:
    """
    Return a colorblind-friendly ListedColormap.

    Available options:
        'paul_tol'  — Paul Tol's 7-color palette (default)
        'okabe_ito' — Okabe-Ito 8-color palette

    Args:
        option (str): Palette name. Must be 'paul_tol' or 'okabe_ito'.

    Returns:
        mcolors.ListedColormap: The requested colormap.

    Raises:
        Exception: If `option` is not a recognised palette name.

    Example:
        >>> cmap = cmap_colorblind_friendliness()
        >>> plt.scatter(x, y, c=z, cmap=cmap)
    """
    # Okabe-Ito palette — 8 colors as normalised (0–1) RGB tuples
    okabe_ito_colors = [
        (0.902, 0.624, 0.000),  # orange
        (0.000, 0.451, 0.698),  # blue
        (0.835, 0.369, 0.000),  # vermillion
        (0.000, 0.600, 0.500),  # bluish green
        (0.800, 0.475, 0.655),  # reddish purple
        (0.600, 0.600, 0.600),  # gray
        (0.000, 0.000, 0.000),  # black
        (0.941, 0.894, 0.259),  # yellow
    ]
    cmap_okabe_ito = mcolors.ListedColormap(okabe_ito_colors, name='okabe_ito')

    # Paul Tol palette — 7 hex colors
    paul_tol_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    cmap_paul_tol   = mcolors.ListedColormap(paul_tol_colors, name='paul_tol')

    if option == 'paul_tol':
        return cmap_paul_tol
    elif option == 'okabe_ito':
        return cmap_okabe_ito
    else:
        raise ValueError("option must be 'paul_tol' or 'okabe_ito'.")


def custom_cmap(clist: list, cname: str = 'custom_cmap', N: int = 256) -> mcolors.LinearSegmentedColormap:
    """
    Build a continuous LinearSegmentedColormap from a list of colors.

    Args:
        clist (list) : List of colors (hex strings, RGB tuples, or named colors).
        cname (str)  : Internal name for the colormap.
        N     (int)  : Number of discrete levels in the colormap.

    Returns:
        mcolors.LinearSegmentedColormap
    """
    return mcolors.LinearSegmentedColormap.from_list(cname, clist, N=N)


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

def init_sci_env(
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
    return default_colors


def reset_sci_env() -> None:
    """Reset all pandas and matplotlib settings to their defaults."""
    pd.reset_option('all')
    plt.rcParams.update(plt.rcParamsDefault)


# ===========================================================================
# Canvas & Axes Setup
# ===========================================================================

def setup_canvas(
        nx: int, ny: int,
        figsize: tuple   = (8, 5),
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


def line_dot(
        df: pd.DataFrame, ax,
        colors: list        = None,
        legend: bool        = True,
        markercolor: str    = 'white',
        edgecolor: str      = 'k',
        markersize: int     = 20,
        zorder: int         = 10,
):
    """
    Plot dashed lines with hollow dot markers at each data point.

    Args:
        df          (DataFrame) : Data to plot; index = x-axis, columns = series.
        ax                      : Target axes.
        colors      (list)      : Line/marker colors; defaults to `default_colors`.
        legend      (bool)      : Show legend.
        markercolor (str)       : Marker fill color.
        edgecolor   (str)       : Marker edge color.
        markersize  (int)       : Marker size in points².
        zorder      (int)       : Drawing order for markers (above lines).

    Returns:
        Axes: The modified axes.
    """
    if colors is None:
        colors = default_colors

    df.plot(ax=ax, style='--', color=colors, legend=legend)

    for col in df.columns:
        ax.scatter(df.index, df[col], color=markercolor, edgecolor=edgecolor,
                   s=markersize, zorder=zorder)
    return ax
