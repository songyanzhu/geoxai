import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from .canvas import setup_canvas

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
