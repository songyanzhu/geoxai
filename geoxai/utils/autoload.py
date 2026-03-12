# ============================================================
# geosai
# Basic scientific computing imports and optional auto-install
# of geospatial / ML dependencies for interactive sessions.
# ============================================================

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import gc
import importlib
import os
import pickle
import shutil
import subprocess
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Third-party core
# ---------------------------------------------------------------------------
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local submodules  (note: .sciplt is imported once — duplicate removed)
# ---------------------------------------------------------------------------
from .commons import *
# from .sciplt import *
# from .utils import *


# ===========================================================================
# Optional dependency auto-installer
# ===========================================================================

def _install(package: str) -> None:
    """Install a PyPI package into the current Python environment."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def _install_github(repo_url: str) -> None:
    """Install a package directly from a GitHub repository URL via pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", repo_url])


def _try_import(
    import_name: str,
    pip_name: str,
    alias: str | None = None,
    extra_pip: list[str] | None = None,
):
    """
    Attempt to import a module by name; install it via pip if missing,
    and optionally assign it to an alias in the global namespace.

    Args:
        import_name (str)       : The module name used in `import <name>`.
                                  Example: 'xarray'.
        pip_name    (str)       : The PyPI package name used for pip install.
                                  Usually the same as import_name.
                                  Example: 'xarray'.
        alias       (str, optional) : Optional name to assign the module to in
                                  globals(), simulating `import module as alias`.
                                  Example: alias='xr' for xarray.
        extra_pip   (list[str], optional) : Additional PyPI packages to install
                                  alongside the main one (e.g., ['netCDF4']).

    Returns:
        module : The imported module object. Can be used directly if no alias
                 is provided.
    """
    try:
        module = importlib.import_module(import_name)
    except ModuleNotFoundError:
        print(f"Module '{import_name}' not found — geosai is installing '{pip_name}' for you...")
        _install(pip_name)
        if extra_pip:
            for pkg in extra_pip:
                _install(pkg)
        module = importlib.import_module(import_name)

    if alias:
        globals()[alias] = module

    return module

def _try_install_github(repo_url: str, fallback_pip: str) -> None:
    """
    Install a package from GitHub; fall back to PyPI on failure.

    Args:
        repo_url    (str): Full GitHub URL, e.g. 'git+https://github.com/...'
        fallback_pip(str): PyPI package name to use if GitHub install fails.
    """
    try:
        _install_github(repo_url)
    except Exception as e:
        print(f"{e} — falling back to PyPI install of '{fallback_pip}'...")
        _install(fallback_pip)


# ---------------------------------------------------------------------------
# Prompt user
# ---------------------------------------------------------------------------
_response = input("Auto-install optional geospatial AI packages? (y/n): ").strip().lower()

if not _response:
    raise ValueError('No input received. Please enter "y" or "n".')

_choice = _response[0]

if _choice == "y":
    # --- Geospatial stack ---
    rio = _try_import("rasterio",  "rasterio")
    gpd = _try_import("geopandas", "geopandas")
    xr = _try_import("xarray",    "xarray")
    rxr = _try_import("rioxarray", "rioxarray")

    # --- Visualisation ---
    sns = _try_import("seaborn", "seaborn")

    # # --- Custom GitHub packages (with PyPI fallbacks) ---
    # _try_install_github("git+https://github.com/soonyenju/scigeo.git", "scigeo")
    # from scigeo import *

    # _try_install_github("git+https://github.com/soonyenju/scieco.git", "scieco")
    # from scieco import *

    # _try_install_github("git+https://github.com/soonyenju/sciml.git",  "sciml")
    # from sciml import *

elif _choice == "n":
    print("Auto-installation skipped. Ensure all required packages are installed.")

else:
    raise ValueError('Invalid input. Please enter "y" or "n".')