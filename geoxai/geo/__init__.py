import pandas as pd
import geopandas as gpd
from pathlib import Path

# Path to the data folder
_data_path = Path(__file__).parent / "data"

# Load csv
world = gpd.read_file(_data_path / "world_naturalearth_lowres.geojson")
world10m = gpd.read_file(_data_path / "geojson/WB_countries_Admin0_10m.geojson")

# expose them as module variables
globals().update(world)
globals().update(world10m)

# control what `import *` exports
__all__ = ["world", "world10m"]