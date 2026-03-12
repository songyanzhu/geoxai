import pandas as pd
from pathlib import Path

# Path to the data folder
_data_path = Path(__file__).parent / "data"

# Load csv
meta_FLUXNET2015 = pd.read_csv(_data_path / "meta_FLUXNET2015.csv", index_col=0)
meta_ICOS = pd.read_csv(_data_path / "meta_ICOS.csv", index_col=0)
meta_AmeriFlux = pd.read_csv(_data_path / "meta_AmeriFlux.csv", index_col=0)
meta_TERN_L6 = pd.read_csv(_data_path / "meta_TERN_L6.csv", index_col=0)
meta_JapanFlux2024 = pd.read_csv(_data_path / "meta_JapanFlux2024.csv", index_col=0)

# expose them as module variables
globals().update(meta_FLUXNET2015)
globals().update(meta_ICOS)
globals().update(meta_AmeriFlux)
globals().update(meta_TERN_L6)
globals().update(meta_JapanFlux2024)

# control what `import *` exports
__all__ = ["meta_TERN_L6", "meta_ICOS", "meta_FLUXNET2015", "meta_AmeriFlux", "meta_JapanFlux2024"]