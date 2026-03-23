import pickle
import warnings
import zipfile

def quiet():
    warnings.simplefilter('ignore')

def unzip(zip_path, extract_path):
    # Open and extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def load_pickle(p):
    with open(p, "rb") as f:
        ds = pickle.load(f)
    return ds

def dump_pickle(ds, p, large = False):
    with open(p, "wb") as f:
        if large:
            pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(ds, f)