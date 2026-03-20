import fsspec
import pandas as pd
import xarray as xr
import rioxarray
import rich.table
import planetary_computer
import pystac_client
from pyresample import geometry
from pyresample.kd_tree import resample_nearest

def get_planetary_catalog(catalog_url="http://planetarycomputer.microsoft.com/api/stac/v1"):
    catalog = pystac_client.Client.open(
        catalog_url,
        modifier=planetary_computer.sign_inplace,
    )
    return catalog

def get_collections_catalog(catalog):
    collections = catalog.get_collections()

    df_cata = []
    for collection in collections:
        df_cata.append([
            collection.id,
            collection.title,
            collection.extra_fields['msft:short_description'],
            [link.href for link in collection.links if link.rel == 'describedby'][0]]
        )

    df_cata = pd.DataFrame(df_cata, columns=['ID', 'title', 'description', 'href']).set_index('ID')

    return df_cata

catalog = get_planetary_catalog()
df_cata = get_collections_catalog(catalog)
df_cata.loc["sentinel-3-synergy-syn-l2-netcdf", 'href']


qc_threshold = 10
scale = 0.01
savefolder = root.joinpath('S3_Europe_test')

for cnt, item in tqdm(enumerate(items)):
    dt = item.datetime.date()
    savefile = savefolder.joinpath(f"{dt.strftime(f'%Y-%m-%d_{cnt}')}.nc")
    if savefile.exists(): continue

    # # ------------------------------------------------------------------------

    # signed_url = planetary_computer.sign(item.assets["syn-oa18-reflectance"].href)
    # image = rioxarray.open_rasterio(signed_url)
    # image = image.rio.reproject("EPSG:4326")

    # # ------------------------------------------------------------------------

    keys = [
        "syn-oa08-reflectance",
        "syn-oa09-reflectance",
        "syn-oa10-reflectance",
        "syn-oa11-reflectance",
        "syn-oa12-reflectance",
        "syn-oa17-reflectance",
        "syn-oa18-reflectance",
    ]

    if xr.open_dataset(fsspec.open(item.assets[keys[0]].href).open()).to_dataframe().dropna().empty: continue

    geo = xr.open_dataset(fsspec.open(item.assets["geolocation"].href).open())

    def read(key: str) -> xr.Dataset:
        dataset = xr.open_dataset(fsspec.open(item.assets[key].href).open())
        dataset = dataset.assign_coords(
            {
                "lat": geo.lat,
                "lon": geo.lon,
            }
        )
        return dataset


    datasets = [read(key) for key in keys]
    dataset = xr.combine_by_coords(datasets, join="exact", combine_attrs="drop_conflicts")
    # for band in ['SDR_Oa08', 'SDR_Oa09', 'SDR_Oa10', 'SDR_Oa11', 'SDR_Oa12', 'SDR_Oa17', 'SDR_Oa18']:
    #     err = dataset[band + '_err']
    #     dataset[band] = dataset[band].where(err < qc_threshold)
    # --------------------------------------------------------------------------
    # dataset = dataset.coarsen(rows = 10, columns = 10, boundary = 'trim').mean()
    # --------------------------------------------------------------------------
    nco = []
    for band in list(dataset.data_vars):
        try:
            lats = dataset['lat'].values    # shape (rows, columns)
            lons = dataset['lon'].values
            data = dataset[band].values

            # Mask out invalid lat/lon
            valid_mask = np.isfinite(lats) & np.isfinite(lons) & np.isfinite(data)
            lats = np.where(valid_mask, lats, np.nan)
            lons = np.where(valid_mask, lons, np.nan)
            data = np.where(valid_mask, data, np.nan)


            swath_def = geometry.SwathDefinition(lons=lons, lats=lats)

            if np.isnan(lons).all() or np.isnan(lats).all():continue
            # Define regular grid (resolution: scale °)
            lat_new = np.arange(np.nanmin(lats), np.nanmax(lats), scale)
            lon_new = np.arange(np.nanmin(lons), np.nanmax(lons), scale)

            lon_grid, lat_grid = np.meshgrid(lon_new, lat_new)

            area_def = geometry.GridDefinition(lons=lon_grid, lats=lat_grid)

            # Resample using nearest neighbor (other methods: bilinear, gaussian, etc.)
            resampled = resample_nearest(
                swath_def, data,
                area_def,
                radius_of_influence=500,  # meters
                fill_value=np.nan
            )

            regridded = xr.DataArray(
                resampled,
                dims=["lat", "lon"],
                coords={"lat": lat_new, "lon": lon_new},
                name=band
            )
            nco.append(regridded)
        except Exception as e:
            print(e)
    nco = xr.merge(nco)
    nco.to_netcdf(savefile)