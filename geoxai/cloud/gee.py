import ee
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def init_gee(project_name):
    # Trigger the authentication flow.
    ee.Authenticate()

    # Initialize the library.
    ee.Initialize(project = project_name)


def filter_collection(collection: str, date_range: list, bounds: ee.Geometry) -> str:
    """
    Get a list (collection) of ee images filtered by date range and ee geometry (point or so).
    Input:
    -------
    collection: str, ee collection name
    date_range: list, [start: str, end: str]
    bounds: ee.Geometry, e.g.
        ee.Geometry.Polygon(
        [[
        [minlon, maxlat],
        [maxlon, maxlat],
        [maxlon, minlat],
        [minlon, minlat]
        ]]
        ) 
    Return:
    -------
    A list of ee images (collection)
    """
    collection = ee.ImageCollection(collection)
    collection = collection.filterDate(*date_range)
    collection = collection.filterBounds(bounds)
    return collection.toList(collection.size())

def get_collection_size(collection: ee.ImageCollection) -> int:
    """
    Get the length of collection list
    Input:
    -------
    collecton: ee.ImageCollection
    Return:
    -------
    int, size of collection
    """
    return collection.size().getInfo()

def get_image(collection: ee.ImageCollection, count) -> ee.Image:
    """
    Get the ith image from collection
    Input:
    -------
    collection: ee.ImageCollection, a list of images
    count: index of wanted image in collection
    Return:
    -------
    ee.Image
    """
    image = collection.get(count)
    image = ee.Image(image)
    return image

def get_image_bandnames(image: ee.Image) -> list:
    """
    Get the band names of image
    Input:
    -------
    image: ee.Image
    Return:
    -------
    list, a list of band names
    """
    return image.bandNames().getInfo()

def get_proj(image: ee.Image, band: str, pyfmt: bool):
    """
    Get the projection information of an image
    Input:
    -------
    image: ee.Image
    band: str, band name
    pyfmt: bool, python or ee format of output projection
    Return:
    -------
    Projection of image
    """
    proj = image.select(band).projection()
    if pyfmt:
        proj = proj.getInfo()
    return proj
 
def proj_epsg(epsg: int or str = 4326) -> ee.Projection:
    proj = ee.Projection(f'EPSG:{epsg}')
    return proj
 
def get_scale(image: ee.Image) -> float:
    scale = image.projection().nominalScale().getInfo()
    return scale
 
def get_date(image):
    date = ee.Date(image.get('system:time_start'))
    return date.format('Y-M-d HH:mm:ss.SSSS').getInfo()

def collection2ts(collection, roi, date_range, bandnames, scale, radius=None, frequency=None):
    """
    Extracts a time series of mean values for specified bands from an Earth Engine ImageCollection
    over a given region of interest (ROI) and date range, optionally resampled at a given frequency.

    Args:
        collection (str or ee.ImageCollection): The ImageCollection to sample from.
        roi (list, geopandas.GeoDataFrame, or ee.FeatureCollection): Region of interest.
        date_range (list of str): ['YYYY-MM-DD', 'YYYY-MM-DD'].
        bandnames (list of str): Bands to extract.
        scale (float): Pixel resolution (m).
        radius (float, optional): Buffer (degrees) if roi is a point.
        frequency (str or None): Resampling frequency (e.g., 'M', 'W'), or None to keep original frequency.

    Returns:
        pandas.DataFrame: Time series indexed by datetime.
    """
    import pandas as pd
    import geopandas as gpd
    from datetime import datetime
    import json

    # --- Helper to convert ROI ---
    if isinstance(roi, list):
        lat, lon = roi
        if not radius:
            radius = scale / 1e5 * 2
        roi = ee.Geometry.Rectangle([
            lon - radius, lat - radius,
            lon + radius, lat + radius
        ])
    elif isinstance(roi, gpd.GeoDataFrame):
        roi = ee.FeatureCollection(json.loads(roi.dissolve().to_json())["features"])
    else:
        assert isinstance(roi, ee.FeatureCollection), 'ROI must be list, GeoDataFrame, or ee.FeatureCollection'

    # --- Load and filter collection ---
    if isinstance(collection, str):
        collection = ee.ImageCollection(collection)
    else:
        assert isinstance(collection, ee.ImageCollection), 'Invalid collection type'

    start_date, end_date = date_range
    collection = collection.filterBounds(roi).filterDate(start_date, end_date)

    # --- Original Frequency Mode ---
    if frequency is None:
        def extract_image_info(image):
            image = ee.Image(image).clip(roi)
            date = image.date().format('YYYY-MM-dd')
            stats = image.select(bandnames).reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=scale
            )
            return image.set('Info', ee.List([date]).cat([stats.get(b) for b in bandnames]))

        collection = collection.map(extract_image_info)
        result = collection.aggregate_array('Info').getInfo()

        df = pd.DataFrame(result, columns=['DATETIME'] + bandnames)
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        return df.set_index('DATETIME')

    # --- Aggregated Frequency Mode ---
    else:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        intervals = pd.date_range(start=start_dt, end=end_dt, freq=frequency)

        def reduce_interval(start, end):
            subset = collection.filterDate(start, end)
            image = subset.mean().set('system:time_start', ee.Date(start).millis())
            stats = image.select(bandnames).reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=scale
            )
            return ee.Feature(None, stats).set('date', start)

        features = []
        for i in range(len(intervals) - 1):
            start = intervals[i].strftime('%Y-%m-%d')
            end = intervals[i + 1].strftime('%Y-%m-%d')
            features.append(reduce_interval(start, end))

        fc = ee.FeatureCollection(features)
        timestamps = fc.aggregate_array('date').getInfo()
        data = [fc.aggregate_array(b).getInfo() for b in bandnames]

        df = pd.DataFrame(dict(zip(bandnames, data)))
        df['DATETIME'] = pd.to_datetime(timestamps)
        return df.set_index('DATETIME')


def image2points(image, dfi, scale, longitude = 'longitude', latitude = 'latitude', batch_size = 100):
    from tqdm import tqdm
    dfi = dfi.copy().reset_index()

    # Split DataFrame into batches using np.array_split (handles uneven splits)
    df_batches = np.array_split(dfi, np.ceil(len(dfi) / batch_size))

    # Initialize lists to store results
    results = []

    # Process each batch
    for batch in tqdm(df_batches, desc = 'iterating batches...'):
        if batch.empty: continue

        points = ee.FeatureCollection([
            ee.Feature(
                ee.Geometry.Point(lon_, lat_),
                {**{'longitude': lon_, 'latitude': lat_}, **row.to_dict()}
            )
            for (_, row), (lon_, lat_) in zip(batch.iterrows(), zip(batch[longitude], batch[latitude]))
        ])

        # Sample the image at batch points
        samples = image.sampleRegions(collection = points, scale = scale)

        # Extract values, and index (name of point) from the results
        for feature in samples.getInfo()['features']:
            results.append(pd.Series(feature['properties']).to_frame().T)

    df_results = pd.concat(results, axis = 0)
    return df_results

def gdf2ee(gdf):
    features = []

    for _, row in gdf.iterrows():
        # Convert geometry to GeoJSON and then to EE Geometry
        geom = ee.Geometry(row.geometry.__geo_interface__)

        # Convert attributes to a dictionary
        properties = row.drop('geometry').to_dict()

        # Create an EE Feature
        feature = ee.Feature(geom, properties)
        features.append(feature)

    return ee.FeatureCollection(features)

def image4shape(image, gdf, scale, to_xarray = False, properties = []):
    # Sample the image using the shapefile
    samples = image.sampleRegions(
        collection = gdf2ee(gdf),  # FeatureCollection
        scale = scale,  # Spatial resolution (meters)
        properties = properties,
        geometries = True  # Keep geometry for spatial analysis
    )
    values = samples.getInfo()

    # Extract elevation, slope, and index (name of point) from the results
    dfo = []
    for feature in values['features']:
        dfo.append(
            pd.concat([
                pd.Series(feature['geometry']['coordinates'], index = ['longitude', 'latitude']),
                pd.Series(feature['properties'])
            ]).to_frame().T
        )

    dfo = pd.concat(dfo, axis = 0)
    if to_xarray:
        dfo = dfo.sort_values(by = ['latitude', 'longitude']).set_index(['latitude', 'longitude']).to_xarray()
    return dfo


def gee2drive(image, roi, name, folder, scale, crs='EPSG:4326'):
    if not type(roi) == ee.geometry.Geometry:
        minlon = roi[0]
        maxlon = roi[2]
        minlat = roi[1]
        maxlat = roi[3]
        roi = ee.Geometry.BBox(minlon, minlat, maxlon, maxlat)

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder=folder,
        region=roi,
        scale=scale,
        maxPixels = 1e13,
        crs=crs
    )
    task.start()


def get_status(stats = True):
    def summarize_status(df):
        # Count operations per state
        status_counts = df['state'].value_counts().to_frame(name='INFO')
    
        # Build summary table
        summary = status_counts.copy()
        summary.loc['TOTAL'] = summary['INFO'].sum()
        if 'SUCCEEDED' in summary.index:
            # Calculate durations in seconds for COMPLETED only
            completed_df = df[df['state'] == 'SUCCEEDED'].copy()
            completed_df['duration_sec'] = (completed_df['endTime'] - completed_df['startTime']).dt.total_seconds()

            # Total time in human-readable form
            total_duration_sec = completed_df['duration_sec'].sum()
            total_duration_min = total_duration_sec / 60
            total_duration_hr = total_duration_sec / 3600

            # Add time stats as separate info
            time_summary = pd.Series({
                # 'total_completed_tasks': len(completed_df),
                # 'total_duration_sec': total_duration_sec,
                'total_duration_min': round(total_duration_min, 2),
                'total_duration_hr': round(total_duration_hr, 2),
                'mean_duration_sec': round(completed_df['duration_sec'].mean(), 2),
                'max_duration_min': round(completed_df['duration_sec'].max() / 60, 2),
                'min_duration_min': round(completed_df['duration_sec'].min() / 60, 2),
            }, name = 'INFO')
            if 'PENDING' in summary.index:
                time_summary['estimated_pending_min'] = round(summary.loc['PENDING', 'INFO'] * completed_df['duration_sec'].mean() / 60, 2)
                time_summary['estimated_pending_hr'] = round(summary.loc['PENDING', 'INFO'] * completed_df['duration_sec'].mean() / 3600, 2)
        else:
            time_summary = pd.Series({}, name = 'INFO')

        return pd.concat([summary, time_summary])

    # Get all operations
    operations = ee.data.listOperations()

    # Extract relevant fields into a flat list of dicts
    rows = []
    for op in operations:
        meta = op.get('metadata', {})
        stages = meta.get('stages', [])
        
        # Handle up to two stages (extendable if needed)
        stage1 = stages[0] if len(stages) > 0 else {}
        stage2 = stages[-1] if len(stages) > 1 else {}

        rows.append({
            'name': op.get('name'),
            'type': meta.get('type'),
            'description': meta.get('description'),
            'state': meta.get('state'),
            'createTime': meta.get('createTime'),
            'startTime': meta.get('startTime'),
            'endTime': meta.get('endTime'),
            'attempt': meta.get('attempt'),
            'progress': meta.get('progress'),
            'projectId': meta.get('projectId'),
            'error': meta.get('error', {}).get('message') if 'error' in meta else None,
            'batchEecuUsageSeconds': meta.get('batchEecuUsageSeconds'),
            # Flatten stage 1 (beginning)
            'stage1_displayName': stage1.get('displayName'),
            'stage1_description': stage1.get('description'),
            'stage1_completeWorkUnits': stage1.get('completeWorkUnits'),
            'stage1_totalWorkUnits': stage1.get('totalWorkUnits'),
            # Flatten stage 2 (final)
            'stage2_displayName': stage2.get('displayName'),
            'stage2_description': stage2.get('description'),
            'stage2_completeWorkUnits': stage2.get('completeWorkUnits'),
            'stage2_totalWorkUnits': stage2.get('totalWorkUnits'),
        })

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(rows).set_index('createTime')
    df['endTime'] = df['endTime'].fillna(df['startTime'])
    df.index = pd.to_datetime(df.index, format='ISO8601')
    df['startTime'] = pd.to_datetime(df['startTime'], format='ISO8601')
    df['endTime'] = pd.to_datetime(df['endTime'], format='ISO8601')
    df.sort_index(ascending=False, inplace=True)

    if stats: 
        return summarize_status(df)
    else:
        return df

def gee2local(image, savefile, scale, roi, user_params = {}, folder = 'output', description = ''):
    import os, requests, zipfile
    warnings.warn("gee2local() is an experimental function. Use with caution.", UserWarning)
    params = {
        "crs": 'EPSG:4326',
        "maxPixels": 1e13,
        # "crs_transform": None,
        # "format": "ZIPPED_GEO_TIFF",
        # "description": None
    }
    params["scale"] = scale
    params["region"] = roi
    params.update(user_params)
    url = image.getDownloadURL(params)
    r = requests.get(url, stream = True, timeout = 300, proxies = None)
    assert r.status_code == 200, "A connection error occurred while downloading."
    assert type(roi) == ee.geometry.Geometry, "`roi` should be an ee.geometry.Geometry object."

    with open(savefile, "wb") as fd:
        for chunk in r.iter_content(chunk_size = 1024):
            fd.write(chunk)

    # ------------------------Unzip------------------------
    savefolder = savefile.parent.joinpath(folder)
    savefolder.mkdir(parents = True, exist_ok = True)
    with zipfile.ZipFile(savefile, 'r') as z:
        z.extractall(savefolder)
    os.remove(savefile)
    # ------------------------Rename------------------------
    for p in savefolder.glob('*.tif'):
        p.rename(p.parent.joinpath(p.stem.replace('download.', description) + '.tif'))
    return savefolder

def get_min_max(dataset, region = [-179.9, -89.9, 179.9, 89.9]):
    assert len(dataset.bandNames().getInfo()) == 1, 'ERROR: Only one band is accepted.'
    region = ee.Geometry.Rectangle(region)

    v_min_max = dataset.reduceRegion(
        reducer=ee.Reducer.percentile([1, 99]),
        geometry=region,
        scale=25000,
        maxPixels=1e13
    )

    return list(v_min_max.getInfo().values())

def generate_named_palette(colormap_name='jet', n_colors=18):
    cmap = plt.get_cmap(colormap_name, n_colors)
    hex_colors = [mcolors.to_hex(cmap(i)) for i in range(n_colors)]
    return hex_colors

def generate_vis_params(dataset, colormap_name='jet', vmin=None, vmax=None):
    if (vmin is None) or (vmax is None):
        vmin, vmax = get_min_max(dataset)

    vis_params = {
        'min': vmin,
        'max': vmax,
        'palette': generate_named_palette(colormap_name=colormap_name)
    }

    return vis_params

def interactive_map(dataset, vis_params, name, attr = 'Google Earth Engine', opacity = 0.5, base_map = 'terrain'):
    import folium

    # Ensure opacity is in visualization parameters
    vis_params["opacity"] = opacity

    # Get a GEE tile layer URL
    map_id_dict = dataset.getMapId(vis_params)
    tile_url = map_id_dict['tile_fetcher'].url_format

    # Create a folium map centered at some location
    m = folium.Map(location = [20, 0], zoom_start=2)

    if base_map == 'satellite':
        # Add satellite base layer (ESRI Satellite)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
            overlay=False
        ).add_to(m)
    elif base_map == 'satellite, Google':
        # Add satellite base layer (Google Satellite)
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            name="Satellite",
            overlay=False
        ).add_to(m)

    # Add GEE layer to folium map with transparency
    folium.raster_layers.TileLayer(
        tiles=tile_url,
        attr=attr,
        name=name,
        overlay=True,  # Ensures it can be toggled
        control=True,
        opacity=opacity  # Set transparency
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Display map
    return m

def get_dataset_metadata(dataset_id):
    """
    Retrieves basic metadata from the first image of a Google Earth Engine (GEE) ImageCollection.

    This function extracts key information such as:
      - Band names
      - Pixel resolution (nominal scale) in meters
      - Coordinate reference system (CRS)
      - Available image property names

    Args:
        dataset_id (str): The GEE ImageCollection ID (e.g., "ECMWF/ERA5/DAILY").

    Returns:
        dict: A dictionary containing:
            - "dataset" (str): The dataset ID.
            - "band_names" (List[str]): List of band names in the first image.
            - "pixel_scale_meters" (float): Nominal scale (resolution) of the first band, in meters.
            - "crs" (str): Coordinate reference system of the first band.
            - "properties" (List[str]): List of property names associated with the first image.

    Example:
        >>> get_dataset_metadata("ECMWF/ERA5/DAILY")
        {
            "dataset": "ECMWF/ERA5/DAILY",
            "band_names": ["mean_2m_air_temperature", "total_precipitation", ...],
            "pixel_scale_meters": 27830.71,
            "crs": "EPSG:4326",
            "properties": ["system:time_start", "system:index", ...]
        }
    """
    # Load collection and get first image
    collection = ee.ImageCollection(dataset_id)
    first_image = collection.first()

    # Get band names
    band_names = first_image.bandNames().getInfo()

    # Get projection info from first band
    band_info = first_image.select(band_names[0]).projection()
    crs = band_info.crs().getInfo()
    nominal_scale = band_info.nominalScale().getInfo()

    # Get other image properties
    properties = first_image.propertyNames().getInfo()

    return {
        "dataset": dataset_id,
        "band_names": band_names,
        "pixel_scale_meters": nominal_scale,
        "crs": crs,
        "properties": properties
    }

# # Example: ERA5 Daily data
# metadata = get_dataset_metadata("ECMWF/ERA5/DAILY")
# for key, value in metadata.items():
#     print(f"{key}: {value}")