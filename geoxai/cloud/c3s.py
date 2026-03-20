import cdsapi

for yearmonth in pd.date_range('2019-11-01', '2024-01-01', freq = '1m'):
    year = str(yearmonth.year)
    month = str(yearmonth.month).zfill(2)

    c = cdsapi.Client(url = "https://cds.climate.copernicus.eu/api/v2", key = 'your-key', verify = 0)

    c.retrieve(
        'reanalysis-era5-land-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                '2m_temperature', 'soil_temperature_level_1', 'surface_net_solar_radiation',
                'surface_net_thermal_radiation', 'surface_pressure', 'surface_solar_radiation_downwards',
                'total_precipitation',
            ],
            'year': year,
            'month': month,
            'time': '00:00',
            'format': 'netcdf',
        },
        root_proj.joinpath(f'ERA5/land-{year}-{month}.nc'))