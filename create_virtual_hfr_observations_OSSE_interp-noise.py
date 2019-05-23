import xarray as xr
import numpy as np
import tools
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import random

import scipy.interpolate as interp
from datetime import datetime 
from datetime import timedelta
import shapefile
from shapely.geometry import Point
from shapely.geometry import Polygon


""" Create virtual HF radar Observations for OSSE from Ibiza, Formentera and two supossed future stations in Denia
    WMOP Forecast is used as Nature Run
    A Shapefile was created as possible coverage area of the hipothetic antennas. Velocity fields are masked and 
    only observations within the coverage area are considered. 
    A percentage of observations within the area are randomly retired as possible antenna gaps and saved for validation 
    
    Author: Jaime Hernandez Lasheras (jhernandez@socib.es)
    Date of creation: 28-March-2019 (Adapted from previous create_virtual_hfr_observations_OSSE.py) 
    
     * Include interpolation to 3km and observation noise. Error dicreased!

    """
coverage_name = 'future'
# Path  to Nature Run
nr_path = '/home/modelling/data/WMOP/WMOP_FORECAST/Outputs/FORECAST_CMEMS_RESTARTS/forecast_scratch/'

# Path to shapefile
shape_path = '/home/jhernandez/Escritorio/hfr_coverage_geodata/'
shape_file = glob(shape_path + '/*' + coverage_name + '*shp')
# Load Shapefile
shape = shapefile.Reader(shape_file[0])

feature = shape.shapeRecords()[0]
first = feature.shape.__geo_interface__ 
coverage = Polygon(feature.shape.points)


date = datetime(2014,9,20)
date_end = datetime(2014,10,24)

while date < date_end:

    strdate = date.strftime('%Y%m%d')

    # Load WMOP file to extract Longitude and Latitude from
    file_wmop = '{0}/roms_WMOP_FORECAST_{1}_his.nc'.format(nr_path, strdate)
    ds_wmop = xr.open_dataset(file_wmop)

    # Get index of bounding box limits
    id1 = np.abs(ds_wmop.lon_u[1,:].values-0).argmin()
    id2 = np.abs(ds_wmop.lon_u[1,:].values-1.4).argmin()
    id3 = np.abs(ds_wmop.lat_u[:,1].values-38.1).argmin()
    id4 = np.abs(ds_wmop.lat_u[:,1].values-39.2).argmin()

    # Extract longitude and latitude for U and V for the area
    lonu = ds_wmop.lon_u[0,id1:id2].values
    latu = ds_wmop.lat_u[id3:id4,0].values
    lonv = ds_wmop.lon_v[0,id1:id2].values
    latv = ds_wmop.lat_v[id3:id4,0].values

    # Create new Grid with 3km resolution
    lon_hfr = np.arange(lonu[0], lonu[-1], 1.5*np.diff(lonu)[0])
    lat_hfr = np.arange(latu[0], latu[-1], 1.5*np.diff(latu)[0])

    # Interp U
    u = ds_wmop.u[:, -1, id3:id4, id1:id2].mean(axis=0).values
    u[np.isnan(u)] = 100000

    fu = interp.interp2d(lonu, latu, u, kind='linear')
    u_interp = fu( lon_hfr, lat_hfr)
    u_interp[u_interp>1000] = np.nan

    # Interp V
    v = ds_wmop.v[:, -1, id3:id4, id1:id2].mean(axis=0).values
    v[np.isnan(v)] = 100000

    fv = interp.interp2d(lonv, latv, v, kind='linear')
    v_interp = fv( lon_hfr, lat_hfr)
    v_interp[v_interp>1000] = np.nan

    # Create grid with HFR lon and lat. And convert to array
    x, y = np.meshgrid(lon_hfr, lat_hfr)
    x = x.reshape(-1)
    y = y.reshape(-1)

    # See  if point are contained within the polygon defined
    mask_u = [coverage.contains(Point(x[i], y[i])) for i in range(len(x))]
    mask_v = [coverage.contains(Point(x[i], y[i])) for i in range(len(x))] 

    # Reshape to 2D
    mask_u = np.array(mask_u).reshape(len(lat_hfr), len(lon_hfr))
    mask_v = np.array(mask_v).reshape(len(lat_hfr), len(lon_hfr))

    u_interp[mask_u==0] = np.nan
    v_interp[mask_v==0] = np.nan


    # Reshape as array
    u_1d = u_interp.reshape(-1)
    lon_u = x[~np.isnan(u_1d)]
    lat_u = y[~np.isnan(u_1d)]
    u_1d = u_1d[~np.isnan(u_1d)]

    v_1d = v_interp.reshape(-1)
    lon_v = x[~np.isnan(v_1d)]
    lat_v = y[~np.isnan(v_1d)]
    v_1d = v_1d[~np.isnan(v_1d)]

    # Extract randomly a certain number of elements
    idu = list(range(np.min([len(u_1d), len(v_1d)])))
    idv = list(range(len(v_1d)))
    random.shuffle(idu)
    random.shuffle(idv)
    # number of elements to extract
    nb = int(np.round(len(u_1d)*0.15))    # 15%

    # Extract randomly values for validation
    u1 = u_1d[idu[1:nb]]
    v1 = v_1d[idu[1:nb]]
    lonu = lon_u[idu[1:nb]]
    lonv = lon_v[idu[1:nb]]
    latu = lat_u[idu[1:nb]]
    latv = lat_v[idu[1:nb]]

    u_1d[idu[1:nb]] = np.nan
    v_1d[idv[1:nb]] = np.nan

    # Get only valid observations
    lon_u = lon_u[np.isnan(u_1d)==0]
    lon_v = lon_v[np.isnan(v_1d)==0]
    lat_u = lat_u[np.isnan(u_1d)==0]
    lat_v = lat_v[np.isnan(v_1d)==0]
    v_1d = v_1d[np.isnan(v_1d)==0]
    u_1d = u_1d[np.isnan(u_1d)==0]

    # Create DataFrame of subset for Validation
    du_val = {'var': 'u', 'source': 'HF_Radar', 'year': date.year, 'month': date.month, 'day': date.day,'hour': date.hour,
            'minute': date.minute,'lon': lonu, 'lat':latu, 'depth': 0.0, 'val':u1, 'err': 0.0025, 'rep': 1}
    dv_val = {'var': 'v', 'source': 'HF_Radar', 'year': date.year, 'month': date.month, 'day': date.day,'hour': date.hour,
            'minute': date.minute,'lon': lonv, 'lat':latv, 'depth': 0.0, 'val':v1, 'err': 0.0025, 'rep': 1}

    dfu_val = pd.DataFrame(data=du_val)    # create dataframe with u and v obs
    dfv_val = pd.DataFrame(data=dv_val)    
    df_val = pd.concat([dfu_val, dfv_val])   # Concatenate


    # Create New DataFrame of Sinthetic Observations
    du_new = {'var': 'u', 'source': 'HF_Radar', 'year': date.year, 'month': date.month, 'day': date.day,'hour': date.hour,
            'minute': date.minute,'lon': lon_u, 'lat':lat_u, 'depth': 0.0, 'val':u_1d, 'err': 0.0025, 'rep': 1}
    dv_new = {'var': 'v', 'source': 'HF_Radar', 'year': date.year, 'month': date.month, 'day': date.day,'hour': date.hour,
            'minute': date.minute,'lon': lon_v, 'lat':lat_v, 'depth': 0.0, 'val':v_1d, 'err': 0.0025, 'rep': 1}

    dfu_new = pd.DataFrame(data=du_new)        # create dataframe with u and v obs
    dfv_new = pd.DataFrame(data=dv_new)
    df_hfr_new = pd.concat([dfu_new, dfv_new])   # Concatenate

    # Add observation Noise
    df_hfr_new['val'] = df_hfr_new['val'] + 0.02 * np.random.normal(0,1,len(df_hfr_new['err']))


    # Paths for saving files
    save_path = '/DATA/jhernandez/WMOP_ASSIM/Observations/HFR_OSSE_v2/'
    output_file = '{0}assim_obs_HFR_{1}_{2}.obs'.format(save_path, coverage_name, strdate)
    output_file_val = '{0}assim_obs_HFR_{1}_validation_{2}.obs'.format(save_path, coverage_name, strdate)

    # save DataFrames
    df_val.to_csv(output_file_val, header=None, sep=' ', index=False,  float_format='%.6f')
    df_hfr_new.to_csv(output_file, header=None, sep=' ', index=False,  float_format='%.6f')

    print('File of synthetic observations of HF Radar on the whole Ibiza Channel created for {0}'.format(date.strftime('%d-%m-%Y')))

    date = date + timedelta(days=3)




