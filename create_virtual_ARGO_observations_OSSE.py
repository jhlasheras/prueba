import xarray as xr
import numpy as np
import tools
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

import pyroms

import scipy.interpolate as interp
from datetime import datetime 
from datetime import timedelta

""" Create virtual ARGO Observations for OSSE equvalent to real observations.  WMOP Forecast is used as Nature Run
    Fields of the Nature Run model are considered as the true state of the ocean and observations extracted from it 
    and interpolated to real observation points. Closest observation times are considered.
    
    Author: Jaime Hernandez Lasheras (jhernandez@socib.es)
    Date of creation: 4-March-2019  
    last modification: 27-March-2019 (Add observation noise)
    """


obs_path = '/DATA/jhernandez/WMOP_ASSIM/Observations/HFR_all_nudging_Oct2014/'
nr_path = '/home/modelling/data/WMOP/WMOP_FORECAST/Outputs/FORECAST_CMEMS_RESTARTS/forecast_scratch/'

obsfile = glob(obs_path + '*obs')
obsfile.sort()

df = tools.read_obsfile(obsfile[0])

strdate = '20140920'
# Define Model file to extract grid from
file_wmop = '{0}/roms_WMOP_FORECAST_{1}_his.nc'.format(nr_path, strdate)
ds_wmop = xr.open_dataset(file_wmop)

# Load Grid
grid = pyroms.grid.get_ROMS_grid('wmop', hist_file=file_wmop, grid_file=file_wmop)
# Get depth values
z_r = grid.vgrid.z_r[:]
lon_wmop = ds_wmop.lon_rho.values[1,:]
lat_wmop = ds_wmop.lat_rho.values[:,1]

# Interpolate WMOP Nature Run to Argo position

date = datetime(2014,9,20)
date_end = datetime(2014,10,27)

while date < date_end:

    t0 = datetime.now()  # time operation

    strdate = date.strftime('%Y%m%d')

    # Load Observations as DataFrame
    obsfile = '{0}assim_obs_SLA_SSH_Argo_HFR_{1}.obs'.format(obs_path, strdate)
    df = tools.read_obsfile(obsfile)

    # Subset Argo Observations by variable measured
    df_t = df[(df['var']=='t') & (df['source']=='ARGO')]
    df_s = df[(df['var']=='s') & (df['source']=='ARGO')]

    # Get unique values of latitude and longitude to identify different profiles
    lonlat = df_t[['lon', 'lat','year', 'month', 'day', 'hour', 'minute']].drop_duplicates().reset_index(drop=True)
    
    # Get id of value of the WMOP grid closest to Argo profile
    id_lon = [np.abs(lon_wmop-lonlat.iloc[i].lon).argmin() for i in range(len(lonlat))] 
    id_lat = [np.abs(lat_wmop-lonlat.iloc[i].lat).argmin() for i in range(len(lonlat))] 

    # Create new DataFrame to fill in wih synthetic interpolated values
    df_argo_new = pd.DataFrame(columns=['var', 'source', 'year', 'month', 'day', 'hour', 'minute', 'lon', 'lat',
        'depth', 'val', 'err', 'rep','val2'])

    for i in range(len(id_lon)):
        
        date_obs = datetime(int(lonlat.iloc[i]['year']), int(lonlat.iloc[i]['month']), int(lonlat.iloc[i]['day']), int(lonlat.iloc[i]['hour']), int(lonlat.iloc[i]['minute']))
        strdate_obs = date_obs.strftime('%Y%m%d')
        
        # Load NR file to extract observations from
        file_wmop = '{0}/roms_WMOP_FORECAST_{1}_his.nc'.format(nr_path, strdate_obs)
        ds_wmop = xr.open_dataset(file_wmop)
        
        # Get time slot to extract obs from
        dt = np.diff(ds_wmop.ocean_time.values) / np.timedelta64(1, 'h')
        dt = int(np.unique(dt))
        idt = date_obs.hour // dt

        # Extract Temperatures for WMOP and Argo profiles
        df1 = df_t[(df_t.lon==lonlat.iloc[i].lon) & (df_t.lat==lonlat.iloc[i].lat)].reset_index(drop=True)
        df2 = df_s[(df_s.lon==lonlat.iloc[i].lon) & (df_s.lat==lonlat.iloc[i].lat)].reset_index(drop=True)

        temp_argo = df1['val'].values
        salt_argo = df2['val'].values
        zeta_temp = df1['depth'].values
        zeta_salt = df2['depth'].values

        # Get model profiles at closest location
        zeta_roms = z_r[:,id_lat[i], id_lon[i]]
        temp_wmop = ds_wmop.temp.values[idt,:, id_lat[i], id_lon[i]]
        salt_wmop = ds_wmop.salt.values[idt,:, id_lat[i], id_lon[i]]

        # Interpolae model to Argo depths
        temp_interp = [np.interp( -z, zeta_roms, temp_wmop)  for z in zeta_temp]
        salt_interp = [np.interp( -z, zeta_roms, salt_wmop)  for z in zeta_salt]

        # Substitute Argo values by synthetic observations in subset DataFrame
        df1['val'] = temp_interp
        df2['val'] = salt_interp

        
        # COncatenate interpolated observations to create new DataFrame
        df_argo_new = pd.concat([df_argo_new, df1, df2], sort=False).reset_index(drop=True)

    # Add observation Noise
    df_argo_new['val'] = df_argo_new['val'] + np.sqrt(df_argo_new['err']) * np.random.normal(0,1,len(df_argo_new['err']))

    # Paths for saving files
    save_path = '/DATA/jhernandez/WMOP_ASSIM/Observations/HFR_OSSE_v2/'
    output_file = '{0}assim_obs_ARGO_virtual_{1}.obs'.format(save_path, strdate)
    
    # save DataFrames
    df_argo_new.to_csv(output_file, header=None, sep=' ', index=False,  float_format='%.6f')

    print('File of synthetic observations of ARGO equivalents for OSSE created for {0}'.format(date.strftime('%d-%m-%Y')))
    print('Lenght of ocean time = {0}.  time step of outputs = {1}h\n'.format(ds_wmop.ocean_time.shape[0], dt))
    date = date + timedelta(days=3)

    tf = datetime.now()
    print(' Tiempo total = {0}\n'.format((tf-t0).total_seconds()))