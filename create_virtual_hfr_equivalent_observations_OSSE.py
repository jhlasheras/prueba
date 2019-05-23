import xarray as xr
import numpy as np
import tools
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

import scipy.interpolate as interp
from datetime import datetime 
from datetime import timedelta

""" Create virtual HF radar Observations for OSSE equvalent to real observations.  WMOP Forecast is used as Nature Run
    Fields of the Nature Run model are considered as the true state of the ocean and observations extracted from it 
    and interpolated to real observation points. Daily mean velocity fields are considered.
    
    Author: Jaime Hernandez Lasheras (jhernandez@socib.es)
    Date of creation: 28-February-2019  
    """

obs_path = '/DATA/jhernandez/WMOP_ASSIM/Observations/HFR_all_nudging_Oct2014/'
nr_path = '/home/modelling/data/WMOP/WMOP_FORECAST/Outputs/FORECAST_CMEMS_RESTARTS/forecast_scratch/'


date = datetime(2014,9,20)
date_end = datetime(2014,10,27)

while date < date_end:

    t0 = datetime.now()  # time operation

    strdate = date.strftime('%Y%m%d')

    file_wmop = '{0}/roms_WMOP_FORECAST_{1}_his.nc'.format(nr_path, strdate)
    ds_wmop = xr.open_dataset(file_wmop)

    # Get time slot to extract obs from
    dt = np.diff(ds_wmop.ocean_time.values) / np.timedelta64(1, 'h')
    dt = int(np.unique(dt))

    # Load Observations as DataFrame
    obsfile = '{0}assim_obs_SLA_SSH_Argo_HFR_{1}.obs'.format(obs_path, strdate)
    df = tools.read_obsfile(obsfile)

    df_uhfr = df[(df['source']=='HF_Radar') & (df['var']=='u')].reset_index(drop=True)
    df_vhfr = df[(df['source']=='HF_Radar') & (df['var']=='v')].reset_index(drop=True)


    # Interp U
    u = ds_wmop.u[:,-1,:,:].mean(axis=0).values
    u[np.isnan(u)] = 100000

    fu = interp.interp2d(ds_wmop.lon_u[1,:].values, ds_wmop.lat_u[:,1].values, u, kind='linear')
    u_interp = [fu(df_uhfr.iloc[i].lon, df_uhfr.iloc[i].lat) for i in range(len(df_uhfr))]
    u_interp = np.array(u_interp)
    u_interp[u_interp>1000] = np.nan
    df_uhfr['val'] = u_interp


    # Interp V
    v = ds_wmop.v[:,-1,:,:].mean(axis=0).values
    v[np.isnan(v)] = 100000

    fv = interp.interp2d(ds_wmop.lon_v[1,:].values, ds_wmop.lat_v[:,1].values, v, kind='linear')
    v_interp = [fv(df_vhfr.iloc[i].lon, df_vhfr.iloc[i].lat) for i in range(len(df_vhfr))]
    v_interp = np.array(v_interp)
    v_interp[v_interp>1000] = np.nan
    df_vhfr['val'] = v_interp

    # Concatenate SLA and SST obs
    df_hfr = pd.concat([df_uhfr, df_vhfr])
    df_hfr = df_hfr.dropna().reset_index(drop=True)

    # Paths for saving files
    save_path = '/DATA/jhernandez/WMOP_ASSIM/Observations/HFR_OSSE/'
    output_file = '{0}assim_obs_HFR_equivalent_virtual_{1}.obs'.format(save_path, strdate)
    
    # save DataFrames
    df_hfr.to_csv(output_file, header=None, sep=' ', index=False,  float_format='%.6f')
   
    print('File of synthetic observations of HFR equivalents for OSSE created for {0}'.format(date.strftime('%d-%m-%Y')))
    print('Lenght of ocean time = {0}.  time step of outputs = {1}h\n'.format(ds_wmop.ocean_time.shape[0], dt))
    date = date + timedelta(days=3)

    tf = datetime.now()
    print(' Tiempo total = {0}\n'.format((tf-t0).total_seconds()))