import xarray as xr
import numpy as np
import tools
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

import scipy.interpolate as interp
from datetime import datetime 
from datetime import timedelta



""" Create virtual SLA and SST Observations for OSSE.  WMOP Forecast is used as Nature Run
    Fields of the Nature Run model are considered as the true state of the ocean and observations 
    extracted from it and interpolated in time and space to real observation points.

    Author: Jaime Hernandez Lasheras (jhernandez@socib.es)
    Date of creation: 28-February-2019  
    Last modification: 27-March-2019 (Add observation noise)

    """

obs_path = '/DATA/jhernandez/WMOP_ASSIM/Observations/HFR_all_nudging_Oct2014/'
nr_path = '/home/modelling/data/WMOP/WMOP_FORECAST/Outputs/FORECAST_CMEMS_RESTARTS/forecast_scratch/'

# WMOP MDT
file_mdt = '/data/modelling/WMOP_ASSIM/Inputs/roms_WMOP_HINDCAST_synthetic_201505_201704_mean.nc'
ds_mdt = xr.open_dataset(file_mdt)

date = datetime(2014,9,20)
date_end = datetime(2014,10,27)

while date < date_end:

    t0 = datetime.now()  # time operation

    strdate = date.strftime('%Y%m%d')

    file_wmop = '{0}/roms_WMOP_FORECAST_{1}_his.nc'.format(nr_path, strdate)
    ds_wmop = xr.open_dataset(file_wmop)

    # Load Observations as DataFrame
    obsfile = '{0}assim_obs_SLA_SSH_Argo_HFR_{1}.obs'.format(obs_path, strdate)
    df = tools.read_obsfile(obsfile)

    # Subset Altimetry and SST Observations
    df_sla = df[df['var']=='h'].reset_index(drop=True)
    df_sst = df[df['source']=='GHRSST_JPL'].reset_index(drop=True)

    print('----- DAY {0} -----'.format(date.strftime('%d-%m-%Y')))
    print(' Creating virtual SLA observations from WMOP NR equivalent to Almtimetry observations')
    # Create empty array
    sla_interp = np.array([])

    for i in range(len(df_sla)):

        obs = df_sla.iloc[i]
        date_obs = datetime(obs['year'], obs['month'], obs['day'], obs['hour'], obs['minute'])
        strdate_obs = date.strftime('%Y%m%d')
        
        # Load NR file to extract observations from
        file_wmop = '{0}/roms_WMOP_FORECAST_{1}_his.nc'.format(nr_path, strdate_obs)
        ds_wmop = xr.open_dataset(file_wmop)
        
        # Get time slot to extract obs from
        dt = np.diff(ds_wmop.ocean_time.values) / np.timedelta64(1, 'h')
        dt = int(np.unique(dt))
        idt = obs.hour // dt
        
        # Get SSH
        sla0 = ds_wmop.zeta[idt,:,:].values - ds_mdt.zeta[0,:,:].values
        sla0[np.isnan(sla0)] = 10000
        fz = interp.interp2d(ds_wmop.lon_rho[1,:].values, ds_wmop.lat_rho[:,1].values, sla0, kind='linear')
        
        sla1 = ds_wmop.zeta[idt+1,:,:].values - ds_mdt.zeta[0,:,:].values
        sla1[np.isnan(sla1)] = 10000
        fz1 = interp.interp2d(ds_wmop.lon_rho[1,:].values, ds_wmop.lat_rho[:,1].values, sla1, kind='linear')

        s0 = fz(obs.lon, obs.lat)
        s1 = fz(obs.lon, obs.lat)
        
        # ponderate sla obs with weights
        w1 = ( (obs.hour + obs.minute/60) / dt) - idt
        sla_interp = np.concatenate((sla_interp, w1*s1 + (1-w1)*s0))

    # Substitute Value of altimeter by virtual observations
    df_sla['val'] =  sla_interp
    df_sla = df_sla.dropna().reset_index(drop=True)

    # Add observation Noise
    df_sla['val'] = df_sla['val'] + np.sqrt(df_sla['err']) * np.random.normal(0,1,len(df_sla['err']))

    # Load NR file to extract observations from
    file_wmop = '{0}/roms_WMOP_FORECAST_{1}_his.nc'.format(nr_path, strdate)
    ds_wmop = xr.open_dataset(file_wmop)

    # Get time slot to extract obs from
    dt = np.diff(ds_wmop.ocean_time.values) / np.timedelta64(1, 'h')
    dt = int(np.unique(dt))
    hour = 9  # observation time desired
    idt = hour // dt
    
    print(' Creating virtual SST observations from WMOP NR equivalent to GHRSST JPL-MUR observations')

    # Get SSH. Interpolate for two time steps according to weights 
    sst0 = ds_wmop.temp[idt,-1,:,:].values
    sst1 = ds_wmop.temp[idt+1,-1,:,:].values

    # ponderate sla obs with weights
    w1 = ( hour / dt) - idt
    sst = w1*sst1 + (1-w1)*sst0

    sst[np.isnan(sst)] = 10000
    fz = interp.interp2d(ds_wmop.lon_rho[1,:].values, ds_wmop.lat_rho[:,1].values, sst, kind='linear')

    sst_interp = [fz(df_sst.iloc[i].lon, df_sst.iloc[i].lat) for i in range(len(df_sst))]
    sst_interp = np.array(sst_interp)

    sst_interp[sst_interp>35] = np.nan

    # Substitute Value of altimeter by virtual observations
    df_sst['val'] =  sst_interp
    df_sst = df_sst.dropna().reset_index(drop=True)

    # Add observation Noise
    df_sst['val'] = df_sst['val'] + np.sqrt(df_sst['err']) * np.random.normal(0,1,len(df_sst['err']))
    
    # Concatenate SLA and SST obs
    df_obs = pd.concat([df_sla, df_sst])

    # Paths for saving files
    save_path = '/DATA/jhernandez/WMOP_ASSIM/Observations/HFR_OSSE_v2/'
    output_file = '{0}assim_obs_SLA_SST_virtual_{1}.obs'.format(save_path, strdate)
    
    # save DataFrames
    df_obs.to_csv(output_file, header=None, sep=' ', index=False,  float_format='%.6f')
   
    print('File of synthetic observations of SLA and SST for OSSE created for {0}'.format(date.strftime('%d-%m-%Y')))
    print('Lenght of ocean time = {0}.  time step of outputs = {1}h\n'.format(ds_wmop.ocean_time.shape[0], dt))
    date = date + timedelta(days=3)

    tf = datetime.now()
    print(' Tiempo total = {0}\n'.format((tf-t0).total_seconds()))
