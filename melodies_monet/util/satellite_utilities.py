# SPDX-License-Identifier: Apache-2.0
#
# File started by Maggie Bruckner. 
# Contains satellite specific pairing operators
import numpy as np
import pandas as pd
import xarray as xr
import stratify

def vertical_regrid(input_press, input_values, output_press):
    '''
    This function uses interp1d to regrid vertical layers in a 3D array
    
    Function requires:
        input_press = input pressure levels in hPa and same dimensions as input_values (lon, lat, alt)
        input_values = Dataarray of input values to be regridded (lon, lat, alt)
        output_press = output pressure levels in hPa, dimensions are the same as input values, except for the altitude (lon, lat, newalt)
        
    Function Returns:
        regrid_array = the data regridded to the new pressure levels

    '''
    from scipy import interpolate
    
    out_array = np.full_like(output_press,np.nan)
    for y in range (input_press.shape[0]):
        # Longitude values
        for x in range (input_press.shape[1]):
            xx = input_press[y,x,:]
            yy = input_values[y,x,:]
            xnew = output_press[y,x,:]
            f = interpolate.interp1d(xx, yy, fill_value="extrapolate")

            out_array[y,x,:] = f(xnew)
    return out_array

def mod_to_overpasstime(modobj,opass_tms,partial_col=None):
    '''
    Interpolate model to satellite overpass time.

    Parameters
    ----------

    modobj : xarray.Dataset
        model data
    opass_tms : pandas.DatetimeIndex
        satellite overpass local time
    partial_col : str
        variable to calculate partial columns for
    Output
    ------
    outmod : xarray.Dataset 
        revised model data at local overpass time
    '''

    nst, = opass_tms.shape
    # nmt, = modobj.time.shape
    # ny,nx = modobj.longitude.shape
    
    # Determine local time offset
    local_utc_offset = (modobj['longitude']/15).round().astype('timedelta64[h]')
    # initialize local time as variable
    modobj['localtime'] = modobj['time'] + local_utc_offset

    # initialize new model object with satellite datetimes
    outmod = []

    for ti in np.arange(nst):
        # Apply filter to select model data within +/- 1 output time step of the overpass time
        tempmod = modobj.where(np.abs(modobj['localtime'] - opass_tms[ti].to_datetime64()) < (modobj.time[1] - modobj.time[0]))
        
        # determine factors for linear interpolation in time
        tfac = 1 - (np.abs(tempmod['localtime'] - opass_tms[ti].to_datetime64())/(modobj.time[1] - modobj.time[0]))
        tempmod = tempmod.drop_vars('localtime')
        # Carry out time interpolation
        ## Note regarding current behavior: will only carry out time interpolation if at least 2 model timesteps
        outmod.append((tfac*tempmod).sum(dim='time', min_count=2,keep_attrs=True))
    #print(outmod)
    outmod = xr.concat(outmod,dim='time')
    outmod['time'] = (['time'],opass_tms)
    
    if partial_col:
        from .tools import calc_partialcolumn        
        outmod[f'{partial_col}_col'] = calc_partialcolumn(outmod,var=partial_col)
        
    return outmod

def mopitt_l3_pairing(model_data,obs_data,co_ppbv_varname,global_model=True):
    ''' Calculate model CO column, with MOPITT averaging kernel applied.
    '''
    try:
        import xesmf as xe
    except ImportError:
        print('satellite_utilities: xesmf module not found')
        raise
    
    ## Check if obs are monthly or daily
    if obs_data.attrs['monthly']:
        # if obs_data is monthly, take monthly mean of model data
        model_obstime = model_data.resample(time='MS').mean()
        filtstr = '%Y-%m'
    elif not obs_data.attrs['monthly']:
        # obs_data is daily, so model and obs seem to be on same time step
        model_obstime = model_data
        filtstr = '%Y-%m-%d'
    else:
        # check frequency of model data 
        # Should not get here.
        tstep = xr.infer_freq(model_data.time.dt.round('D'))
        if tstep == 'MS' or tstep == 'M':
            model_obstime = model_data
            filtstr = '%Y-%m'
        else:
            print('Time resolution of model data and MOPITT data is incompatible')
            raise
        
    # initialize regridder for horizontal interpolation 
    # from model grid to MOPITT grid
    grid_adjust = xe.Regridder(model_obstime[['latitude','longitude']],obs_data[['lat','lon']],
                               'bilinear',periodic=global_model,unmapped_to_nan=True)
    co_model_regrid = grid_adjust(model_obstime[co_ppbv_varname])
    pressure_model_regrid = grid_adjust(model_obstime['pres_pa_mid']/100.)
    
    # enforce dimension order as (time,lat,lon,z)
    co_model_regrid = co_model_regrid.transpose('time','lon','lat','z')
    pressure_model_regrid = pressure_model_regrid.transpose('time','lon','lat','z')
    
    # vertical regrid of model to satellite
    co_regrid = xr.full_like(obs_data['pressure'], np.nan)
    # MEB: loop over time outside of regrid lowers memory usage
    for t in range(obs_data.time.size):
        obs_day = obs_data.time[t].dt.strftime(filtstr)
        co_regrid[t] = vertical_regrid(pressure_model_regrid.sel(time=obs_day).values.squeeze(), 
                                       co_model_regrid.sel(time=obs_day).values.squeeze(), 
                                       obs_data['pressure'][t].values)
    
    # apply AK
    ## log apriori and model data
    log_ap = np.log10(obs_data['apriori_prof'])
    log_mod = np.log10(co_regrid)
    diff_arr = log_mod-log_ap
    ## smooth/apply ak
    smoothed = obs_data['apriori_col'] + (obs_data['ak_col']*diff_arr).sum(dim='alt', min_count=1)
    
    # Add variable name to smoothed model dataarray, combine with obs_data
    smoothed = smoothed.rename(co_ppbv_varname+'_column_model')
    ds = xr.merge([smoothed,obs_data.copy(deep=True)]) 
    
    # Apply scaling to drop scientific notation (x10^{18} molec/cm2 instead of molec/cm2)
    ##  Taylor plot doesn't work if don't do this.
    ds[co_ppbv_varname+'_column_model'] /= 1e18
    ds[co_ppbv_varname+"_column_model"] = ds[co_ppbv_varname+'_column_model'].assign_attrs(units='$10^{18} molec./cm^{2}$')
    ds['column'] /= 1e18
    ds["column"] = ds['column'].assign_attrs(units='$10^{18} molec./cm^{2}$')
    
    # rename dims from lon/lat to x/y for consistency with other datasets
    ds = ds.rename_dims({'lat':'x','lon':'y'})
    # Makde lat/lon coordinates 2d
    lat_2d,lon_2d = np.meshgrid(ds.lat,ds.lon)
    ds['latitude'] = (['y','x'],lat_2d)
    ds['longitude'] = (['y','x'],lon_2d)
    ds = ds.reset_coords().set_coords(['latitude','longitude','time','alt'])
    return ds    

def omps_l3_daily_o3_pairing(model_data,obs_data,ozone_ppbv_varname):
    '''Calculate model ozone column from model ozone profile in ppbv. Move data from model grid 
        to 1x1 degree OMPS L3 data grid. Following data grid matching, take daily mean for model data.
    '''
    try:
        import xesmf as xe
    except ImportError:
        print('satellite_utilities: xesmf module not found')
        raise
    
    # factor for converting ppbv profiles to DU column
    # also requires conversion of dp from Pa to hPa
    du_fac = 1.0e-5*6.023e23/28.97/9.8/2.687e19
    column = (du_fac*(model_data['dp_pa']/100.)*model_data[ozone_ppbv_varname]).sum('z')
    
    # initialize regrid and apply to column data
    grid_adjust = xe.Regridder(model_data[['latitude','longitude']],obs_data[['latitude','longitude']],'bilinear',periodic=True)
    mod_col_obsgrid = grid_adjust(column)
    # Aggregate time-step to daily means
    daily_mean = mod_col_obsgrid.resample(time='1D').mean()
    # change dimension name for date to time
    daily_mean = daily_mean.rename(ozone_ppbv_varname)

    return xr.merge([daily_mean,obs_data])


def calc_satellite_dp(swath_data,
                      model_surface_pressure,
                      apriori_varname='apriroi',
                      sat_pressure_varname='pressure'):
    ''' Calculates the satellite layer thickness in pressure. Currently satellite pressure is in hPa, as this was written with OMPS NM data.
        Model surface pressure is used for calculating thickness of the lowest layer. 
    '''
    
    dp_swath_hPa = xr.full_like(swath_data[apriori_varname],np.nan)
    
    down = swath_data[sat_pressure_varname].roll(z=-1)
    up = swath_data[sat_pressure_varname].roll(z=1)
    down[-1] = 0
    
    dp_swath_hPa[:,:,:] = ((up-swath_data[sat_pressure_varname])/2+(swath_data[sat_pressure_varname]-down)/2).values
    dp_swath_hPa[:,:,0] = ((swath_data[sat_pressure_varname]-down)[0]/2) + (model_surface_pressure/100.-swath_data[sat_pressure_varname][0])
    dp_swath_hPa[:,:,-1] = ((up-swath_data[sat_pressure_varname])[-1]/2+(swath_data[sat_pressure_varname]-down)[-1]).values
    
    return dp_swath_hPa

def select_model_timesteps(obs_times,mod_freq):
    ''' Selects timestamps of model data needed for pairing to observations. 
        Assumes model timestamps are "on the hour" (eg. not something like 12:15)
    '''
    asflt = int(mod_freq/np.timedelta64(1,'h'))

    start = obs_times[0].dt.round(freq=f'{asflt}h').values
    
    # if obs data starts before nearest rounded timestamp, feed in one before.
    if start > obs_times[0].values:
        start -= mod_freq
    end = obs_times[-1].dt.round(freq=f'{asflt}h').values
    # if flight data ends after nearest rounded timestamp, feed in one after
    if end < obs_times[-1].values:
        end += mod_freq
    return pd.date_range(start,end,freq=f'{asflt}h')

def omps_nm_pairing(model_data,obs_data,o3varname,apply_apriori=True):
    '''Calculate model total column ozone with or without OMPS NM averaging kernel applied. 

    Parameters
    ----------
    model_data : xarray.Dataset
        model data
    obs_data : xarray.Dataset
        satellite data
    o3varname : str
        model ozone variable name
    apply_apriori : bool
        If true the satellite apriori and averaging kernel will be applied
    Returns
    -------
    xarray.Dataset
    '''
    try:
        import xesmf as xe
    except ImportError:
        print('satellite_utilities: xesmf module not found')
        raise
    
    du_fac = 1.0e-5*6.023e23/28.97/9.8/2.687e19
    model_output_freq = (model_data.time[1] - model_data.time[0]).values.astype('timedelta64')
    
    # set up dataset of paired data, resturctured to daily avg. on model horiz. grid.
    dys = [pd.to_datetime(dy) for dy in obs_data.keys()]
    ndys = len(dys)
    all_days_paired = []
    
    for day in obs_data.keys():
        paired_day = []
        for swath in obs_data[day]: 
            # select relevant model time steps for specific observation time
            modtsteps = select_model_timesteps(swath.time,model_output_freq)
            temp_moddat = model_data.sel(time=drange)

            # horizontal regrid to satellite pixels
            temp_moddat = temp_moddat.rename({'time':'modtime'})
            regridder_mod_to_swath = xe.Regridder(temp_moddat[['latitude','longitude']],swath[['latitude','longitude']],method='bilinear',periodic=True,unmapped_to_nan=True)
            mod_on_swath = regridder_mod_to_swath(temp_moddat)

            # linearly interpolate in time and enforce an order for dimensions where z is last.
            tfac = (1-(np.abs(mod_on_swath.modtime-mod_on_swath.time)/mfreq)).where(np.abs(mod_on_swath.modtime - mod_on_swath.time) <= mfreq)
            needs_vertical = (tfac*mod_on_swath).sum('modtime')
            needs_vertical = needs_vertical.where(tfac.sum('modtime').round() == 1.0).transpose('x','y','z')
            if apply_apriori: 
                # vertical interpolation
                o3_at_sat = stratify.interpolate(swath.pressure*100,needs_vertical.pres_pa_mid.values,needs_vertical[o3varname].values,axis=-1)
                
                delp_omps_hPa = calc_satellite_dp(swath,needs_vertical['surfpres_pa'],apriori_varname='apriori',sat_pressure_varname='pressure')

                # Calculate partial columns and apply AK/apriori following Rodgers, 2000
                mod_o3_partialcol = (du_fac*delp_omps_hPa*o3_at_sat)
                mod_o3_col = (swath['apriori']).sum('z') + (swath['layer_efficiency']*(mod_o3_partialcol - swath['apriori'])).sum('z')
            else:
                mod_o3_col = (du_fac*needs_vertical[o2varname]*needs_vertical['dp_pa']).sum('z')
            mod_o3_col = mod_o3_col.where(~np.isnan(swath['ozone_column']))
            paired_day.append(xr.merge([mod_o3_col,swath['ozone_column']]))
