# Amir Ibrahim :: amir.ibrahim@nasa.gov

# These are imports necessary to run the script
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def column_ozone(filepath, lat, lon, altitude, time):
    ozone = xr.open_dataset(filepath)
    ma = 48
    mo3 = 28.97
    MMR = ozone.O3
    VMR = MMR/(ma/mo3)*1e6
    DELP = ozone.DELP
    g0 = 9.80665 # m/s2
    T0 = 273.15 # K
    k = 1.3807e-23 # J/K/molecule
    p0 = 1.01325e5 #Pa
    Na = 6.022e23
    R = 287.3 #J/Kg/K
    vmr_indx = np.where(3.28084*ozone.H[0,:,200,200].values/1000 >= altitude)
    constant = 10*(R*T0)/(g0*p0)
    Column = np.zeros([8,len(vmr_indx[0]),361,576])
    for i in vmr_indx[0]:
        Column[:,i,:,:] = 0.5 * (VMR[:,i,:,:] + VMR[:,i+1,:,:]) * DELP[:,i,:,:]
    O3 = 1e-2*constant*np.sum(Column, axis=1)
    O3_x = xr.full_like(VMR[:,0,:,:], O3)
    C_O3 = O3_x.interp(lat = lat, lon = lon, time = time )
    return C_O3.values

def ozone_transmittance(C_O3, delta_O3, lunar_zenith):
    ozone_abs = 'k_o3.txt'
    d = pd.read_table(ozone_abs, header=35, delim_whitespace=False)
    o3_wl = np.empty(len(d))
    ko3 = np.empty(len(d))
    for i in range(len(d)):
        o3_wl[i] = d.values[:,0][i][0:5]
        ko3[i]   = d.values[:,0][i][7:]
    TO3 = np.exp(-C_O3*1e-3*ko3/np.cos(np.deg2rad(lunar_zenith)))
    sigma_CO3 = delta_O3*C_O3*1e-3
    sigma_TO3 = TO3*np.abs(-ko3/np.cos(np.deg2rad(lunar_zenith)))*sigma_CO3
    return o3_wl, TO3, sigma_TO3
    
# Here to run each case
# Path to nc4 merra-2 files
path = Path('/glusteruser/aibrahi2/merra_ozone/')
## Nov-13-2019
filename = Path('MERRA2_400.inst3_3d_asm_Nv.20191113.nc4')
filepath = Path.joinpath(path,filename)
time = '2019-11-13T07:16:00.000000000'
lat = 35.4385
lon = -116.332
altitude = 68
lunar_zenith = 90 - 68
delta_O3 = 0.05 ## assume 5% error on column ozone
C_O3 = column_ozone(filepath, lat, lon, altitude, time)
wl, T1, sigma_T1 = ozone_transmittance(C_O3, delta_O3, lunar_zenith)


## Nov-14-2019
filename = Path('MERRA2_400.inst3_3d_asm_Nv.20191114.nc4')
filepath = Path.joinpath(path,filename)
time = '2019-11-14T07:46:00.000000000'
lat = 36.97716666
lon = -118.43466
altitude = 68
lunar_zenith = 90 - 65
delta_O3 = 0.05 ## assume 5% error on column ozone
C_O3 = column_ozone(filepath, lat, lon, altitude, time)
wl, T2, sigma_T2 = ozone_transmittance(C_O3, delta_O3, lunar_zenith)

## Nov-15-2019
filename = Path('MERRA2_400.inst3_3d_asm_Nv.20191115.nc4')
filepath = Path.joinpath(path,filename)
time = '2019-11-15T08:28:00.000000000'
lat = 36.8261
lon = -118.1978
altitude = 68
lunar_zenith = 90 - 65
delta_O3 = 0.05 ## assume 5% error on column ozone
C_O3 = column_ozone(filepath, lat, lon, altitude, time)
wl, T3, sigma_T3 = ozone_transmittance(C_O3, delta_O3, lunar_zenith)

## Nov-16-2019
filename = Path('MERRA2_400.inst3_3d_asm_Nv.20191116.nc4')
filepath = Path.joinpath(path,filename)
time = '2019-11-16T09:17:00.000000000'
lat = 37.3216
lon = -117.8948
altitude = 68
lunar_zenith = 90 - 65
delta_O3 = 0.05 ## assume 5% error on column ozone
C_O3 = column_ozone(filepath, lat, lon, altitude, time)
wl, T4, sigma_T4 = ozone_transmittance(C_O3, delta_O3, lunar_zenith)

## Nov-17-2019
filename = Path('MERRA2_400.inst3_3d_asm_Nv.20191117.nc4')
filepath = Path.joinpath(path,filename)
time = '2019-11-17T09:23:00.000000000'
lat = 37.3135
lon = -117.6856
altitude = 68
lunar_zenith = 90 - 55
delta_O3 = 0.05 ## assume 5% error on column ozone
C_O3 = column_ozone(filepath, lat, lon, altitude, time)
wl, T5, sigma_T5 = ozone_transmittance(C_O3, delta_O3, lunar_zenith)


# Save data into a netcdf file
loc = ['wavelength (nm)']
coord = [wl]
transmittance1 = xr.DataArray(T1, coords=coord, dims=loc,name='Transmittance')
transmittance2 = xr.DataArray(T2, coords=coord, dims=loc,name='Transmittance')
transmittance3 = xr.DataArray(T3, coords=coord, dims=loc,name='Transmittance')
transmittance4 = xr.DataArray(T4, coords=coord, dims=loc,name='Transmittance')
transmittance5 = xr.DataArray(T5, coords=coord, dims=loc,name='Transmittance')

dt1 = xr.DataArray(sigma_T1, coords=coord, dims=loc,name='Transmittance_unc')
dt2 = xr.DataArray(sigma_T2, coords=coord, dims=loc,name='Transmittance_unc')
dt3 = xr.DataArray(sigma_T3, coords=coord, dims=loc,name='Transmittance_unc')
dt4 = xr.DataArray(sigma_T4, coords=coord, dims=loc,name='Transmittance_unc')
dt5 = xr.DataArray(sigma_T5, coords=coord, dims=loc,name='Transmittance_unc')

transmittance_ = xr.Dataset({'T13': transmittance1, 'T14' : transmittance2, 'T15': transmittance3
                            , 'T16' : transmittance4, 'T17': transmittance5 
                            , 'dT13': dt1, 'dT14': dt2, 'dT15': dt3, 'dT16': dt4, 'dT17': dt5})
transmittance_.to_netcdf(path='./ER2_ozone_transmittance.nc')
