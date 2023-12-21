# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import glob
import os
import sys
sys.path.append('/mnt/shared/users/wessim.omezzine/ClimaX/src/climax')
sys.path.append('/mnt/shared/users/wessim.omezzine/ClimaX/src/')
sys.path.append("/mnt/shared/users/wessim.omezzine/hip-analysis")

import click
import numpy as np
from scipy.ndimage import zoom
import xarray as xr
from tqdm import tqdm
import sys 
import numpy as np
from scipy.interpolate import interp2d

PATH = "/mnt/shared/users/wessim.omezzine/ClimaX/Data/Downscaling/Senegal/data/"



# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from utils.data_utils import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR, DEKAD_PER_YEAR
from hip.analysis.aoi import AnalysisArea
from hip.analysis.data._datasources import DATASOURCE_CONFIGS
import dask
from dask.distributed import Client, progress
import rioxarray as rxr


def resample(x, out_h, out_w):
    time, channels, h, w = x.shape
    new_h_indices = np.linspace(0, h - 1, out_h)
    new_w_indices = np.linspace(0, w - 1, out_w)
    interpolated_x = np.zeros((time, channels, out_h, out_w))
    for t in range(time):
        for c in range(channels):
            channel_slice = x[t, c, :, :]

            # Create interpolation function for the current channel slice
            f_interp = interp2d(np.arange(w), np.arange(h), channel_slice, kind='linear')

            # Perform the 2-dimensional interpolation
            interpolated_channel = f_interp(new_w_indices, new_h_indices)

            interpolated_x[t, c, :, :] = interpolated_channel
    return interpolated_x

def get_data(area):
    
    
    ndvi_5 = area.get_dataset(["MODIS","NDVI_smoothed_5KM"])
    lst_5 = area.get_dataset(["MODIS","LST_smoothed_5KM"])
    rfh = area.get_dataset(["CHIRPS","rfh_dekad"])
    r1h_dekad = area.get_dataset(["CHIRPS","r1h_dekad"])
    r2h_dekad = area.get_dataset(["CHIRPS","r2h_dekad"])
    r3h_dekad = area.get_dataset(["CHIRPS","r3h_dekad"])


    rfh = rfh.sel(time=lst_5.time.values)    #Align the time grid
    r1h_dekad = r1h_dekad.sel(time=lst_5.time.values)    #Align the time grid
    r2h_dekad = r2h_dekad.sel(time=lst_5.time.values)    #Align the time grid
    r3h_dekad = r3h_dekad.sel(time=lst_5.time.values)    #Align the time grid   

    ndvi_5.to_netcdf("ndvi_5.nc")
    lst_5.to_netcdf("lst_5.nc")
    rfh.to_netcdf("rfh.nc")
    r1h_dekad.to_netcdf("Data/CHIRPS_MODIS/data/r1h_dekad.nc") 
    r2h_dekad.to_netcdf("Data/CHIRPS_MODIS/data/r2h_dekad.nc") 
    r3h_dekad.to_netcdf("Data/CHIRPS_MODIS/data/r3h_dekad.nc") 
    
    
def nc2np(path, variables, start_year, end_year, save_dir, partition,num_shards_per_year):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)
    
    if partition == "train":
        normalize_mean = {}
        normalize_std = {}
    climatology = {}

    years = range(start_year, end_year)

    for year in tqdm(years):
        np_vars = {}
        
        i=0
        for variable_name, variable_data in variables.items():
            var = variable_name
            variable_year_data = variable_data.sel(time=variable_data.time.dt.year == year)
            
            ds = variable_year_data
            # code = list(ds.data_vars)[1]
            if len(list(ds.data_vars))==2:
                code = list(ds.data_vars)[1]
            else :
                code = list(ds.data_vars)[0]
                
                

            
            if len(ds[code].shape) == 3:
                ds[code] = ds[code].expand_dims("val", axis=1)
                print(ds, var)
                np_vars[var] = ds[code].to_numpy()[:,:]
                time_size,channels, y_height, y_width = np_vars[var].shape
                out_h, out_w = y_height, y_width

            
            
            if partition == "train":
                var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3))
                var_std_yearly = np_vars[var].std(axis=(0, 2, 3))
                
    
                
                if var not in normalize_mean:
                    normalize_mean[var] = [var_mean_yearly]
                    normalize_std[var] = [var_std_yearly]
                else:
                    normalize_mean[var].append(var_mean_yearly)
                    normalize_std[var].append(var_std_yearly)

            clim_yearly = np_vars[var].mean(axis=0)
            if var not in climatology:
                climatology[var] = [clim_yearly]
            else:
                climatology[var].append(clim_yearly)
        
        
        assert DEKAD_PER_YEAR % num_shards_per_year == 0
        num_hrs_per_shard = DEKAD_PER_YEAR // num_shards_per_year
        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
            sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"),
                **sharded_data,
                )

#         np.savez(os.path.join(save_dir, partition, f"{year}.npz"), **np_vars)
        

    if partition == "train":
        for var in normalize_mean.keys():
            normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
            normalize_std[var] = np.stack(normalize_std[var], axis=0)

        for var in normalize_mean.keys():
            mean, std = normalize_mean[var], normalize_std[var]
            variance = (std ** 2).mean(axis=0) + (mean ** 2).mean(axis=0) - mean.mean(axis=0) ** 2
            std = np.sqrt(variance)
            mean = mean.mean(axis=0)
            normalize_mean[var] = mean
            normalize_std[var] = std

        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)

    for var in climatology.keys():
        climatology[var] = np.stack(climatology[var], axis=0)
    climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
    
    
    
    np.savez(
        os.path.join(save_dir, partition, "climatology.npz"),
        **climatology,
    )
    
    
def select_shape(array, out_lon, out_lat,n1=False):
    
    if n1:
        array = array.rename({'x': 'longitude'})
        array = array.rename({'y': 'latitude'})
        
    print(array.dims)
    array = array.isel(latitude=slice(0, out_lat) )
    print(array.dims)
    array = array.isel(longitude=slice(0, out_lon) )
    print(array.dims)
    

#     if array.longitude.shape[0] %2 != 0:
#         array = array.isel(longitude=slice(0, -1))

#     if array.latitude.shape[0] %2 != 0:
#         array = array.isel(latitude=slice(0, -1))
    return array

def main(
    root_dir = PATH,
    save_dir = '/mnt/shared/users/wessim.omezzine/ClimaX/Data/Downscaling/Senegal/data_npz' ,
):
       
    
    ndvi_1 = xr.open_dataset(PATH+"ndvi_1.nc").fillna(0)    
    ndvi_1_lta = xr.open_dataset(PATH+"ndvi_1_lta.nc").fillna(0)
    ndvi_5 = xr.open_dataset(PATH+"ndvi_5_near.nc").fillna(0)
    
    
    ##### TO CHANGE!!!!
    dimensions = 32
    # ndvi_1 = select_shape(ndvi_1, dimensions, dimensions, True )
    # ndvi_1_lta = select_shape(ndvi_1_lta, dimensions, dimensions, True )
    # ndvi_5 = select_shape(ndvi_5, ndvi_5.dims["latitude"], ndvi_5.dims["longitude"], True)
   
    

    variables = {
    "ndvi_1": ndvi_1,
    "ndvi_1_lta": ndvi_1_lta,
    "ndvi_5": ndvi_5,
    
}
    ndvi_1 = ndvi_1.where(ndvi_1 >= 0, 0)
    ndvi_1_lta = ndvi_1_lta.where(ndvi_1_lta >= 0, 0)
    ndvi_5 = ndvi_5.where(ndvi_5 >= 0, 0)

   
    
    
  
    os.makedirs(save_dir, exist_ok=True)
    
    num_shards_per_year = 1
    
    nc2np(path = root_dir  ,variables = variables, start_year =2003, end_year=2018, save_dir = save_dir , partition='train',num_shards_per_year=num_shards_per_year )
    nc2np(path = root_dir  ,variables = variables, start_year =2018, end_year=2020, save_dir = save_dir , partition='val',num_shards_per_year=num_shards_per_year)
    nc2np(path = root_dir  ,variables = variables, start_year =2020, end_year=2023, save_dir = save_dir , partition='test', num_shards_per_year=num_shards_per_year)
    
    # save lat and lon data
    ps = glob.glob(PATH + 'ndvi_1.nc')
    x = xr.open_mfdataset(ps[0], parallel=True)
    x = x.rename({'x': 'longitude'})
    x = x.rename({'y': 'latitude'})
    x =x.isel(latitude=slice(0, dimensions))
    
        
    lat = x["latitude"].to_numpy()
    print("lat.shape ", lat.shape)
    lon = x["longitude"].to_numpy()
    
    np.save(os.path.join(save_dir, "lat.npy"), lat)
    np.save(os.path.join(save_dir, "lon.npy"), lon)
    
if __name__=="__main__":
    main()






