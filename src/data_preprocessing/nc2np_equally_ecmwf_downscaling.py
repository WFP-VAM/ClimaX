# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import glob
import os
import sys
sys.path.append('/mnt/shared/users/wessim.omezzine/ClimaX/src/climax')
sys.path.append('/mnt/shared/users/wessim.omezzine/ClimaX/src/')
sys.path.append("/mnt/shared/users/wessim.omezzine/hip-analysis")
sys.path.append("/s3/scratch/wessim.omezzine/ClimaX/hip-analysis")
sys.path.append("/s3/scratch/wessim.omezzine/ClimaX/src")
sys.path.append("/s3/scratch/wessim.omezzine/ClimaX/src/climax")




import click
import numpy as np
from scipy.ndimage import zoom
import xarray as xr
from tqdm import tqdm
import sys 
import numpy as np
from scipy.interpolate import interp2d






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



def nc2np(path_ecmwf,path_chirps, variables, start_year, end_year, save_dir, partition,num_shards_per_year):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)
    
    if partition == "train":
        normalize_mean_input = {}
        normalize_std_input = {}
        normalize_mean_output = {}
        normalize_std_output = {}

    years = range(start_year, end_year)
    
    

    for year in tqdm(years):
        np_ecmwf_vars = {}
        np_chirps_vars = {}
        
        input_variables = {
        'rfh_lta': variables['rfh_lta'],
        'forecast': variables['forecast'],
        # 'r2h': variables['r2h'],
        # 'r3h': variables['r3h'],
        }
        for variable_name, variable_data in input_variables.items():
            
            
#             ps_ecmwf = glob.glob(os.path.join(path_ecmwf, var, f"*{year}*.nc"))
#             ps_chirps = glob.glob(os.path.join(path_chirps, var, f"*{year}*.nc"))
#             ds_ecmwf = xr.open_mfdataset(ps_ecmwf, combine="by_coords", parallel=True)
#             ds_chirps = xr.open_mfdataset(ps_chirps, combine="by_coords", parallel=True)
            
            var = variable_name
        
            if variable_name == "forecast":
                code = 'tp'
            else: 
                code = 'band'
            
            variable_year_data = variable_data.sel(time=variable_data.time.dt.year == year)
            ds_ecmwf =  variable_year_data
            ds_ecmwf[code] = ds_ecmwf[code].expand_dims("val", axis=1)
            
            # print(ds_ecmwf.variables)
            np_ecmwf_vars[var] = ds_ecmwf[code].to_numpy()
            
            

            
            
            
                
            if partition == "train":
                var_mean_ecmwf_yearly = np_ecmwf_vars[var].mean(axis=(0, 2, 3))
                var_std_ecmwf_yearly = np_ecmwf_vars[var].std(axis=(0, 2, 3))
                if var not in normalize_mean_input:
                    normalize_mean_input[var] = [var_mean_ecmwf_yearly]
                    normalize_std_input[var] = [var_std_ecmwf_yearly]
                else:
                    normalize_mean_input[var].append(var_mean_ecmwf_yearly)
                    normalize_std_input[var].append(var_std_ecmwf_yearly)
                

        variable_year_data = variables['rfh'].sel(time=variables['rfh'].time.dt.year == year)
        ds_chirps = variable_year_data
        code = 'band'
        ds_chirps[code] = ds_chirps[code].expand_dims("val", axis=1)
        np_chirps_vars["rfh"] = ds_chirps[code].to_numpy()
        if partition == "train":
            rfh_mean_yearly = np_chirps_vars["rfh"].mean(axis=(0, 2, 3))
            rfh_std_yearly = np_chirps_vars["rfh"].std(axis=(0, 2, 3))
            if "rfh" not in normalize_mean_output:
                normalize_mean_output["rfh"] = [rfh_mean_yearly]
                normalize_std_output["rfh"] = [rfh_std_yearly]
            else:
                normalize_mean_output["rfh"].append(rfh_mean_yearly)
                normalize_std_output["rfh"].append(rfh_std_yearly)
                        
          

        assert DEKAD_PER_YEAR % num_shards_per_year == 0
        num_hrs_per_shard = DEKAD_PER_YEAR // num_shards_per_year
        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
            sharded_input = {k: np_ecmwf_vars[k][start_id:end_id] for k in np_ecmwf_vars.keys()}
            sharded_output = {k: np_chirps_vars[k][start_id:end_id] for k in np_chirps_vars.keys()}
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id:02}_inp.npz"),
                **sharded_input,
            )
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id:02}_out.npz"),
                **sharded_output,
            )
        
        
        # assert DEKAD_PER_YEAR % num_shards_per_year == 0
        # num_hrs_per_shard = DEKAD_PER_YEAR // num_shards_per_year
        # for shard_id in range(num_shards_per_year):
        #     start_id = shard_id * num_hrs_per_shard
        #     end_id = start_id + num_hrs_per_shard
        #     sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
        #     np.savez(
        #         os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"),
        #         **sharded_data,
        #         )

#         np.savez(os.path.join(save_dir, partition, f"{year}.npz"), **np_vars)


    if partition == "train":
        print(normalize_mean_input.keys())
        for var in normalize_mean_input.keys():
            normalize_mean_input[var] = np.stack(normalize_mean_input[var], axis=0)
            normalize_std_input[var] = np.stack(normalize_std_input[var], axis=0)
        for var in normalize_mean_output.keys():
            normalize_mean_output[var] = np.stack(normalize_mean_output[var], axis=0)
            normalize_std_output[var] = np.stack(normalize_std_output[var], axis=0)

        for var in normalize_mean_input.keys():  # aggregate over the years
            # input
            mean_input, std_input = normalize_mean_input[var], normalize_std_input[var]
            variance_input = (std_input**2).mean(axis=0) + (mean_input**2).mean(axis=0) - mean_input.mean(axis=0) ** 2
            std_input = np.sqrt(variance_input)
            mean_input = mean_input.mean(axis=0)
            normalize_mean_input[var] = mean_input
            normalize_std_input[var] = std_input
            
            
        for var in normalize_mean_output.keys():
            # output
            mean_output, std_output = normalize_mean_output[var], normalize_std_output[var]
            variance_output = (std_output**2).mean(axis=0) + (mean_output**2).mean(axis=0) - mean_output.mean(axis=0) ** 2
            std_output = np.sqrt(variance_output)
            mean_output = mean_output.mean(axis=0)
            normalize_mean_output[var] = mean_output
            normalize_std_output[var] = std_output


        np.savez(os.path.join(save_dir, "normalize_mean_input.npz"), **normalize_mean_input)
        np.savez(os.path.join(save_dir, "normalize_std_input.npz"), **normalize_std_input)
        np.savez(os.path.join(save_dir, "normalize_mean_output.npz"), **normalize_mean_output)
        np.savez(os.path.join(save_dir, "normalize_std_output.npz"), **normalize_std_output)


#     for var in climatology.keys():
#         climatology[var] = np.stack(climatology[var], axis=0)
#     climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
    
    
    
#     np.savez(
#         os.path.join(save_dir, partition, "climatology.npz"),
#         **climatology,
#     )


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


@click.command()
@click.option('--region', default="Mozambique/Members/region_1", help='Region path to use in preprocessing.')

def main(region):
       
    path_ecmwf = f"/s3/scratch/wessim.omezzine/ClimaX/Data/Downscaling/Rainfall/{region}/data/"
    path_chirps = f"/s3/scratch/wessim.omezzine/ClimaX/Data/Downscaling/Rainfall/{region}/data/"
    save_dir = f"/s3/scratch/wessim.omezzine/ClimaX/Data/Downscaling/Rainfall/{region}/data_npz/" 
    
    
    rfh = xr.open_dataset(path_chirps+"r1h.nc").fillna(0)
    # r2h = xr.open_dataset(path_chirps+"r2h_lta.nc").fillna(0)
    # r3h = xr.open_dataset(path_chirps+"r3h_lta.nc").fillna(0)
    rfh_lta = xr.open_dataset(path_chirps+"r1h_lta.nc").fillna(0)
    forecast = xr.open_dataset(path_ecmwf+"forecast.nc").fillna(0)
    
    
    ##### TO CHANGE!!!!
    dimensions_lat = 32
    dimensions_lon = 64
    # ndvi_1 = select_shape(ndvi_1, dimensions, dimensions, True )
    # ndvi_1_lta = select_shape(ndvi_1_lta, dimensions, dimensions, True )
    # ndvi_5 = select_shape(ndvi_5, ndvi_5.dims["latitude"], ndvi_5.dims["longitude"], True)
   
    

    variables = {
    "rfh": rfh,
    # "r2h": r2h,
    # "r3h": r3h,
    "rfh_lta": rfh_lta,
    "forecast": forecast,

}
    
  
    os.makedirs(save_dir, exist_ok=True)
    
    num_shards_per_year = 1
    
    nc2np(path_ecmwf = path_ecmwf, path_chirps=path_chirps  ,variables = variables, start_year =1983, end_year=2018, save_dir = save_dir , partition='train',num_shards_per_year=num_shards_per_year )
    nc2np(path_ecmwf = path_ecmwf, path_chirps=path_chirps ,variables = variables, start_year =2018, end_year=2020, save_dir = save_dir , partition='val',num_shards_per_year=num_shards_per_year)
    nc2np(path_ecmwf = path_ecmwf, path_chirps=path_chirps  ,variables = variables, start_year =2020, end_year=2023, save_dir = save_dir , partition='test', num_shards_per_year=num_shards_per_year)
    
    # save lat and lon data
    ps = glob.glob(path_chirps + 'r1h.nc')
    x = xr.open_mfdataset(ps[0], parallel=True)
    # x = x.rename({'x': 'longitude'})
    # x = x.rename({'y': 'latitude'})
    x =x.isel(latitude=slice(0, dimensions_lat), longitude=slice(0, dimensions_lon))
    
        
    lat = x["latitude"].to_numpy()
    print("lat.shape ", lat.shape)
    lon = x["longitude"].to_numpy()
    
    np.save(os.path.join(save_dir, "lat.npy"), lat)
    np.save(os.path.join(save_dir, "lon.npy"), lon)

if __name__=="__main__":
    main()






