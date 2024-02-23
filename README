# RAM-C work


## Scripts

* `src/data_preprocessing/nc2np_equally_ecmwf_downscaling.py`: A preprocessing script for ecmwf forecasts downscaling
	1. The script takes as input the paths for ecmwf and for chirps.
	2. Inputs data are stored in .nc files for each variable.Example: rfh.nc, forecast.nc. In these xarrays we find the data organised in years
	3. The script preprocess the data to output them in numpy format in `save_dir` where they will be saved as 
	
	|-save_dir
		|-train
			|-2003_inp.npz
			|-2003_out.npz
			|-2004_inp.npz
			|-2004_out.npz
			...
	4. The script also computes the mean and std of the variables that will be used later for normalization. lat and longitude are saved also as lat.nyp and lon.npy
	5. In the `main()` function we define 
	`variables = {"rfh": rfh, "weighted": weighted,"rfh_lta": rfh_lta,"forecast": forecast}

* In src/Climax we add `climate_downscaling`. The modifications are:

	* `train.py` : add LoRA. The choice between using LoRA or the original model is coded in the cli variable: `LightningCLI(model_class=lora_model,...)` or `LightningCLI(model_class=ClimateDownscalingModule ,...)`   

* In configs/ we define the parameters of the architecture, the training,... 


* For ClimaX finetuning, we use climate_downscaling/train.py script with selecting needed arguments that are already defined in the config file. For example 

`python src/climax/climate_downscaling --mdoel.lr = 1e-1 --model.net.img_size=[32,64]`

## Notebooks

* Data Preparation.ipynb: an exhaustive explanation of the preprocessing with the description of the sahpes
* ECMWF Downscaling.ipynb;: The finetuning and evaluation step
