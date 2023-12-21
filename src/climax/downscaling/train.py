# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
sys.path.append('/mnt/shared/users/wessim.omezzine/ClimaX/src/climax')
sys.path.append('/mnt/shared/users/wessim.omezzine/ClimaX/src')

from climax.downscaling.datamodule import DownscalingDataModule
from climax.downscaling.module import DownscalingModule
from pytorch_lightning.cli import LightningCLI

import torch

def print_unused_parameters(model):
    unused_parameters = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_parameters.append(name)
    print(unused_parameters)

def main():
    print("main")
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=DownscalingModule,
        datamodule_class=DownscalingDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)
    cli.model.set_val_clim(cli.datamodule.val_clim)
    cli.model.set_test_clim(cli.datamodule.test_clim)
    
    
    
    
    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    print("UNUSED PARAMETERS")
    print_unused_parameters(cli.model)
    

    # test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")


    
import gc
def report_gpu():
    print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available")

        # Get the current CUDA device
        current_device = torch.cuda.current_device()
        print("Current CUDA device:", torch.cuda.get_device_name(current_device))
    else:
        print("CUDA is not available")
        
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    main()
