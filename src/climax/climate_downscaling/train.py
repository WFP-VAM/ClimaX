# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
sys.path.append('/s3/scratch/wessim.omezzine/ClimaX/src/climax')
sys.path.append('/s3/scratch/wessim.omezzine/ClimaX/src')


import os

from pytorch_lightning.cli import LightningCLI
from climax.climate_downscaling.module import ClimateDownscalingModule
from climax.climate_downscaling.datamodule import ClimateDownscalingDataModule
import loralib as lora


import matplotlib.pyplot as plt  # Import matplotlib

def main():
    
    #Wrap your model with LoRA: Wrap your language model with LoRA using the lora.LoRA class:
    lora_model = ClimateDownscalingModule
    
    #Mark the LoRA parameters as trainable: You need to mark the LoRA parameters as trainable so that they can be updated during training. You can do this using the lora.mark_only_lora_as_trainable function
    # lora.mark_only_lora_as_trainable(lora_model)

    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=lora_model,
        datamodule_class=ClimateDownscalingDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        # auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    
    
    
    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    
  

    # test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path='best')


if __name__ == "__main__":
    import torch

    torch.cuda.empty_cache()
    # torch.set_max_memory_split_size(128)

    main()