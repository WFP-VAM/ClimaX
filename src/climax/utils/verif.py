from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms
import time


def plot_train(x,y,batch_idx):
    t = 0
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Plot the first figure on the left subplot
    im1 = axes[0].imshow(x.cpu().numpy()[t][0], cmap=plt.cm.RdBu)  # You can change the colormap as needed
    axes[0].set_title('Forecast')
    
    im2 = axes[1].imshow(x.cpu().numpy()[t][1], cmap=plt.cm.RdBu)  # You can change the colormap as needed
    axes[1].set_title('LTA')
    
    im3 = axes[2].imshow(y.cpu().numpy()[t][0], cmap=plt.cm.RdBu)  # You can change the colormap as needed
    axes[2].set_title('')

   
    
    fig.colorbar(im1)
    fig.colorbar(im2)
    fig.colorbar(im3)
    
    timestamp = int(time.time())
    filename = f"tests/epoch{batch_idx}_{timestamp}.png"
    plt.savefig(filename)
    
def plot_pred(preds,y, log_postfix):
    for t in range(0,preds.shape[0],50):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the first figure on the left subplot
        im1 = axes[0].imshow(preds[t].detach().squeeze().cpu().numpy(), cmap=plt.cm.RdBu)  # You can change the colormap as needed
        axes[0].set_title('Forecast')

        im2 = axes[1].imshow(y.cpu().numpy()[t][0], cmap=plt.cm.RdBu)  # You can change the colormap as needed
        axes[1].set_title('GT ')


        fig.colorbar(im1)
        fig.colorbar(im2)

        timestamp = int(time.time())
        filename = f"tests/predictions_{timestamp}.png"
        plt.savefig(filename)
    