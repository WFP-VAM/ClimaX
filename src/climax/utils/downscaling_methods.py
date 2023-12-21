import numpy as np
import torch 
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision.transforms import transforms
import torch.nn.functional as F
from prettytable import PrettyTable
from scipy import stats


def climax_downscale(model_module, data_module,mean_norm, std_norm,var_id, year_to_select, data_path=None, save_path = None):
    # Set device based on CUDA availability
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Selected device:", device)

    # Move the model to the appropriate device
    model_module.to(device).eval()
    dataloader = data_module.test_dataloader()
    mean_norm_tensor = torch.Tensor(mean_norm).to(device)
    std_norm_tensor = torch.Tensor(std_norm).to(device)
    dataloader_iterator = iter(dataloader)

    # Skip to the desired year's data
    batches_to_skip = year_to_select - 2019  # Calculate how many batches to skip
    for _ in range(batches_to_skip):
        batch_data = next(dataloader_iterator)

    x, y, lead_times, in_vars, out_vars = batch_data
    x = x.to(device)
    y = y.to(device)
    lead_times = lead_times.to(device)

    

    with torch.no_grad():
        _, pred = model_module.net.forward(x, y, lead_times, in_vars, out_vars, None, model_module.lat)
    
    inv_normalize = model_module.denormalization
    
    
    tensors=[]
    for i in range(x.shape[1]):
        tensors.append(inv_normalize(x)[:,i])
    init_condition = inv_normalize(x)[:,0]

    
    normalization = data_module.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    gt = y * torch.Tensor(std_norm).to(device) + torch.Tensor(mean_norm).to(device)
    pred = pred * torch.Tensor(std_norm).to(device) + torch.Tensor(mean_norm).to(device)
    bias = pred - gt
    tensors_pred = [init_condition, gt, pred, bias]
    
        
    del x
    del y
    return gt,pred,tensors,tensors_pred


def bilinear_interpolate (forecast,t,verbose=False):
    new_height = forecast.shape[1]
    new_width = forecast.shape[2]
    forecast_resample=forecast[t]
    image_tensor  = forecast_resample[:30:5,:30:5]
    
    rescaled_image = F.interpolate(image_tensor.unsqueeze(0).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)[0]
    
    if verbose:
        
        plt.imshow(rescaled_image)
        plt.show()
    
    
    
    return rescaled_image



# Define a custom weighting function (e.g., higher weights in center)
def custom_weight_function_mean(square):
    normalized_weights = square/np.mean(square)
    return normalized_weights

def custom_weight_function(square):
    local_std = np.std(square)
    normalized_weights = local_std / np.mean(square)
    
    
    return normalized_weights

def Downscale_Weight(rfh_lta,forecast, verbose=False):

    # Initialize an empty weight mask
    weight_mask = np.zeros_like(forecast)

    # Compute weights and apply them
    
    step = 10
    for row in range(0,30,step):
        for col in range(0,60,10):
            rfh_lta_square = rfh_lta[row:row + step, col:col + step]
            weights = custom_weight_function_mean(rfh_lta_square)
            weight_mask[row:row + step, col:col + step] = weights
            
    
    for row in range(0,32,step):
            rfh_lta_square = rfh_lta[row:row + step, 60:]
            weights = custom_weight_function_mean(rfh_lta_square)
            weight_mask[row:row + step, 60:] = weights
        
        
    for col in range(0,64,step):
            rfh_lta_square = rfh_lta[30:, col:col + step]
            weights = custom_weight_function_mean(rfh_lta_square)
            weight_mask[30:, col:col + step] = weights



    # Apply the weight mask to forecast
    weighted_forecast = forecast * weight_mask


    if verbose:
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        axes[0].imshow(forecast)
        axes[1].imshow(rfh_lta)
        axes[2].imshow(weighted_forecast)
        axes[3].imshow(weighted_forecast-rfh_lta)

        axes[0].set_title(f"forecastKm", fontsize='xx-large')
        axes[1].set_title(f"rfh_lta", fontsize='xx-large')
        axes[2].set_title(f"NDVI_Downscaled_det", fontsize='xx-large')
        axes[3].set_title(f"Bias", fontsize='xx-large')
        plt.show()

    return weighted_forecast


def lat_weighted_mse_val(pred, y, transform, lat):
    """Latitude weighted mean squared error
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """
    
  

    error = (pred - y) ** 2  # [B, V, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))

    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad(): 
        
            res =  (error * w_lat).mean()
    return res


def lat_weighted_rmse(pred, y, transform, lat ):
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    # pred = transform(pred)
    # y = transform(y)

    error = (pred - y) ** 2  # [B, V, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    with torch.no_grad():
        res = torch.mean(
                torch.sqrt(torch.mean(error * w_lat, dim=(-2, -1)))
            )

    return res




def remove_nans(pred: torch.Tensor, gt: torch.Tensor):
    # pred and gt are two flattened arrays
    pred_nan_ids = torch.isnan(pred) | torch.isinf(pred)
    pred = pred[~pred_nan_ids]
    gt = gt[~pred_nan_ids]

    gt_nan_ids = torch.isnan(gt) | torch.isinf(gt)
    pred = pred[~gt_nan_ids]
    gt = gt[~gt_nan_ids]

    return pred, gt


def pearson(pred, y, transform,lat):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """

    # pred = transform(pred)
    # y = transform(y)

    loss_dict = {}
    with torch.no_grad():
        pred_, y_ = pred.flatten(), y.flatten()
        pred_, y_ = remove_nans(pred_, y_)
        res = stats.pearsonr(pred_.cpu().numpy(), y_.cpu().numpy())[0]

    return res

def spearman(pred, y, transform,lat):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """

    # pred = transform(pred)
    # y = transform(y)

    loss_dict = {}
    with torch.no_grad():
        pred_, y_ = pred.flatten(), y.flatten()
        pred_, y_ = remove_nans(pred_, y_)
        res = stats.spearmanr(pred_.cpu().numpy(), y_.cpu().numpy())[0]

    return res

def mean_bias(pred, y, transform,lat):
    """
    y: [B, V, H, W]
    pred: [B, V, H, W]
    vars: list of variable names
    lat: H
    """

    # pred = transform(pred)
    # y = transform(y)

    loss_dict = {}
    with torch.no_grad():
        pred_, y_ = pred.flatten(), y.flatten()
        pred_, y_ = remove_nans(pred_, y_)
        res= pred_.mean() - y_.mean()

    return res

