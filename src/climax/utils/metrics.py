# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from scipy import stats


def mse(pred, y, vars, lat=None, mask=None):
    """Mean squared error

    Args:
        pred: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """
    
    

    loss = (pred - y) ** 2

    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (loss[:, i] * mask).sum() / mask.sum()
            else:
                loss_dict[var] = loss[:, i].mean()

    if mask is not None:
        loss_dict["loss"] = (loss.mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = loss.mean(dim=1).mean()

    return loss_dict


import torch
import numpy as np

def huber_loss(error, delta=.5):
    """
    Calculate the Huber loss.
    """
    abs_error = torch.abs(error)
    quadratic = torch.where(abs_error < delta, 0.5 * error ** 2, delta * (abs_error - 0.5 * delta))
    return quadratic

def loss_function_training(pred, y, vars, lat,var_weights=[.7,.3], mask=None, delta=1.):
    """
    Latitude weighted Huber loss.

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
        delta: threshold for Huber loss
    """

    error = pred - y  # [N, C, H, W]

    # Calculate the Huber loss
    error = huber_loss(error, delta)

    # Latitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[var] = torch.mean(torch.sqrt(var_weights[i] * torch.mean(error[:, i] * w_lat, dim=(-2, -1)))) + torch.abs(pred.mean() - y.mean())

    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()

    return loss_dict



def loss_function_training2(pred, y, vars, lat, mask=None):
    """Latitude weighted mean squared error

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """
    

    error = (pred - y) ** 2  # [N, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[var] =   torch.mean(torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1))))  + torch.abs(pred.mean() - y.mean())
    # (error[:, i] * w_lat).mean() + (torch.abs(pred- y)*w_lat).mean() +
    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()
    
    
    
    return loss_dict


def lat_weighted_mse(pred, y, vars, lat, mask=None):
    """Latitude weighted mean squared error

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [N, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum() +torch.abs(pred- y).mean() + torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )
            else:
                loss_dict[var] = (error[:, i] * w_lat).mean() + torch.abs(pred- y).mean() + torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1))) 
            )

    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()
    
    
    
    return loss_dict

def lat_weighted_mean_bias(pred, y, vars, lat):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """
    
    # Latitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_ = pred[:, i].flatten()
            y_ = y[:, i].flatten()
            pred_, y_ = remove_nans(pred_, y_)
            loss_dict[f"mean_absolute_error_{var}"] = torch.abs(pred_ - y_).mean()

    loss_dict["loss"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict



def lat_weighted_rmse_train(pred, y, vars, lat):
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

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[var] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )

    loss_dict["w_rmse_train"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict



def lat_weighted_mse_val(pred, y, transform, vars, lat, clim, log_postfix):
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
        for i, var in enumerate(vars):
           
            loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()

    loss_dict["w_mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])
    
    
    return loss_dict


def lat_weighted_rmse(pred, y, transform, vars, lat, clim, log_postfix):
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    pred = transform(pred)
    y = transform(y)

    error = (pred - y) ** 2  # [B, V, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )

    loss_dict["w_rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_acc(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """
    
    

    pred = transform(pred)
    y = transform(y)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    # clim = torch.mean(y, dim=(0, 1), keepdim=True)
    clim = clim.to(device=y.device).unsqueeze(0)
    pred = pred - clim
    y = y - clim
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_prime = pred[:, i] - torch.mean(pred[:, i])
            y_prime = y[:, i] - torch.mean(y[:, i])
            loss_dict[f"acc_{var}_{log_postfix}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
                torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
            )

    loss_dict["acc"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_nrmses(pred, y, transform, vars, lat, log_steps, log_days, clim):
    """
    y: [N, T, C, H, W]
    pred: [N, T, C, H, W]
    vars: list of variable names
    lat: H
    """
    
 

    pred = transform(pred)
    y = transform(y)
    y_normalization = clim

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))  # (H,)
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(-1).to(dtype=y.dtype, device=y.device)  # (H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for day, step in zip(log_days, log_steps):
                pred_ = pred[:, step - 1, i]  # N, H, W
                y_ = y[:, step - 1, i]  # N, H, W
                error = (torch.mean(pred_, dim=0) - torch.mean(y_, dim=0)) ** 2  # (H, W)
                error = torch.mean(error * w_lat)
                loss_dict[f"w_nrmses_{var}"] = torch.sqrt(error) / y_normalization
    return loss_dict


def lat_weighted_nrmseg(pred, y, transform, vars, lat, log_steps, log_days, clim):
    """
    y: [N, T, C, H, W]
    pred: [N, T, C, H, W]
    vars: list of variable names
    lat: H
    """

    y = transform(y)
    y_normalization = clim

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))  # (H,)
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=y.dtype, device=y.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for day, step in zip(log_days, log_steps):
                pred_ = pred[:, step - 1, i]  # N, H, W
                pred_ = torch.mean(pred_ * w_lat, dim=(-2, -1))  # N
                y_ = y[:, step - 1, i]  # N, H, W
                y_ = torch.mean(y_ * w_lat, dim=(-2, -1))  # N
                error = torch.mean((pred_ - y_) ** 2)
                loss_dict[f"w_nrmseg_{var}"] = torch.sqrt(error) / y_normalization
    return loss_dict


def lat_weighted_nrmse(pred, y, transform, vars, lat, log_steps, log_days, clim):
    """
    y: [N, T, C, H, W]
    pred: [N, T, C, H, W]
    vars: list of variable names
    lat: H
    """
    print("pearson pred",pred.shape)
    print("pearson y",y.shape)
    nrmses = lat_weighted_nrmses(pred, y, transform, vars, lat, log_steps, log_days, clim)
    nrmseg = lat_weighted_nrmseg(pred, y, transform, vars, lat, log_steps, log_days, clim)
    loss_dict = {}
    for var in vars:
        loss_dict[f"w_nrmses_{var}"] = nrmses[f"w_nrmses_{var}"]
        loss_dict[f"w_nrmseg_{var}"] = nrmseg[f"w_nrmseg_{var}"]
        loss_dict[f"w_nrmse_{var}"] = nrmses[f"w_nrmses_{var}"] + 5 * nrmseg[f"w_nrmseg_{var}"]
    return loss_dict


def remove_nans(pred: torch.Tensor, gt: torch.Tensor):
    # pred and gt are two flattened arrays
    pred_nan_ids = torch.isnan(pred) | torch.isinf(pred)
    pred = pred[~pred_nan_ids]
    gt = gt[~pred_nan_ids]

    gt_nan_ids = torch.isnan(gt) | torch.isinf(gt)
    pred = pred[~gt_nan_ids]
    gt = gt[~gt_nan_ids]

    return pred, gt


def pearson(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_, y_ = pred[:, i].flatten(), y[:, i].flatten()
            pred_, y_ = remove_nans(pred_, y_)
            loss_dict[f"pearsonr_{var}_{log_postfix}"] = stats.pearsonr(pred_.cpu().numpy(), y_.cpu().numpy())[0]

    loss_dict["pearsonr"] = np.mean([loss_dict[k] for k in loss_dict.keys()])

    return loss_dict

def spearman(pred, y, transform,vars, lat, clim, log_postfix):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_, y_ = pred[:, i].flatten(), y[:, i].flatten()
            pred_, y_ = remove_nans(pred_, y_)
            loss_dict[f"spearmanr_{var}_{log_postfix}"] = stats.spearmanr(pred_.cpu().numpy(), y_.cpu().numpy())[0]

    loss_dict["spearmanr"] = np.mean([loss_dict[k] for k in loss_dict.keys()])

    return loss_dict

def mean_bias(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B, V, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_, y_ = pred[:, i].flatten(), y[:, i].flatten()
            pred_, y_ = remove_nans(pred_, y_)
            loss_dict[f"mean_bias_{var}_{log_postfix}"] = pred_.mean() - y_.mean()

    loss_dict["mean_bias"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


