import torch
import model

def depth_gt_loss(gt: torch.tensor, depth: torch.tensor):
    z_error = depth - gt
    z_MSE = 
    pass