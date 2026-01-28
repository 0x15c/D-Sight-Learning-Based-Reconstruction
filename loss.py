import torch
import torch.nn
import model

pixelwiseLoss = torch.nn.MSELoss()
def depth_gt_loss(gt: torch.tensor, depth: torch.tensor):
    z_error = depth - gt
    z_MSE = pixelwiseLoss(gt,depth)
    