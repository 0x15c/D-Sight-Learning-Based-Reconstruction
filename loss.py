"""Depth supervision through the differentiable Poisson reconstruction."""

import torch
from torch import nn
from torch.nn import functional as F

from poisson_solver import PoissonSolver, height_to_grad


class ReconstructionLoss(nn.Module):
    def __init__(self, image_size, gradient_weight=0.0, boundary="dirichlet", reference_region=None):
        super().__init__()
        self.solver = PoissonSolver(*image_size, boundary=boundary, reference_region=reference_region)
        self.gradient_weight = gradient_weight

    def forward(self, gradients, target):
        # FFT and physical losses remain float32 even when the U-Net uses AMP.
        with torch.autocast(device_type=gradients.device.type, enabled=False):
            gradients = gradients.float()
            depth = self.solver(gradients)
            target = target.float()
            # Neumann integration determines shape up to one constant. Apply the
            # same height reference to supervision as to the reconstructed depth.
            depth_loss = F.mse_loss(depth, self.solver.reference(target))
            gradient_loss = depth_loss.new_zeros(())
            if self.gradient_weight:
                gradient_loss = F.mse_loss(gradients, height_to_grad(target))
            total = depth_loss + self.gradient_weight * gradient_loss
        return total, depth_loss, gradient_loss


def depth_gt_loss(gt, depth):
    return F.mse_loss(depth, gt)
