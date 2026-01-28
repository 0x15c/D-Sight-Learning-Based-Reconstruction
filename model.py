import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# trying to employ U-Net to predict a gradient map
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.grad_head = nn.Conv2d(base_channels, 2, kernel_size=3, padding=1)

    @staticmethod
    def _match_size(src, ref):
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat([self._match_size(dec3, enc3), enc3], dim=1))
        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([self._match_size(dec2, enc2), enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([self._match_size(dec1, enc1), enc1], dim=1))
        return self.grad_head(dec1)


class Gradient2D(nn.Module):
    def __init__(self, in_channels=2, base_channels=16):
        super().__init__()
        self.unet = UNet2D(in_channels=in_channels, base_channels=base_channels)

    def forward(self, img_RGB):
        grad = self.unet(img_RGB)
        # returning [H x W x 2] tensor
        return grad

def getDepth(grad_hw2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    assert grad_hw2.ndim == 3 and grad_hw2.shape[-1] == 2
    H, W, _ = grad_hw2.shape
    device = grad_hw2.device
    dtype = grad_hw2.dtype

    p = grad_hw2[..., 0]
    q = grad_hw2[..., 1]

    P = torch.fft.fft2(p)
    Q = torch.fft.fft2(q)

    # frequencies in radians per pixel (periodic)
    kx = 2.0 * math.pi * torch.fft.fftfreq(W, d=1.0, device=device).to(dtype)  # (W,)
    ky = 2.0 * math.pi * torch.fft.fftfreq(H, d=1.0, device=device).to(dtype)  # (H,)
    KX, KY = torch.meshgrid(kx, ky, indexing="xy")  # (W,H) in xy indexing

    # make to (H,W)
    KX = KX.T
    KY = KY.T

    denom = (KX**2 + KY**2)
    denom[0, 0] = 1.0

    Z = (1j * KX * P + 1j * KY * Q) / (denom + eps)
    Z[0, 0] = 0.0 + 0.0j

    z = torch.fft.ifft2(Z).real
    return z

class Depth_Reconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad = Gradient2D()
    def forward(self, img_RGB):
        grad = self.grad(img_RGB)
        z = getDepth(grad)
        return z