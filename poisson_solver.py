"""
poisson_reconstruct.py
Fast Poisson Reconstruction in Python

Copyright (c) 2014 Jack Doerner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def height_to_grad(height):
    """BHW height (mm) -> B2HW forward pixel differences, ordered x, y."""
    dx = F.pad(height[..., 1:] - height[..., :-1], (0, 1))
    dy = F.pad(height[..., 1:, :] - height[..., :-1, :], (0, 0, 0, 1))
    return torch.stack((dx, dy), dim=1)


def _dst1(x):
    """Orthonormal DST-I on the last axis via an odd extension and FFT."""
    n = x.shape[-1]
    zero = torch.zeros_like(x[..., :1])
    extended = torch.cat((zero, x, zero, -x.flip(-1)), dim=-1)
    return -torch.fft.rfft(extended, dim=-1).imag[..., 1:n + 1] / math.sqrt(2 * (n + 1))


def _dst2(x):
    return _dst1(_dst1(x).transpose(-2, -1)).transpose(-2, -1)


def _dct1(x, cosine, sine, normalization, inverse=False):
    """Orthonormal DCT-II / inverse DCT-III using a real FFT of length 2N."""
    n = x.shape[-1]
    cosine, sine, normalization = (v.to(x.dtype) for v in (cosine, sine, normalization))
    if inverse:
        scaled = x * normalization
        spectrum = torch.complex(scaled * cosine, scaled * sine)
        spectrum = F.pad(spectrum, (0, 1))  # Mirrored signal has zero Nyquist term.
        return torch.fft.irfft(spectrum, n=2 * n, dim=-1)[..., :n]
    spectrum = torch.fft.rfft(torch.cat((x, x.flip(-1)), dim=-1), dim=-1)[..., :n]
    return (spectrum.real * cosine + spectrum.imag * sine) / normalization


class PoissonSolver(nn.Module):
    """Batched, differentiable integration of forward pixel differences.

    Input: B x 2 x H x W, [forward x difference, forward y difference].
    Output: B x H x W, in the same units as the supplied pixel differences.
    The last column of gx and last row of gy are unused.

    dirichlet: legacy zero-height outer rim, integrated with a DST-I.
    neumann: minimize the squared gradient error on every horizontal/vertical
    image edge. Border equations include the supplied flux; they do NOT impose
    zero slope. A DCT-II diagonalizes this free-boundary graph Laplacian.
    A constant shift sets the reference patch [x,y,width,height] mean to zero;
    reference_region=None uses the outer perimeter's mean, counting corners once.
    This fixes only the integration offset, not the height of individual pixels.
    FFTs run in float32 (or float64 for numerical tests), including under AMP.
    """

    def __init__(self, height, width, boundary="dirichlet", reference_region=None):
        super().__init__()
        self.boundary = boundary
        self.reference_region = reference_region
        if boundary == "dirichlet":
            y = torch.arange(1, height - 1, dtype=torch.float64)
            x = torch.arange(1, width - 1, dtype=torch.float64)
            eigenvalues = (2 * torch.cos(math.pi * y / (height - 1))[:, None]
                           + 2 * torch.cos(math.pi * x / (width - 1))[None, :] - 4)
            self.register_buffer("eigenvalues", eigenvalues.float())
        elif boundary == "neumann":
            y = torch.arange(height, dtype=torch.float64)
            x = torch.arange(width, dtype=torch.float64)
            eigenvalues = (2 * torch.cos(math.pi * y / height)[:, None]
                           + 2 * torch.cos(math.pi * x / width)[None, :] - 4)
            eigenvalues[0, 0] = 1  # The additive constant is fixed after inversion.
            self.register_buffer("eigenvalues", eigenvalues)
            for axis, size in (("x", width), ("y", height)):
                phase = torch.arange(size, dtype=torch.float64) * (math.pi / (2 * size))
                normalization = torch.full((size,), math.sqrt(2 * size), dtype=torch.float64)
                normalization[0] *= math.sqrt(2)
                self.register_buffer(f"{axis}_cosine", phase.cos())
                self.register_buffer(f"{axis}_sine", phase.sin())
                self.register_buffer(f"{axis}_normalization", normalization)
        else:
            raise ValueError("Poisson boundary must be 'dirichlet' or 'neumann'.")

    def _cosine_transform(self, value, inverse=False):
        value = _dct1(value, self.x_cosine, self.x_sine, self.x_normalization, inverse)
        value = _dct1(value.transpose(-2, -1), self.y_cosine, self.y_sine,
                      self.y_normalization, inverse)
        return value.transpose(-2, -1)

    def reference(self, height):
        """Apply the same constant-height reference to predictions and targets."""
        if self.boundary == "dirichlet":
            return height
        if self.reference_region is not None:
            x, y, width, region_height = self.reference_region
            offset = height[..., y:y + region_height, x:x + width].mean(dim=(-2, -1), keepdim=True)
        else:
            h, w = height.shape[-2:]
            border_sum = (height[..., 0, :].sum(-1) + height[..., -1, :].sum(-1)
                          + height[..., 1:-1, 0].sum(-1) + height[..., 1:-1, -1].sum(-1))
            offset = border_sum[..., None, None] / (2 * h + 2 * w - 4)
        return height - offset

    def forward(self, gradients):
        with torch.autocast(device_type=gradients.device.type, enabled=False):
            g = gradients if gradients.dtype == torch.float64 else gradients.float()
            gx, gy = g[:, 0], g[:, 1]
            if self.boundary == "dirichlet":
                divergence = (gx[:, 1:-1, 1:-1] - gx[:, 1:-1, :-2]
                              + gy[:, 1:-1, 1:-1] - gy[:, :-2, 1:-1])
                interior = _dst2(_dst2(divergence) / self.eigenvalues.to(g.dtype))
                return F.pad(interior, (1, 1, 1, 1))
            # -D^T g for edges that actually connect image pixels. The endpoint
            # terms supply the boundary flux and retain constant-slope planes.
            px, py = gx[..., :-1], gy[..., :-1, :]
            divergence = (F.pad(px, (0, 1)) - F.pad(px, (1, 0))
                          + F.pad(py, (0, 0, 0, 1)) - F.pad(py, (0, 0, 1, 0)))
            coefficients = self._cosine_transform(divergence) / self.eigenvalues.to(g.dtype)
            coefficients[..., 0, 0] = 0
            return self.reference(self._cosine_transform(coefficients, inverse=True))


def poisson_reconstruct(grady, gradx, boundarysrc):
    """Compatibility with the old NumPy API, including nonzero boundaries.

    Live reconstruction uses PoissonSolver directly and stays on the GPU.
    """
    gx = torch.as_tensor(gradx, dtype=torch.float32)
    gy = torch.as_tensor(grady, dtype=torch.float32)
    boundary = torch.as_tensor(boundarysrc, dtype=torch.float32).clone()
    boundary[1:-1, 1:-1] = 0
    gradients = torch.stack((gx, gy))[None] - height_to_grad(boundary[None])
    depth = PoissonSolver(*boundary.shape)(gradients)[0] + boundary
    return depth.numpy()
