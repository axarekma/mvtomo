import torch
import torch.nn.functional as F
import tomosipo as ts

import numpy as np
from tomosipo.Operator import _to_link
from scipy import ndimage as ndi
import warnings
from tomosipo.Operator import Operator, BackprojectionOperator
from tomosipo.Operator import _to_link, direct_bp, direct_fp, Data


def gaussian_kernel1d(size: int, sigma: float):
    """Generate a 1D Gaussian kernel."""
    coords = torch.arange(size) - size // 2
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= kernel.sum()  # Normalize
    return kernel


def apply_nearest_neighbor_padding(tensor: torch.Tensor, kernel_size: int, axis: int):
    """Apply nearest neighbor padding along the specified axis."""
    pad = kernel_size // 2
    padding = [0, 0, 0, 0, 0, 0]

    if axis == 0:  # Depth axis
        padding[0] = pad
        padding[1] = pad
    elif axis == 0:  # Depth axis
        padding[2] = pad
        padding[3] = pad
    elif axis == 2:  # Width axis
        padding[4] = pad
        padding[5] = pad

    return F.pad(tensor, padding, mode="replicate")


def gaussian_blur_sep_nn(tensor: torch.Tensor, kernel_size=5, sigma=1.0):
    """Apply a 3D Gaussian blur over axes (0,2) while keeping axis 1 intact."""
    kernel_1d = gaussian_kernel1d(kernel_size, sigma).to(tensor.device)
    kernel_1d = kernel_1d.view(1, 1, 1, 1, kernel_size)

    # Reshape tensor for Conv3d: (batch=1, channels=D, H, W) → (1, D, H, W)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
    tensor_padded = apply_nearest_neighbor_padding(tensor, kernel_size, axis=0)
    modified = F.conv3d(tensor_padded, kernel_1d).permute((0, 1, 3, 4, 2))
    tensor_padded = apply_nearest_neighbor_padding(modified, kernel_size, axis=0)
    modified = F.conv3d(tensor_padded, kernel_1d).permute((0, 1, 4, 2, 3))
    return modified.squeeze(0).squeeze(0)


class GaussianOperator(Operator):
    """
    A modified operator that applies a Gaussian blur to projections
    during forward and backward projection.

    Parameters:
    -----------
    A : Operator
        An instance of the base `Operator` class that provides forward
        and backward projection functionality.
    sigma : float
        The standard deviation of the Gaussian filter.
    n_sigma : int, optional (default=4)
        Determines the kernel size as `2 * n_sigma * sigma + 1`.
    """

    def __init__(self, A, sigma, n_sigma=3):
        # Copy all attributes from original A
        for attr, value in A.__dict__.items():
            setattr(self, attr, value)

        self.sigma = sigma
        self.n_sigma = n_sigma
        self.kernel_size = int(n_sigma * sigma) * 2 + 1

        self._transpose = BackprojectionOperator(self)

    def conv_np(self, volume):
        return ndi.gaussian_filter(volume, (self.sigma, 0, self.sigma), mode="nearest")

    def conv_pytorch(self, volume):
        return gaussian_blur_sep_nn(
            volume, kernel_size=self.kernel_size, sigma=self.sigma
        )

    def _fp(self, volume, out=None):
        if out is not None and isinstance(out, np.ndarray):
            warnings.warn("np.ndarray inplace operations not supported]")

        vlink = _to_link(self.astra_compat_vg, volume)

        if out is not None:
            plink = _to_link(self.astra_compat_pg, out)
        else:
            if self.additive:
                plink = vlink.new_zeros(self.range_shape)
            else:
                plink = vlink.new_empty(self.range_shape)

        direct_fp(self.astra_projector, vlink, plink, additive=self.additive)
        if isinstance(plink.data, np.ndarray):
            plink.data[:] = self.conv_np(plink.data)
        elif isinstance(plink.data, torch.Tensor):
            plink.data[:] = self.conv_pytorch(plink.data)

        if isinstance(volume, Data):
            return ts.data(self.projection_geometry, plink.data)
        else:
            return plink.data

    def _bp(self, projection, out=None):
        if out is not None and isinstance(out, np.ndarray):
            warnings.warn("np.ndarray inplace operations not supported]")

        plink = _to_link(self.astra_compat_pg, projection)

        if out is not None:
            vlink = _to_link(self.astra_compat_vg, out)
        else:
            if self.additive:
                vlink = plink.new_zeros(self.domain_shape)
            else:
                vlink = plink.new_empty(self.domain_shape)

        if isinstance(plink.data, np.ndarray):
            plink.data[:] = self.conv_np(plink.data)
        elif isinstance(plink.data, torch.Tensor):
            plink.data[:] = self.conv_pytorch(plink.data)
        direct_bp(
            self.astra_projector,
            vlink,
            plink,
            additive=self.additive,
        )

        if isinstance(projection, Data):
            return ts.data(self.volume_geometry, vlink.data)
        else:
            return vlink.data
