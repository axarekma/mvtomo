import tomosipo as ts
import torch
from torch.fft import rfft, irfft
import numpy as np

from .base import from_numpy


def ram_lak(n):
    """Compute Ram-Lak filter in real space

    Computes a real space Ram-Lak filter optimized w.r.t. discretization bias
    introduced if a naive ramp function is used to filter projections in
    reciprocal space. For details, see section 3.3.3 in Kak & Staley,
    "Principles of Computerized Tomographic Imaging", SIAM, 2001.

    :param n: `int`
        Length of the filter.

    :returns:
        Real space Ram-Lak filter of length n.
    :rtype: `torch.tensor`
    """

    filter = torch.zeros(n)
    filter[0] = 0.25
    # even indices are zero
    # for odd indices j, filter[j] equals
    #   -1 / (pi * j) ** 2,          when 2 * j <= n
    #   -1 / (pi * (n - j)) ** 2,    when 2 * j >  n
    odd_indices = torch.arange(1, n, 2)
    cond = 2 * odd_indices > n
    odd_indices[cond] = n - odd_indices[cond]
    filter[1::2] = -1 / (np.pi * odd_indices) ** 2

    return filter


FILTER_BANK = {"ram_lak": ram_lak}


def filter_sino(y, filter=None, padded=True, batch_size=10, overwrite_y=False):
    """Filter sinogram for use in FBP

    :param y: `torch.tensor`
        A three-dimensional tensor in sinogram format (height, num_angles, width).

    :param filter: `torch.tensor` (optional)
        If not specified, the ram-lak filter is used. This should be
        one-dimensional tensor that is as wide as the sinogram `y`.

    :param padded: `bool`
        By default, the reconstruction is zero-padded as it is
        filtered. Padding can be skipped by setting `padded=False`.

    :param batch_size: `int`
        Specifies how many projection images will be filtered at the
        same time. Increasing the batch_size will increase the used
        memory. Computation time can be marginally improved by
        tweaking this parameter.

    :param overwrite_y: `bool`
        Specifies whether to overwrite y with the filtered version
        while running this function. Choose `overwrite_y=False` if you
        still want to use y after calling this function. Choose
        `overwrite_y=True` if you would otherwise run out of memory.

    :returns:
        A sinogram filtered with the provided filter.
    :rtype: `torch.tensor`
    """

    original_width = y.shape[-1]
    if padded:
        filter_width = 2 * original_width
    else:
        filter_width = original_width

    filter_real = FILTER_BANK[filter](filter_width)
    filter_dev = filter_real.to(y.device)
    filter_rfft = rfft(filter_dev)

    # Filter the sinogram in batches
    def filter_batch(batch):
        # Compute real FFT using zero-padding of the signal
        batch_rfft = rfft(batch, n=filter_width)
        # Filter the sinogram using complex multiplication:
        batch_rfft *= filter_rfft
        # Invert fourier transform.
        # Make sure inverted data matches the shape of y (for
        # sinograms with odd width).
        batch_filtered = irfft(batch_rfft, n=filter_width)
        # Remove padding
        return batch_filtered[..., :original_width]

    if overwrite_y:
        y_filtered = y
    else:
        y_filtered = torch.empty_like(y)

    for batch_start in range(0, y.shape[1], batch_size):
        batch_end = min(batch_start + batch_size, y.shape[1])
        batch = y[:, batch_start:batch_end, :]
        y_filtered[:, batch_start:batch_end, :] = filter_batch(batch)

    return y_filtered


class FBP:
    def __init__(self, x, y, angles):
        dev = torch.device("cuda")
        self.x = from_numpy(x, dev)
        self.y = from_numpy(y, dev)
        self.angles = angles

        vg = ts.volume(shape=x.shape)
        pg = ts.parallel(angles=angles, shape=y.shape[::2])
        self.A = ts.operator(vg, pg)

    def __call__(self, padded=True, filter="ram_lak", batch_size=10, overwrite_y=False):
        y_filtered = filter_sino(
            self.y,
            filter=filter,
            padded=padded,
            batch_size=batch_size,
            overwrite_y=overwrite_y,
        )
        rec = self.A.T(y_filtered)

        vg, pg = self.A.astra_compat_vg, self.A.astra_compat_pg

        pixel_height = pg.det_size[0] / pg.det_shape[0]
        voxel_volume = np.prod(np.array(vg.size / np.array(vg.shape)))
        scaling = (np.pi / pg.num_angles) * pixel_height / voxel_volume
        rec *= scaling

        return rec.detach().cpu().numpy().squeeze()
