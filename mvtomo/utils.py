import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from typing import Union


def getslice(data, dim, slicenr=None):
    """Get slice of a n-dimensional array

    Arguments:
        data {ndarray} -- Input array.
        dim {int} -- dimension of the slicing

    Keyword Arguments:
        slicenr {int} -- Slice number, if not assigned gets the middle

    Returns:
        [ndarray] -- Output array
    """

    if slicenr is None:
        slicenr = int(data.shape[dim] / 2)
    assert -1 < slicenr < data.shape[dim], f"Index {slicenr} is out of range"

    return np.take(data, slicenr, axis=dim)


def volshow(data, slices=(None, None, None)):
    """Show preview of volume

    Arguments:
        data {ndarray} -- Input array.

    Keyword Arguments:
        slices {tuple} -- Shown slices, default is middle (default: {(-1, -1, -1)})
    """

    ndim = data.ndim
    assert ndim == 3, "Volume must be a 3d ndarray"

    _ = plt.figure(figsize=(13, 5))
    axes = [plt.subplot(gsi) for gsi in gridspec.GridSpec(1, ndim)]
    images = [getslice(data, d, s) for d, s in zip(range(ndim), slices)]

    for axis, image in zip(axes, images):
        axis.imshow(image)


def plot_imgs(height=3, cmap="gray", clim=(None, None), colorbar=True, **kwargs):
    fig, axes = plt.subplots(
        nrows=1, ncols=len(kwargs), figsize=(height * len(kwargs), height)
    )
    if len(kwargs) == 1:
        axes = [axes]
    for ax, (k, v) in zip(axes, kwargs.items()):
        pcm = ax.imshow(v, cmap=cmap, clim=clim)
        if colorbar:
            fig.colorbar(pcm, ax=ax)
        ax.set_title(k)
    fig.tight_layout()


def add_poisson(absorption_images: np.ndarray, I0: Union[float, np.ndarray], seed=None):
    """Add Poisson noise to absorption images

    Args:
        absorption_images (np.ndarray): Array of absorption images.
        I0 (float or np.ndarray): Incident intensity (scalar or array matching `absorption_images`).
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Raises:
        ValueError: If `I0` is an array and does not match the shape of `absorption_images`.

    Returns:
        np.ndarray: Noisy absorption images.
    """
    if isinstance(I0, np.ndarray) and I0.shape != absorption_images.shape:
        raise ValueError(
            "I0 must be a scalar or have the same shape as absorption_images."
        )

    I_image = I0 * np.exp(-absorption_images)
    np.random.seed(seed)
    I_image = np.random.poisson(I_image)
    I_image[I_image < 1] = 1  # Avoid log(0) issues
    return -np.log(1.0 * I_image / I0)
