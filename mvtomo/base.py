from abc import ABC, abstractmethod
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
from tqdm.auto import trange
import tomosipo as ts
import torch

import ncxtutils
from .voxproj import VoxelOperator


def factory_tomosipo():
    def func(x, y, angles):
        vg = ts.volume(shape=x.shape)
        pg = ts.parallel(angles=angles, shape=y.shape[::2])
        return ts.operator(vg, pg)

    return func


def factory_voxelproj(z_order):
    def func(x, y, angles):

        return VoxelOperator(angles, z_order=z_order, x_shape=x.shape, y_shape=y.shape)

    return func


initfunc = {
    "tomosipo": factory_tomosipo,
    "voxelproj": factory_voxelproj,
}


class Operator:
    def __init__(self):
        self.projfactory = None

    def set(self, name, **kwargs):
        assert (
            name in initfunc
        ), f"Name {name} not in available initializers {list(initfunc.keys())}"
        self.projfactory = initfunc[name](**kwargs)

    def projector(self, *args, **kwargs):
        return self.projfactory(*args, **kwargs)

    def init_paralell_list(self, x, y, angles):
        return [self.projector(x, y, a) for y, a in zip(y, angles)]


operator = Operator()


def set_backend(name, **kwargs):
    operator.set(name, **kwargs)


def get_operator(*args, **kwargs):
    return operator.projector(*args, **kwargs)


def from_numpy(x, device):
    """
    Converts a NumPy array to a PyTorch tensor with the correct dtype and memory layout.

    Ensures:
    - The dtype is compatible with PyTorch (`float32` by default for tensors).
    - The strides are contiguous to avoid issues with in-place operations.

    Args:
        x (numpy.ndarray): Input NumPy array.
        device (torch.device): Target device for the tensor (e.g., "cuda" or "cpu").

    Returns:
        torch.Tensor: A PyTorch tensor with `float32` dtype and contiguous memory layout.
    """
    return torch.Tensor(np.ascontiguousarray(x)).to(device)


class Algorithm(ABC):
    """Abstract base class for reconstruction algorithms."""

    def __init__(self, disable_tqdm=False):
        self.disable_tqdm = disable_tqdm
        self.loss = []
        self.A = None

    def init_projector(self, x, y, angles):
        self.A = operator.projector(x, y, angles)

    def init_projector_list(self, x, y, angles):
        self.A = [operator.projector(x, yi, ai) for yi, ai in zip(y, angles)]

    @abstractmethod
    def setup(self):
        """Set up algorithm-specific parameters before iteration."""
        pass

    @abstractmethod
    def iterate(self) -> float:
        """Perform one iteration of the reconstruction algorithm."""
        return 0.0

    def metric(self, x, oracle):
        """
        Default PSNR metric
        """
        data_range = np.percentile(oracle, 99)
        return peak_signal_noise_ratio(x, oracle, data_range=data_range)

    def set_metric(self, func):
        setattr(self, "metric", func)

    def __call__(self, n_iter, oracle=None, stop_at_best=False):
        """Run the reconstruction algorithm for a given number of iterations.

        Args:
            n_iter (int): The maximum number of iterations to perform.
            oracle (array-like, optional): A reference image or volume to compare against for quality assessment.
            If provided, the method tracks the best reconstruction based on the
            specified metric.. Defaults to None.
            stop_at_best (bool, optional):  If True, stops early when the reconstruction quality declines for
            multiple iterations, returning the best recorded reconstruction.. Defaults to False.

        Returns:
            numpy.ndarray, list: If oracle is True:
                    (The best reconstruction found during iterations.,
                    A list of metric values computed over the iterations.)
            numpy.ndarray: Reconstruction after n_iter iterations.
        """
        self.setup()
        bar = trange(n_iter, leave=False, disable=self.disable_tqdm)

        if oracle is not None:
            metric_list = []
            x_best = np.copy(self.x)

        for i in bar:
            self.loss.append(self.iterate())

            if oracle is not None:
                metric_list.append(self.metric(self.x, oracle))

                if np.argmax(metric_list) == i:
                    x_best = np.copy(self.x)
                elif np.argmax(metric_list) < i - 2 and stop_at_best:
                    return x_best, metric_list

        if oracle is not None:
            return x_best, metric_list
        return np.copy(self.x)
