from abc import ABC, abstractmethod
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
from tqdm.auto import trange
import tomosipo as ts
import torch


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

    def init_paralell(self, x, y, angles):
        vg = ts.volume(shape=x.shape)
        pg = ts.parallel(angles=angles, shape=y.shape[::2])
        self.A = ts.operator(vg, pg)

    def init_paralell_list(self, x, y, angles):
        vg = ts.volume(shape=x.shape)
        pg_list = [
            ts.parallel(angles=angles, shape=y.shape[::2])
            for y, angles, in zip(self.y_l, self.angles_l)
        ]
        [self.init_paralell(x, y, a) for y, a in zip(y, angles)]
        self.A = [ts.operator(vg, pg) for pg in pg_list]

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
