import astra
import torch
import tomosipo as ts
import numpy as np

from .base import Algorithm, from_numpy

EPSILON = 1e-6


class SIRT(Algorithm):
    def __init__(self, x, y, angles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.angles = angles
        self.init_projector(x, y, angles)

    def setup(self):
        dev = torch.device("cuda")

        self._y = from_numpy(self.y, dev)
        self._x = from_numpy(self.x, dev)

        self._residual = torch.ones_like(self._y)
        self._x_dir = torch.ones_like(self._x)

        self.C = self.A.T(torch.ones(self.y.shape, device=dev))
        self.C[self.C < EPSILON] = np.Inf
        self.C.reciprocal_()

        self.R = self.A(torch.ones(self.x.shape, device=dev))
        self.R[self.R < EPSILON] = np.Inf
        self.R.reciprocal_()

    def iterate(self):
        self.A(self._x, out=self._residual)
        self._residual -= self._y

        curr_residual = torch.square(self._residual).sum().item()

        self._residual *= self.R
        self.A.T(self._residual, out=self._x_dir)
        self._x_dir *= self.C

        self._x -= self._x_dir
        # POCS NN
        self._x[self._x < 0] = 0
        self.x = self._x.detach().cpu().numpy().squeeze()

        return curr_residual


class MLEM(Algorithm):
    def __init__(self, x, y, angles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.angles = angles

        self.init_projector(x, y, angles)

    def setup(self):
        dev = torch.device("cuda")

        self._y = torch.clamp(from_numpy(self.y, dev), min=0)
        self._x = from_numpy(self.x, dev)

        # non-masked ratio had Boundary condition issues
        # that introduced instability in the corners of a  full image
        y_mask = self.A(torch.ones(self.x.shape, device=dev)) > 10
        self._y *= y_mask

        self.C = self.A.T(torch.ones_like(self._y)) + EPSILON
        self.C.reciprocal_()

        # containers
        self._y_cont = torch.ones_like(self._y)
        self._x_cont = torch.ones_like(self._x)

    def iterate(self):
        # y estimate
        self.A(self._x, out=self._y_cont)
        curr_residual = torch.square(self._y - self._y_cont).sum().item()

        # ratio = y/(y_est+eps)
        self._y_cont += EPSILON
        self._y_cont.reciprocal_()
        self._y_cont *= self._y

        # coeff = AT(ratio)*C
        self.A.T(self._y_cont, out=self._x_cont)
        self._x_cont *= self.C

        self._x *= self._x_cont
        self.x = self._x.detach().cpu().numpy().squeeze()
        return curr_residual


class CGNE(Algorithm):

    def __init__(self, x, y, angles, nn_step=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.angles = angles
        self.gamma0 = 0

        self.proximal_step = nn_step if nn_step is not None else -1
        self.proximal_counter = 0

        self.init_projector(x, y, angles)

    def setup(self):
        dev = torch.device("cuda")

        self._y = from_numpy(self.y, dev)
        self._x = from_numpy(self.x, dev)

        # containers
        self._r = torch.ones_like(self._y)
        self._q = torch.ones_like(self._y)

        self._s = torch.ones_like(self._x)
        self._p = torch.ones_like(self._x)

        self.init_conj()

    def init_conj(self):
        # self._r = self._y -self.A(self._x)
        self.A(self._x, out=self._r)
        self._r -= self._y
        self._r *= -1

        self.A.T(self._r, out=self._s)
        self.gamma0 = torch.square(self._s).sum()
        self._p = 1.0 * self._s

    def iterate(self):
        self.proximal_counter += 1

        self.A(self._p, out=self._q)
        alpha = self.gamma0 / torch.square(self._q).sum()

        self._r -= alpha * self._q
        # Doe to numerical instability, this may drift a bit
        # This means thast both the residual and the conjugate directionis off.
        # Reset and continue:
        resnorm = torch.square(self._r).sum().item()
        if self.loss and resnorm > self.loss[-1]:
            self.init_conj()
            return resnorm

        self._x += alpha * self._p

        if self.proximal_counter == self.proximal_step:
            self._x = torch.clamp(self._x, min=0)
            self.init_conj()
            self.proximal_counter = 0

        else:
            self.A.T(self._r, out=self._s)
            gamma = torch.square(self._s).sum()
            beta = gamma / self.gamma0

            self.gamma0 = gamma
            self._p *= beta
            self._p += self._s

        self.x = self._x.detach().cpu().numpy().squeeze()
        return torch.square(self._r).sum().item()
