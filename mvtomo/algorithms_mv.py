import astra
import torch
import tomosipo as ts
import numpy as np

from .base import Algorithm,from_numpy

class MLEM(Algorithm):
    def __init__(self, x, y, angles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y_l = y
        self.angles_l = angles

        self.init_paralell_list(x, y, angles)
        
    def setup(self):
        dev = torch.device("cuda")

        self._x = from_numpy(self.x, dev)
        self._y_l = [torch.clamp(from_numpy(y, dev), min=0) for y in self.y_l]

        # non-masked ratio had Boundary condition issues
        # that introduced instability in the corners of a  full image
        for A,y in zip(self.A,self._y_l):
            y*=A(torch.ones(A.domain_shape, device=dev))>0

        C_temp = [
            (A.T(torch.ones_like(y)) )
            for A, y in zip(self.A, self._y_l)
        ]

        self.C = torch.stack(C_temp, dim=0).sum(dim=0)+ ts.epsilon
        self.C.reciprocal_()

        # containers
        self._y_cont = [torch.ones_like(y) for y in self._y_l]
        self._x_cont = torch.ones_like(self._x)

    def iterate(self):
        
        # y_est_l = [A(self._x) for A in self.A_l]
        for A,y in zip(self.A, self._y_cont):
            A(self._x,out = y)

        l2_residual = 0
        for y, y_est in zip(self._y_l, self._y_cont):
            l2_residual+=torch.square(y_est-y).sum().item()

        # ratios = [y / (y_est + ts.epsilon) * y_mask for y,y_est,y_mask in zip(self._y_l, y_est_l, self._y_mask_l)]
        for y_est, y in zip(self._y_cont, self._y_l):
            y_est+=ts.epsilon
            y_est.reciprocal_()
            y_est*=y

        self._x_cont*=0
        for A,ratio in zip(self.A,self._y_cont):
            A.additive = True
            A.T(ratio, out = self._x_cont)
            A.additive = False

        self._x_cont*=self.C        
        self._x *=  self._x_cont

        self.x = self._x.detach().cpu().numpy().squeeze()
        return l2_residual
    
class OSEM(Algorithm):
    def __init__(self, x, y, angles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y_l = y
        self.angles_l = angles

        self.init_paralell_list(x, y, angles)

    def setup(self):
        dev = torch.device("cuda")

        self._x = from_numpy(self.x, dev)
        self._y_l = [torch.clamp(from_numpy(y, dev), min=0) for y in self.y_l]

        # non-masked ratio had Boundary condition issues
        # that introduced instability in the corners of a  full image
        for A,y in zip(self.A,self._y_l):
            y*=A(torch.ones(A.domain_shape, device=dev))>0

        self.C_l = [
            (A.T(torch.ones_like(y)) + ts.epsilon).reciprocal_()
            for A, y in zip(self.A, self._y_l)
        ]

        # containers
        self._y_cont = [torch.ones_like(y) for y in self._y_l]
        self._x_cont = torch.ones_like(self._x)

    def iterate(self):
        l2_residual = 0
        for y,y_tmp, C,A in zip(self._y_l,self._y_cont,self.C_l, self.A):
            A(self._x,y_tmp)
            l2_residual+=torch.square(y_tmp-y).sum().item()

            # ratio = (y / (y_est + ts.epsilon))     
            y_tmp+=ts.epsilon
            y_tmp.reciprocal_()
            y_tmp*=y

            # self._x *=  A.T(ratio)*C
            A.T(y_tmp, out = self._x_cont)
            self._x_cont*=C        
            self._x *=  self._x_cont

        self.x = self._x.detach().cpu().numpy().squeeze()

        return l2_residual

class CGNE(Algorithm):

    def __init__(self, x, y, angles, nn_step=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y_l = y
        self.angles_l = angles

        self.gamma0 = 0

        self.proximal_step = nn_step if nn_step is not None else -1
        self.proximal_counter = 0

    def setup(self):
        dev = torch.device("cuda")

        vg = ts.volume(shape=self.x.shape)
        pg_list = [
            ts.parallel(angles=angles, shape=y.shape[::2])
            for y, angles, in zip(self.y_l, self.angles_l)
        ] 
        self.A_l = [ts.operator(vg, pg) for pg in pg_list]

        self._y_l = [torch.from_numpy(y).to(dev) for y in self.y_l]
        self._x = torch.from_numpy(self.x).to(dev)
        
        self.init_conj()

    def init_conj(self):
        self._r = [y - A(self._x) for A,y in zip(self.A_l, self._y_l)] 
        self._s = [A.T(r) for A,r in zip(self.A_l,self._r)]

        _s = torch.stack(self._s, dim=0).sum(dim=0)
        self.gamma0 = (_s * _s).sum()
        self._p = 1.0 * _s

    def iterate(self):
        self.proximal_counter += 1

        self._q = [A(self._p) for A in self.A_l]
        q2_sum = sum([(q**2).sum() for q in self._q])
        alpha = self.gamma0 / q2_sum

        self._x += alpha * self._p
        self._r = [r-alpha*q for r,q in zip(self._r, self._q)]

        if self.proximal_counter == self.proximal_step:
            self._x = torch.clamp(self._x, min=0)
            self.init_conj()
            self.proximal_counter = 0

        else:
            self._s = [A.T(r) for A,r in zip(self.A_l,self._r)]
            _s = torch.stack(self._s, dim=0).sum(dim=0)
            gamma = (_s * _s).sum()
            beta = gamma / self.gamma0

            self.gamma0 = gamma
            self._p = _s + beta * self._p  # Corrected update

        self.x = self._x.detach().cpu().numpy().squeeze()
        return ((torch.stack(self._s, dim=0).sum(dim=0))**2).sum().item()
    
class CGNE2(Algorithm):

    def __init__(self, x, y, angles, nn_step=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y_l = y
        self.angles_l = angles

        self.gamma0 = 0

        self.proximal_step = nn_step if nn_step is not None else -1
        self.proximal_counter = 0

        self.init_paralell_list(x, y, angles)

    def setup(self):
        dev = torch.device("cuda")

        self._x = from_numpy(self.x, dev)
        self._y_l = [torch.clamp(from_numpy(y, dev), min=0) for y in self.y_l]

        # containers
        # containers
        self._r = [torch.ones_like(y) for y in self._y_l]
        self._q = [torch.ones_like(y) for y in self._y_l]

        self._s = torch.ones_like(self._x)
        self._p = torch.ones_like(self._x)
            
        self.init_conj()

    def init_conj(self):
        # self._r = [y - A(self._x) for A,y in zip(self.A_l, self._y_l)] 
        for A,residual, y in zip(self.A, self._r, self._y_l):
            A(self._x, out = residual)
            residual -= y
            residual*=-1

        # self._s = [A.T(r) for A,r in zip(self.A_l,self._r)]
        self._s*=0
        for A,residual in zip(self.A,self._r):
            A.additive = True
            A.T(residual, out = self._s)
            A.additive = False

        self.gamma0 = torch.square(self._s).sum()
        self._p = 1.0 * self._s

    def iterate(self):
        self.proximal_counter += 1

        # self._q = [A(self._p) for A in self.A_l]
        for A,q in zip(self.A, self._q):
            A(self._p, out = q)
            
        q2_sum = sum([torch.square(q).sum() for q in self._q])
        alpha = self.gamma0 / q2_sum

        self._x += alpha * self._p
        for r,q in zip(self._r, self._q):
            r-=alpha*q

        if self.proximal_counter == self.proximal_step:
            self._x = torch.clamp(self._x, min=0)
            self.init_conj()
            self.proximal_counter = 0
        else:
            self._s*=0
            for A,residual in zip(self.A,self._r):
                A.additive = True
                A.T(residual, out = self._s)
                A.additive = False
            gamma = torch.square(self._s).sum()
            beta = gamma / self.gamma0

            self.gamma0 = gamma
            self._p *= beta
            self._p+=self._s

        self.x = self._x.detach().cpu().numpy().squeeze()
        return sum([torch.square(r).sum() for r in self._r]).item()


class CGNE_astra(Algorithm):
    """Conjugate Gradient Normal Equations (CGNE) implementation."""

    def __init__(self, x, y, angles, nn_step=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.angles = angles
        self.gamma0 = 0

        self.proximal_step = nn_step if nn_step is not None else -1
        self.proximal_counter = 0

    def setup(self):

        n_slices, n_rows, n_cols = self.x.shape
        vol_geom = astra.create_vol_geom(n_rows, n_cols, n_slices)
        height, _, width = self.y.shape
        proj_geom1 = astra.create_proj_geom(
            "parallel3d", 1, 1, height, width, self.angles
        )

        def A(x):
            id_, _y = astra.create_sino3d_gpu(x, proj_geom1, vol_geom)
            # _y1 = ndi.gaussian_filter(_y1, (sigma1,0,sigma1), mode = 'nearest')
            astra.data3d.delete(id_)
            return _y

        def AT(y):
            id_, vol = astra.create_backprojection3d_gpu(y, proj_geom1, vol_geom)
            astra.data3d.delete(id_)
            return vol

        setattr(self, "A", A)
        setattr(self, "AT", AT)

        self.init_conj()

    def init_conj(self):
        self._r = self.y - self.A(self.x)
        self._s = self.AT(self._r)
        self.gamma0 = (self._s * self._s).sum()
        self._p = 1.0 * self._s

    def iterate(self):
        self.proximal_counter += 1

        self._q = self.A(self._p)
        alpha = self.gamma0 / (self._q * self._q).sum()

        self.x += alpha * self._p
        self._r -= alpha * self._q

        if self.proximal_counter == self.proximal_step:
            self.x = np.maximum(self.x, 0)
            self.init_conj()
            self.proximal_counter = 0

        else:
            self._s = self.AT(self._r)
            gamma = (self._s * self._s).sum()
            beta = gamma / self.gamma0

            self.gamma0 = gamma
            self._p = self._s + beta * self._p  # Corrected update

        return (self._r**2).sum().item()
