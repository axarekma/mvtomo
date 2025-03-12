import astra
import torch
import tomosipo as ts
import numpy as np

from .base import Algorithm,from_numpy
    

class SIRT(Algorithm):    
    def __init__(self, x,y,angles,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.x = x
        self.y = y
        self.angles = angles
        self.init_paralell(x,y,angles)

    def setup(self):
        dev = torch.device("cuda")       

        self._y =  from_numpy(self.y, dev)
        self._x =  from_numpy(self.x, dev)

        self._residual = torch.ones_like(self._y)
        self._x_dir = torch.ones_like(self._x)

        self.C = self.A.T(torch.ones(self.A.range_shape, device=dev))
        self.C[self.C < ts.epsilon] = np.Inf
        self.C.reciprocal_()

        self.R = self.A(torch.ones(self.A.domain_shape, device=dev))
        self.R[self.R < ts.epsilon] = np.Inf
        self.R.reciprocal_()

    def iterate(self):
        self.A(self._x, out=self._residual)
        self._residual -= self._y

        curr_residual = torch.square(self._residual).sum().item()

        self._residual*=self.R
        self.A.T(self._residual, out=self._x_dir )
        self._x_dir*=self.C

        self._x -= self._x_dir
        # POCS NN
        self._x[self._x<0]=0
        self.x = self._x.detach().cpu().numpy().squeeze()

        return curr_residual
    
   
class MLEM(Algorithm):    
    def __init__(self, x,y,angles,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.x = x
        self.y = y
        self.angles = angles

        self.init_paralell(x,y,angles)

    def setup(self):
        dev = torch.device("cuda")

        self._y =  torch.clamp(from_numpy(self.y, dev), min=0)
        self._x =  from_numpy(self.x, dev)
        
        # non-masked ratio had Boundary condition issues
        # that introduced instability in the corners of a  full image
        y_mask = self.A(torch.ones(self.A.domain_shape, device=dev))>0
        self._y *=y_mask

        self.C = self.A.T(torch.ones_like(self._y ))+ts.epsilon
        self.C.reciprocal_()

        # containers
        self._y_cont = torch.ones_like(self._y)
        self._x_cont = torch.ones_like(self._x)


    def iterate(self):
        # y estimate
        self.A(self._x, out=self._y_cont)
        curr_residual = torch.square(self._y-self._y_cont).sum().item()

        # ratio = y/(y_est+eps)
        self._y_cont+=ts.epsilon
        self._y_cont.reciprocal_()
        self._y_cont*=self._y

        # coeff = AT(ratio)*C
        self.A.T(self._y_cont, out= self._x_cont)
        self._x_cont*= self.C

        self._x *= self._x_cont
        self.x = self._x.detach().cpu().numpy().squeeze()
        return curr_residual
    
   
class CGNE(Algorithm):
    
    def __init__(self, x,y,angles,nn_step = 3,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.x = x
        self.y = y
        self.angles = angles
        self.gamma0 = 0  

        self.proximal_step = nn_step if nn_step is not None else -1
        self.proximal_counter = 0

        self.init_paralell(x,y,angles)

    def setup(self):
        dev = torch.device("cuda")

        self._y =  from_numpy(self.y,dev)
        self._x =  from_numpy(self.x,dev)

        # containers
        self._r = torch.ones_like(self._y) 
        self._q = torch.ones_like(self._y)
        
        self._s = torch.ones_like(self._x)
        self._p = torch.ones_like(self._x)

        self.init_conj()

    def init_conj(self):
        # self._r = self._y -self.A(self._x)
        self.A(self._x, out = self._r)
        self._r -= self._y
        self._r*=-1

        self.A.T(self._r, out = self._s)
        self.gamma0 = torch.square(self._s).sum()
        self._p = 1.0*self._s  

    def iterate(self):
        self.proximal_counter+=1

        self.A(self._p, out = self._q)
        alpha = self.gamma0 / torch.square(self._q).sum()

        self._x += alpha * self._p
        self._r -= alpha * self._q

        if self.proximal_counter==self.proximal_step:
            self._x = torch.clamp(self._x, min=0)
            self.init_conj()
            self.proximal_counter=0

        else:
            self.A.T(self._r, out = self._s)
            gamma = torch.square(self._s).sum()
            beta = gamma / self.gamma0

            self.gamma0 = gamma
            self._p *= beta
            self._p+=self._s

        self.x = self._x.detach().cpu().numpy().squeeze()
        return torch.square(self._r).sum().item()
    
class CGNE_astra(Algorithm):
    
    def __init__(self, x,y,angles,nn_step = 3,*args,**kwargs):
        super().__init__(*args,**kwargs)
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
        proj_geom1 = astra.create_proj_geom("parallel3d", 1, 1, height, width, self.angles)
        
        def A(x):
            id_, _y = astra.create_sino3d_gpu(x, proj_geom1, vol_geom)
            # _y1 = ndi.gaussian_filter(_y1, (sigma1,0,sigma1), mode = 'nearest')
            astra.data3d.delete(id_)
            return _y
        def AT(y):
            id_, vol = astra.create_backprojection3d_gpu(y, proj_geom1, vol_geom)
            astra.data3d.delete(id_)
            return vol
        setattr(self, 'A', A)
        setattr(self, 'AT', AT)

        self.init_conj()

    def init_conj(self):
        self._r = self.y - self.A(self.x)
        self._s = self.AT(self._r)
        self.gamma0 = (self._s * self._s).sum()
        self._p = 1.0*self._s  

    def iterate(self):
        self.proximal_counter+=1

        self._q = self.A(self._p)
        alpha = self.gamma0 / (self._q * self._q).sum()

        self.x += alpha * self._p
        self._r -= alpha * self._q

        if self.proximal_counter==self.proximal_step:
            self.x = np.maximum(self.x, 0)
            self.init_conj()
            self.proximal_counter=0

        else:
            self._s = self.AT(self._r)
            gamma = (self._s * self._s).sum()
            beta = gamma / self.gamma0

            self.gamma0 = gamma
            self._p = self._s + beta * self._p  # Corrected update

        return (self._r**2).sum().item()