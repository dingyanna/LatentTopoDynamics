import numpy as np
from Dynamics.GeneralNet import GeneralNet

class Popu(GeneralNet):
    def __init__(self, N, p, name, topo, m, seed):
        GeneralNet.__init__(self, N, p, 'popu', name, topo,m, seed=seed)
        self.gt = np.array([1,1,2])
    
    def f(self, x, param):
        f,h = param
        return - (x ** f)

    def f1(self, x, param):
        f,h = param
        return - f * (x ** (f-1))

    def g(self, x, y, param):
        f,h = param
        return np.ones_like(x) * (y ** h)   

    def g1(self, x, y, param):
        f,h = param
        return np.zeros_like(x)

    def g2(self, x, y, param):
        f,h = param  
        return np.ones_like(x) * h * (y ** (h-1))
 
    def dxdt(self, x,t, param, A=None):
        '''
        Compute the full dynamics f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        '''
        if A is None:
            A = self.A
        dxdt = self.f(x,param) + A @ self.g(x, x, param)
        return dxdt 
          