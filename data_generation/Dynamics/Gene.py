import numpy as np
from Dynamics.GeneralNet import GeneralNet

class Gene(GeneralNet):
    def __init__(self, N, p, name, topo, m, seed):
        GeneralNet.__init__(self, N, p, 'gene', name, topo,m, seed=seed)
        self.gt = np.array([1,1,2])
    
    def f(self, x, param):
        B,f,h = param
        return - B * (x ** f)

    def f1(self, x, param):
        B,f,h = param
        return - B * f * (x ** (f-1))

    def g(self, x, y, param):
        B,f,h = param
        return np.ones_like(x) *  ((y ** h) / (y ** h + 1))  

    def g1(self, x, y, param):
        B,f,h = param
        return np.zeros_like(x)

    def g2(self, x, y, param):
        B,f,h = param
        #num = y ** h 
        denom = y ** h + 1
        deriv = (h * (y ** (h-1))) / (denom ** 2)
        return np.ones_like(x) * deriv
 
    def dxdt(self, x,t, param, A=None):
        '''
        Compute the full dynamics f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        '''
        if A is None:
            A = self.A
        dxdt = self.f(x,param) + A @ self.g(x, x, param)
        return dxdt 
          