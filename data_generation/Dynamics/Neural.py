import numpy as np 
from Dynamics.GeneralNet import GeneralNet

class Neural(GeneralNet): 

    def __init__(self, N, p, name, topo, m, seed):
        GeneralNet.__init__(self, N, p, 'wc', name, topo,m, seed=seed)
        self.gt = np.array([1,1])
    def f(self, x, param ):
        return -x 
    
    def f1(self, x, param):
        return -1 * np.ones_like(x)

    def g(self, x, y, param ):
        tau, mu = param   
        return np.ones_like(x) / (1 + np.exp(- tau * (y - mu))) 

    def g1(self, x, y, param ):
        tau, mu = param   
        return np.zeros_like(x)
 
    def g2(self, x, y, param ):
        tau, mu = param   
        denom = 1 + np.exp(- tau * (y - mu))
        denom_y = np.exp(- tau * (y - mu)) * (- tau)
        deriv = - denom_y / (denom ** 2)
        return np.ones_like(x) * deriv
    
    def dxdt(self, x,t, param, A=None ): 
        if A is None: 
            A = self.A 
        return self.f(x, param) + A @ self.g(x, x, param)

    