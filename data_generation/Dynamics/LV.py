import numpy as np 
from Dynamics.GeneralNet import GeneralNet

class LV(GeneralNet): 
    def __init__(self, N, p, name, topo, m, seed):
        GeneralNet.__init__(self, N, p, 'lv', name, topo,m, seed=seed)

    def f(self, x, param ):
        alpha, theta = param[0], param[1]
        return x * (alpha - theta * x) 

    def f1(self, x, param ):
        alpha, theta = param[0], param[1]
        return alpha - 2 * theta * x

    def g(self, x, y, param ): 
        return - x * y

    def g1(self, x, y, param ): 
        return - y * np.ones_like(x)

    def g2(self, x, y, param ): 
        return - x  

    def dxdt(self, x,t, param, A=None ):
        if A is None: 
            A = self.A 
        X = np.repeat(x,self.N).reshape(self.N,self.N) 
        return self.f(x, param) + (A * self.g(X, X.T, param)).sum(1) 

      
 