import numpy as np
from Dynamics.GeneralNet import GeneralNet

class Epidemic(GeneralNet):
    '''
    Create a set of methods for Ecology dynamics.
    '''
    def __init__(self, N=10, p=3, name='', topo='er', m=4, seed=42):
        GeneralNet.__init__(self, N, p, 'epi', name, topo, m, seed=seed)
         
    def f(self, x, param):
        return - param * x

    def f1(self, x, param):
        return - param.reshape(1,self.N).repeat(x.shape[0],axis=0)

    def g(self, x, y, param):
        return (1 - x) * y

    def g1(self, x, y, param):
        return - y * np.ones_like(x)

    def g2(self, x, y, param):
        return 1-x

    
     
    def dxdt(self, x,t, param, A=None):
        '''
        Compute the full dynamics f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        '''
        if A is None:
            A = self.A
        dxdt = self.f(x, param) + (1 - x) * (A @ x)
        return dxdt
 