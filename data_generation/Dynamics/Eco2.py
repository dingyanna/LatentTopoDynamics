import numpy as np
from Dynamics.GeneralNet import GeneralNet
import warnings
warnings.filterwarnings('default') 

class Eco2(GeneralNet):
    '''
    Create a set of methods for Ecology dynamics.
    '''
    def __init__(self, N=10, p=3, name='', topo='er', m=4, seed=42):
        GeneralNet.__init__(self, N, p, 'eco2', name, topo, m, seed=seed)
        
    def f(self, x, param): 
        
        return x * (1 - x ** 2)

    def f1(self, x, param): 
        '''
        (1 - x**2) - 2 * (x ** 2)
        '''
        return 1 - 3 * (np.ones_like(x) ** 2) 

    def g(self, x, y, param):
         
        return x * y / (1 + y)

    def g1(self, x, y, param):  
        return np.ones_like(x) * y[0] / (1+y[0])
    
    
    def g2(self, x, y, param): 
        '''
        (x * (1+y) - x * y) / (denom ** 2)
        = x + xy - xy / (denom ** 2)
        = x / (denom ** 2)
        '''
        denom = 1+y
        return (x) / (denom ** 2)

    def dxdt(self, x,t, param, A=None):
        '''
        Compute the full dynamics f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        '''
        if A is None:
            A = self.A 
        dxdt = self.f(x, param) + x * (A @ ((x) / (1 + x))) 
        return dxdt
 
 