import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import lib.utils as utils
import torch.nn.functional as F 
#from torch_scatter import scatter_add 
 


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method,args,
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.device = device
        self.ode_func = ode_func
        self.args = args
        self.num_atoms = args.num_atoms 

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol



    def forward(self, node, edge, time_steps_to_predict):
        '''
        node: num_batch x num_node x d
        edge: num_batch x num_node x num_node
        ''' 
        # Results
        self.ode_func.edge = edge
        pred_y = odeint(self.ode_func, node, time_steps_to_predict,
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method) #[time_length, K*N + K*N*N, D]
 
        return pred_y 
    
     


class CoupledODEFunc(nn.Module):
    def __init__(self, node_ode_func_net, num_atom, dropout, device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(CoupledODEFunc, self).__init__()

        self.device = device
        self.node_ode_func_net = node_ode_func_net  #input: x, edge_index 
        self.num_atom = num_atom
        self.nfe = 0
        self.dropout = nn.Dropout(dropout)


    def forward(self, t_local, z, backwards = False):
        """
        z: num_batch x num_node x (d + num_node) 
        """
        self.nfe += 1   
        return self.node_ode_func_net(z, self.edge) 

     
