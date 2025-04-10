from lib.base_models import VAE_Baseline
import torch
import numpy as np
import lib.utils as utils
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import time

class CoupledODE(VAE_Baseline):
	def __init__(self, encoder_z0, diffeq_solver, decoder, device, obsrv_std=None):

		super(CoupledODE, self).__init__(
			z0_prior=0,
			device=device, obsrv_std=obsrv_std
		)
 
		self.diffeq_solver = diffeq_solver  
		self.encoder_z0 = encoder_z0  
		self.decoder = decoder 
	
	 
	
	def get_reconstruction(self, batch_en, batch_de, args, loader, is_train=True, pe=None): 
		t1 = time.time()  
		if args.encoder != 'gnn':
			node, edge = self.encoder_z0(batch_en) 
		else: 
			node, edge = self.encoder_z0(batch_en.x, batch_en.edge_weight,
									batch_en.edge_index, batch_en.pos, batch_en.edge_time,
									batch_en.batch, batch_en.y, len(batch_en.topo))
			 
		#print('\n node edge', node.shape, edge.shape)
		sol_y = self.diffeq_solver(node, edge, batch_de["time_steps"]) 
		t2 = time.time()
		T, K, N, D = sol_y.shape 
		sol_y = sol_y.view(T, K*N, D)
		#print('\n', sol_y.shape)
		sol_y = sol_y.permute(1,0,2) # K*N x T x D
		#print('\n', sol_y.shape)
		sol_y = self.decoder(sol_y) # K*N x T x 1  
		#print('\n', sol_y.shape)
		sol_y = sol_y.view(K, N, T, -1)  
		#print('\nsol_y.shape', sol_y.shape)

		return sol_y, 0, t2 - t1
 
	 