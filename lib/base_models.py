import torch.nn as nn
import torch
import lib.utils as utils
import numpy as np  
 

class VAE_Baseline(nn.Module):
	def __init__(self,
		z0_prior, device,
		obsrv_std = 0.01, 
		):

		super(VAE_Baseline, self).__init__()
		

		self.device = device

		self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

		self.z0_prior = z0_prior
 
	  
	def compute_all_losses(self, batch_dict_encoder,batch_dict_decoder, args, loader, kl_coef = 1.,istest=False, pe=None, source_X=None, source_y=None ):
		'''

		:param batch_dict_encoder:
		:param batch_dict_decoder: dict: 1. time 2. data: (K*N, T2, D)
		:param batch_dict_graph: #[K,T2,N,N], ground_truth graph with log normalization
		:param num_atoms:
		:param kl_coef:
		:return:
		''' 
		results = {}
		pred_node, encoder_time, ode_time= self.get_reconstruction(batch_dict_encoder,batch_dict_decoder, args, loader, is_train = (not istest), pe=pe)
		 
		true = batch_dict_decoder["data"]
		  
		mape_node = torch.abs((true - pred_node) / true)  
		loss_state = ((pred_node[~torch.isnan(true)] - true[~torch.isnan(true)]) ) ** 2
		pred_node[torch.isnan(true)] = torch.nan 
		loss_state = loss_state.mean() 
		loss = loss_state
		results["loss"] = loss 
		results["loss_state"] = loss_state   
		results["MAPE"] = torch.mean(mape_node[~torch.isnan(mape_node)]).data.item()  
		results["MAPE_ALL"] = mape_node.cpu().detach().numpy()
		results["encoder_time"] = encoder_time 
		results["ode_time"] = ode_time  
		return results, pred_node.cpu().detach().numpy() 
 