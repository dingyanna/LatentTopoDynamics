from lib.gnn_models import Node_GCN, Node_GAT, GNN
from lib.latent_ode import CoupledODE
from lib.encoder_decoder import *
from lib.diffeq_solver import DiffeqSolver,CoupledODEFunc
from lib.utils import print_parameters

encoder_dict = {
	'mlp': Encoder_MLP,
	'gru_last': Encoder_GRU1, 
	'nri': Encoder_NRI,
	'gnn': GNN,
	'gt': Encoder_Transformer
}

ode_func_dict = {
	'gcn': Node_GCN,
	'gat': Node_GAT
}

def create_CoupledODE_model(args, input_dim, z0_prior, obsrv_std, device ):
	 

	encoder = encoder_dict[args.encoder](input_dim, args.ode_dims, args)
	node_ode_func_net = ode_func_dict[args.ode_type](dim_in=args.ode_dims, dim_hid=args.ode_dims, num_node=args.num_atoms, num_head=args.num_head, dropout=args.dropout ).to(device)
	decoder = Decoder(args.ode_dims, args.output_dim)
	 
	coupled_ode_func = CoupledODEFunc(
		node_ode_func_net=node_ode_func_net, 
		device=device,
		num_atom = args.num_atoms,dropout=args.dropout) 

	diffeq_solver = DiffeqSolver(coupled_ode_func, args.solver, args=args , odeint_rtol=1e-2, odeint_atol=1e-2, device=device)
 
	model = CoupledODE(
		encoder_z0=encoder,    
		diffeq_solver = diffeq_solver ,  
		decoder = decoder,
		device = device,
		obsrv_std = obsrv_std).to(device) 

	print_parameters(model) 

	return model
