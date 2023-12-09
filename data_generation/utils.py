import numpy as np
import networkx as nx 
import random   
import os  
from Dynamics.Gene import Gene 
from Dynamics.Epidemic import Epidemic
 
from Dynamics.LV import LV 
from Dynamics.Neural import Neural 

from Dynamics.Eco2 import Eco2  
from Dynamics.Popu import Popu  

dyn_name = { 
    "gene": "MM",
    "epi": "SIS", 
    "wc": "WC",
    "lv": "LV",
    "eco2": "MP2",  
    "popu": "P"
}
  
def load_data(args, net, logger=None, eval_ss=True):
    print('\n\n\nargs.n', args.n)
    if args.topology == "er":
        suffix = f'_k{args.k}.txt'
    elif args.topology == "sf":
        suffix = f'_m{args.m}.txt'
    elif args.topology == "regular":
        suffix = f'_k{args.k}.txt'
    elif args.topology == "sw":
        suffix = f'_k{args.k}.txt'
    elif args.topology == "com":
        suffix = f'_p{args.p_out}.txt' 
    else:
        suffix = f'_{args.data}.txt'
 
     
    A = get_A(args, source='random')
    net.setTopology(A)
    x0 = get_x0(args, args.dynamics, net.N, source='random') 
    #assert(args.n == net.N)
    print(args.seed)
    if args.seed == 1:
        net.gt = get_param(args.dynamics, net.N, net.A, args, source='random') 
        np.savetxt(f'./{args.data_folder}/{args.dynamics}/{dyn_name[args.dynamics]}_param_seed1_n{args.n}.txt', net.gt) 
         
    else:
        net.gt = np.loadtxt(f'./{args.data_folder}/{args.dynamics}/{dyn_name[args.dynamics]}_param_seed1_n{args.n}.txt' )
   
    if eval_ss:
        if args.dynamics not in ['lv', 'ko']:
            ss, equilibrium_t = net.solve_ss(net.dxdt, x0, net.gt)
        else:
            ss = np.ones(net.N)
            equilibrium_t = 0
    else:
        ss = np.ones(net.N)
        equilibrium_t = 0
    if logger:
        logger.info(f"Equilibrium time {equilibrium_t}")
    else:
        print(f"Equilibrium time {equilibrium_t}")    
    #np.savetxt(f'./{args.data_folder}/{args.dynamics}/{dyn_name[args.dynamics]}_x0_seed{args.seed}_n{args.n}_{args.topology}' + suffix, x0) 
    #np.savetxt(f'./{args.data_folder}/{args.dynamics}/{dyn_name[args.dynamics]}_A_seed{args.seed}_n{args.n}_{args.topology}' + suffix, net.A)
    #np.savetxt(f'./{args.data_folder}/{args.dynamics}/{dyn_name[args.dynamics]}_ss_seed{args.seed}_n{args.n}_{args.topology}' + suffix, ss) 
     
    if logger:
        logger.info(f"Network size: {net.N}\nAverage Weight: {net.A.mean()}\nAverage Degree: {net.binA.sum(1).mean()}")
    else:
        print(f"Network size: {net.N}\nAverage Weight: {net.A.mean()}\nAverage Degree: {net.binA.sum(1).mean()}")  
     
    return x0, ss

def create_net(args):
    if args.topology != 'brn':
        n = args.n
        avg_d = args.k 
        if args.dynamics == 'gene':
            net = Gene(n, avg_d/(n-1), args.data, args.topology, m=args.m, seed=args.seed)
        elif args.dynamics == "epi":
            net = Epidemic(n, avg_d/(n-1), args.data, args.topology, m=args.m, seed=args.seed)
        elif args.dynamics == "lv":
            net = LV(n, avg_d/(n-1), args.data, args.topology, m=args.m, seed=args.seed) 
        elif args.dynamics == "wc":
            net = Neural(n, avg_d/(n-1), args.data, args.topology, m=args.m, seed=args.seed)
        elif args.dynamics == "eco2":
            net = Eco2(n, avg_d/(n-1), args.data, args.topology, m=args.m, seed=args.seed) 
        else:
            net = Popu(n, avg_d/(n-1), args.data, args.topology, m=args.m, seed=args.seed)
        # if args.topology == 'real':
        #     topo, A_exist = load_real_net(f'./data/{args.data}.edges.csv', args.directed)
        #     net.setTopology(topo) 
    return net
 
def get_x0(args, dyn, N, source='pnas'):
    if dyn in ["eco2", "eco"]:
        if args.x0_out == 1:
            return np.random.normal(6, 1, N)
        else:
            return 2 * np.random.rand(N) 
        # if args.seed % 2 == 0:
        #     return 2 * np.random.rand(N)
        # else:
        #     return 5 * np.random.rand(N)
    if dyn in ["popu", "gene2"]:
        if args.x0_out == 1:
            return np.random.normal(6,1, N)
        else:
            if args.topology == 'er':
                return 2 * np.random.rand(N) 
            else:
                return 2 * np.random.rand(N) 
        #return 2 * np.random.rand(N)
    if dyn in ["gene" ]:
        if args.x0_out == 1:
            return np.random.normal(6,1, N)
        else:
            return 2 * np.random.rand(N) 
        #return 2 * np.random.rand(N)
        # if args.seed % 2 == 0:
        #     return 2 * np.random.rand(N)
        # else:
        #     return 5 * np.random.rand(N)
    if dyn == "epi":
        if args.x0_out == 1:
            return np.random.normal(0.5,0.1, N)
        else:
            #x0 = np.random.normal(0.8,0.1, N)
            x0 = np.full(N, 0.1)
            x0[random.sample(range(N), int(N*0.5))] = 0.8
            return x0
        #return 0.1 * np.random.rand(N)
    if dyn == "lv":
        if args.x0_out == 1:
            return np.random.normal(6,1, N)
        else:
            return 20 * np.random.rand(N) 
        #return np.random.rand(N)
    if dyn == "ko":
        if args.x0_out == 1:
            return np.random.normal(np.pi,0.5, N)
        x_init_max = np.pi / 4 
        x_init_min = - np.pi / 4   
        return ( x_init_max - x_init_min )*np.random.rand( N ) - ( x_init_max - x_init_min )/2
    if dyn in ["wc","neural"]:
        if args.x0_out == 1:
            return np.random.normal(6,1, N)
        return 10 * np.random.rand(N)
    
def get_A(args, source='pnas'):
    dyn = args.dynamics
    if args.topology == "real":
        A, _ = load_real_net(args.data)
        return A
    if args.topology == "er":
        if args.k_out != 0:
            k = args.k * 2
        else:
            k = args.k
        G = nx.fast_gnp_random_graph(args.n, k/(args.n-1), seed=args.seed, directed=True)
 
    elif args.topology == 'sf': 
        #G = nx.scale_free_graph(args.n, alpha=args.alpha, beta=(1 - args.alpha)/2, gamma = (1 - args.alpha)/2)
        if args.k_out != 0:
            m = args.m * 2
        else:
            m = args.m
        G = nx.barabasi_albert_graph(args.n, m, seed=args.seed)
    elif args.topology == "regular":
        G = nx.random_regular_graph(args.k, args.n, seed=args.seed) 
        A = nx.to_numpy_array(G,nodelist=(range(len(G))))  
    elif args.topology == "sw":
        G = nx.newman_watts_strogatz_graph(args.n, args.k, 0.5, seed=args.seed) 
    elif args.topology == "com":
        n1 = int(args.n/3)
        n2 = int(args.n/3)
        n3 = int(args.n/4)
        n4 = args.n - n1 - n2 -n3
        if args.k_out != 0:
            p_out = args.p_out * 2
        else:
            p_out = args.p_out
        G = nx.random_partition_graph([n1, n2, n3, n4], .25, p_out, seed=args.seed)
         
    A = nx.to_numpy_array(G, nodelist=range(len(G)))
    if args.s_out != 0:
        link_weights = 2 + np.random.rand(A.shape[0], A.shape[1])
    else:
        link_weights = 0.5 + np.random.rand(A.shape[0], A.shape[1])
    A = A * link_weights  
    return A

def get_param(dyn, N, B, args, source='pnas'):
    if source == 'pnas':
        if dyn == "eco":
            alpha = np.loadtxt('../../Predicting-network-dynamics-without-the-graph-main/results/Eco_alpha.txt')
            theta = np.loadtxt('../../Predicting-network-dynamics-without-the-graph-main/results/Eco_theta.txt')
            print(alpha[:5], theta[:5])
            param = np.concatenate((alpha, theta)).reshape(2,-1)
            print('param', param[:, :5])
        if dyn == "gene":
            param = np.array([1,1,2])
        if dyn == "epi":
            param = np.loadtxt('../../Predicting-network-dynamics-without-the-graph-main/results/SIS_delta.txt')
        if dyn == "lv":
            alpha = np.loadtxt('../../Predicting-network-dynamics-without-the-graph-main/results/LV_alpha.txt')
            theta = np.loadtxt('../../Predicting-network-dynamics-without-the-graph-main/results/LV_theta.txt')
            param = np.concatenate((alpha, theta)).reshape(2,-1)
        if dyn == "ko":
            param = np.loadtxt('../../Predicting-network-dynamics-without-the-graph-main/results/kuramoto_omega.txt')
        if dyn == "wc":
            param = np.array([1,1])
    else:
        if dyn == "eco": 
            alpha = 1 + 0.5*( 2*np.random.rand( N ) - 1 )
            theta = 1 + 0.5*( 2*np.random.rand( N ) - 1 ) 
            #alpha = np.random.rand( N ) * 0.01
            #theta = np.random.rand( N ) * 0.1
            param = np.concatenate((alpha, theta)).reshape(2,-1)
            #param = np.ones((2,N))*0.1
            if args.fix_dyn == 1:
                param = param[:,0]
            #param[0] = 0.00005
            #param[1] = 0.00005 
        elif dyn == "gene":
            param = np.array([1,1,2])
            #param = np.array([.1,1,2])
        elif dyn == "epi":
            delta_init = 1 + 0.5*( 2*np.random.rand( N ) - 1)
            R_0_init = np.max(np.abs(np.linalg.eigvals( np.diag( 1./np.sqrt( delta_init ))@B@np.diag( 1./np.sqrt( delta_init )))))
            print("R_0_init", R_0_init)
            print(np.mean(B), np.mean(delta_init))
            multiplicity_delta = R_0_init / 1.5
            param = multiplicity_delta * delta_init
            #param = multiplicity_delta * delta_init * 0.1 #then it holds eigs(W, 1) ==parameters.SIS.R_0_SIS, where W = diag(1./results.SIS.delta  )*results.B
            #param = np.ones(N) * 5
            if args.fix_dyn == 1:
                param = np.ones(N) * param[0]
        elif dyn == "lv":
            alpha = 1 + 0.5*( 2*np.random.rand(N) - 1)
            theta = 1 + 0.5*( 2*np.random.rand(N) - 1)
            #alpha = np.random.rand( N ) * 0.1
            #theta = np.random.rand( N ) * 0.1

            param = np.concatenate((alpha, theta)).reshape(2,-1)
            if args.fix_dyn == 1:
                param = param[:,0]
            #param = np.ones((2,N))*0.1 
            #param[0] = 0.00001
            #param[1] = 0.00001
        elif dyn == "ko": 
            param = 0.1*np.pi * np.random.randn( N )
            #param = np.random.normal(np.pi,0.1,N)
            if args.fix_dyn == 1:
                param = param[0] * np.ones(N)
        elif dyn == "wc":
            param = np.array([1,1]) 
            #param = np.array([2,2])
        elif dyn == "gene2":
            param = np.array([1,0.4,0.2])
        elif dyn == "neural":
            param = np.array([1,1])
        elif dyn == "popu":
            param = np.array([0.5,0.2])
        else:
            param = np.zeros(1)
    return param
 
 
   
 
def load_real_net(data ):
    '''
    Read adjacency matrix from data files.

    Assumption:
    Each line in data file takes the form 'i,j,{'weight': x}', 
    where (i,j) is an edge and x is the weight of the edge
    '''   
    print(data)
    net = data   
    if net in ['oregon.edges.csv']:
        dlm = '\t'
    else:
        dlm = ','
    edges = np.genfromtxt(data, delimiter=dlm)
     
    if edges.shape[1] == 3:
        edges = edges[:, :2] 
    nodes = np.unique(edges)
    num_node = len(np.unique(edges))
    idx_map = {nodes[i]:i for i in range(num_node)}
    print(np.max(edges))
    print(num_node)
    A = np.zeros((num_node,num_node))
    for i in range(len(edges)): 
        A[idx_map[edges[i][0]],idx_map[edges[i][1]]] = 1
        A[idx_map[edges[i][1]],idx_map[edges[i][0]]] = 1
    print('finish creating graph')
    print('[number of nodes]', num_node)
    print('[number of edges]', len(edges))
    return A, True 
 