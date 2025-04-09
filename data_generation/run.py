 
from utils import *    
from scipy.integrate import odeint 
import numpy as np
 
import random 
from utils import * 
import os  
import argparse
parser = argparse.ArgumentParser('Dynamics Inference') 
parser.add_argument('--dynamics', type=str, default='eco', help="eco, epi, gene")
parser.add_argument('--topology', type=str, default='er', help="er, sf, com")
parser.add_argument('--data', type=str, default='', help="net6, net8")
  
# Topology related
parser.add_argument('--n', type=int, default=200) # Total number of nodes
parser.add_argument('--k', type=int, default=12) # average degree (parameter for ER network)
parser.add_argument('--m', type=int, default=4) # parameter for scale free network
 

parser.add_argument('--seed', type=int, default=42) 
parser.add_argument('--eval_ss', type=bool, default=True)
parser.add_argument('--x0_out', type=int, default=0, help='indicate whether x0 is out-of-dist')
parser.add_argument('--s_out', type=int, default=0, help='indicate whether link weight is out-of-dist')
parser.add_argument('--k_out', type=int, default=0, help='indicate whether degree is out-of-dist')
parser.add_argument('--to_generate', type=str, default='density' )

parser.add_argument('--p_out', type=float, default=0.1 ) 
parser.add_argument('--fix_dyn', type=int, default=0, help='whether the dynamical parameter is identical for all nodes') 
 
     
def preprocess(args):
    random.seed(args.seed)
    np.random.seed(args.seed)  
    if args.dynamics != "eco2":
        args.T = 5
    else:
        args.T = 2      
    net = create_net(args)    
    return args, net 

def gt_traj(args):  
    sfx = args.seed
    folder = f"./data_{args.to_generate}/{args.dynamics}"
    args.data_folder = f"data_{args.to_generate}"
    
    if not os.path.exists(f"{folder}/{args.topology}"):
        os.makedirs(f"{folder}/{args.topology}")

    if args.to_generate not in ['main', 'sample']:
        print('preprocess')
        args, net = preprocess(args)  
        t = np.linspace(0,args.T,200)
        print('loading data')
        x0, ss = load_data(args, net, eval_ss=False)
        print('\n==>before evaluating states')   
        X = odeint(net.dxdt, x0, t, args=(net.gt,), printmessg=True) 
        
        if args.to_generate == 'density':
            sfx = int(args.k/10)
        elif args.to_generate == 'extrap_out':
            if args.topology == 'er' and args.k == 10:
                sfx = 1
            elif args.topology == 'er' and args.k == 50:
                sfx = 2 
            else:
                sfx = 3 
        elif args.to_generate in ['k_out', 'x0_out', 's_out']:
            sfx=f"{args.topology}_{args.seed}"
        print(sfx) 
        if args.topology == 'real':
            np.save(f"{folder}/{args.data}.npy", X.T)
        np.save(f"{folder}/states{sfx}.npy", X.T)
        print('saving to', folder)
        return  
     
    N_lis = [100] 
    if args.topology == 'er': 
        p_lis = np.linspace(0.1,0.5,5)  # ID
        if args.dynamics == 'epi':
            p_lis = p_lis[2:3]
        else:
            p_lis = p_lis[0:1]
        #p_lis = [0.2] # OOD
    elif args.topology == 'sf':
        p_lis = np.linspace(0.4,0.6,5)[0:1]
        p_lis = [5, 7, 9, 11, 13][0:1] # ID
        #p_lis = [7] # OOD 
    elif args.topology == 'com':
        p_lis = np.linspace(0.1,0.3,5)[0:1] # ID
        #p_lis = [0.15] # OOD 
    
    for i_n in range(len(N_lis)):
        n = N_lis[i_n]
        for i_p in range(len(p_lis)):
            p = p_lis[i_p]
            args.n = n 
            if args.topology == 'er': 
                args.k = int(n * p)
            elif args.topology == 'sf':
                args.alpha = p
            elif args.topology == 'sw':
                args.k = p
            elif args.topology == 'com':
                args.p_out = p
            args, net  = preprocess(args)  
            t = np.linspace(0,args.T,200)
            x0, ss = load_data(args, net , eval_ss=False)   
            X = odeint(net.dxdt, x0, t, args=(net.gt,))  
            if np.any(np.isnan(X)):
                print(len(np.where(np.isnan(X))[0]))
                exit()
            np.save(f"{folder}/{args.topology}/states_n{i_n}_p{i_p}_{args.seed}.npy", X.T)
            
            print('<x>', np.mean(X), 'saving to', folder, args.seed)
 
 

if __name__ == '__main__':
    args = parser.parse_args()
    topo = args.topology
    dyn = args.dynamics   
    gt_traj(args)
 