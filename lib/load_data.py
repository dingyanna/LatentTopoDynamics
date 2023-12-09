import numpy as np
import torch
from torch.utils.data import DataLoader as Loader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
 
import lib.utils as utils
from tqdm import tqdm
import os
from torchvision import datasets
import torchvision.transforms as transforms
from scipy.linalg import block_diag
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import random
 
states = [10,1,10,1,1000,1000000,100000]

 
class ParseData(object):

    def __init__(self,args, device="cpu"):
        self.args = args 
        self.random_seed = args.random_seed
        self.pred_length = args.pred_length
        self.batch_size = args.batch_size
        self.args.batch_size_encoder = args.batch_size
        self.device = device

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)   
        
      
    def load_motion_graphs(self, data_type): 
        ''' 
        time range mapping
        0 -> 0
        T2-T1 -> 1
        ''' 
        T1 = self.args.condition_length 
        
        if data_type == 'train':
            loc = np.load("motion/loc_train_motion.npy", allow_pickle=True)
            vel = np.load("motion/vel_train_motion.npy", allow_pickle=True)
            obs_time = np.load("motion/times_train_motion.npy", allow_pickle=True)
        else:
            loc = np.load("motion/loc_test_motion.npy", allow_pickle=True)
            vel = np.load("motion/vel_test_motion.npy", allow_pickle=True)
            obs_time = np.load("motion/times_test_motion.npy", allow_pickle=True)
         
        max_time_len = 0 

        for i in range(obs_time.shape[0]): # loop through all sequence
            max_obs = max([max(l) for l in obs_time[i]])
            if max_obs > max_time_len:
                max_time_len = max_obs + 1
        print('T', max_time_len)
        Loc = np.full((loc.shape[0], loc.shape[1], max_time_len, 3), np.nan)
        Vel = np.full((loc.shape[0], loc.shape[1], max_time_len, 3), np.nan)
        
        for i in range(obs_time.shape[0]): # loop through all sequence 
            for j in range(obs_time.shape[1]): # loop through all nodes  
                for t in range(len(obs_time[i,j])):
                    if t < T1:
                        cur_t = t 
                    else:
                        cur_t = obs_time[i,j][t]
                    Loc[i, j, cur_t] = loc[i,j][t]
                    Vel[i, j, cur_t] = vel[i,j][t]
        seqs = np.concatenate((Loc, Vel), axis=-1) 
        if data_type == 'test':
            print(obs_time[0])
        np.save(f'motion_{data_type}_{self.args.condition_length}.npy', seqs)
        self.times_extrap = torch.linspace(0, 1, max_time_len-T1) 
        features = torch.Tensor(seqs[:, :, :T1])  # N, m, T_cond
        decoder_feature = torch.Tensor(seqs[:, :, T1:]) # N, m, T_pred 
        N, m = features.shape[:2]
        if self.args.encoder == 'mlp':
            features = features.view(N, m, -1) 
        self.num_neurons = m
        self.num_edges = self.num_neurons * self.num_neurons

        Tp = decoder_feature.shape[2] 
        decoder_cat = np.zeros((decoder_feature.shape[0], decoder_feature.shape[1]+1, Tp, 6))
        decoder_cat[:,:-1,:] = decoder_feature 
        decoder_cat[:,-1,:, 0] = self.times_extrap 
        decoder_feature = decoder_cat  

        return features, decoder_feature   
 
    def load_large_graph(self, data_type): 
        ''' 
        time range mapping
        0 -> 0
        T2-T1 -> 1
        '''
        states = []  
        count = 0
        T1 = self.args.condition_length # 200 
        T2 = 200
        max_count = 1 
        for i in range(101):  
            if count == max_count:
                break 
            fn = f"./data_generation/data_{self.args.dataset}/{self.args.dynamics}/states{i}.npy"
            if not os.path.exists(fn):
                print(fn)
                continue 
            count += 1  
            state = np.load(fn)  # N x T
            states.append(state )  
        states = np.asarray(states) # num_mlp x num_nodes x T x 2
        print(states.shape) 
        states = states.reshape(states.shape[0], states.shape[1], states.shape[2], 1)
        self.num_neurons = states.shape[1]   
        self.num_edges = self.num_neurons * self.num_neurons

        print('\nbefore slicing: states.shape', states.shape, '\n')  
         
        cond_len = self.args.condition_length
        self.times_extrap = torch.linspace(0, 1, T2-T1) 
        seqs = states 
        features = torch.Tensor(seqs[:, :, :cond_len])  # N, m, T_cond
        decoder_feature = torch.Tensor(seqs[:, :, cond_len:]) # N, m, T_pred
        self.times_extrap = torch.linspace(0, 1, T2-T1) 
        self.batch_size = self.args.test_bsize 
        N, m  = features.shape[:2]
        Tp = decoder_feature.shape[2] 
        print('decoder', decoder_feature.shape)
        if self.args.encoder == 'mlp':
            features = features.view(N, m, -1) 
         
        decoder_cat = np.zeros((decoder_feature.shape[0], decoder_feature.shape[1]+1, Tp, 1))
        decoder_cat[:,:-1,:] = decoder_feature 
        decoder_cat[:,-1,:, 0] = self.times_extrap 
        decoder_feature = decoder_cat  
        
        self.times_observed = torch.linspace(0, self.args.condition_length/(T2-T1), self.args.condition_length) 
        return features, decoder_feature 

    def load_graphs(self, data_type): 
        ''' 
        time range mapping
        0 -> 0
        T2-T1 -> 1
        '''
        states = []  
        count = 0
        T1 = self.args.condition_length # 200 
        T2 = 200 
        if data_type == 'train':
            init = 1 # initial state seed index
            max_count = 100 #  
            if self.args.dataset == 'data_sample':
                max_count = 8
        elif data_type == 'val':
            init = 100  
            max_count = 20 # 100
            if self.args.dataset == 'data_sample':
                init = 8
                max_count = 2
        elif data_type == 'test':
            init = 120
            max_count = 20 # 100  
            if self.args.dataset == 'data_sample':
                init = 10
                max_count = 2
        fdr = self.args.dataset 
        if self.args.test_case != 'na':
            fdr = f"data_{self.args.test_case}"  
            if self.args.test_case == "noise_obs":
                fdr = "data" # for this type of experiments, we pull out the test data 
            else:
                init = 0
        ### large dataset #####
        if self.args.topology != 'mixed':
            topo = [self.args.topology]
        else:
            topo = ['er', 'sf', 'com']
        N_lis = [0]
        p_lis = [0] 
        #######################
        for network in topo:
            for n in N_lis:
                for p in p_lis:  
                    for i in range(init, init+max_count):    
                        fn = f'./data_generation/{fdr}/{self.args.dynamics}/{network}/states_n{n}_p{p}_{i}.npy'                
                        #print(network, n, p, i) 
                        if not os.path.exists(fn):
                            print("FileNotExist", fn)
                            continue 
                        count += 1  
                        state = np.load(fn) 
                        if np.any(np.isnan(state)): 
                            continue 

                        state_padded = np.full((state.shape[0], T2), np.nan)   
                        state_padded[:,:state.shape[1]] = state.copy()
                        states.append(state_padded)
                
        states = np.asarray(states) # num_mlp x num_nodes x T 
        self.num_neurons = states.shape[1]   
        self.num_edges = self.num_neurons * self.num_neurons
 
        if data_type == 'train': 
            cond_len = self.args.condition_length 
            self.times_extrap = torch.linspace(0, 1, T2-T1) 
            features = torch.Tensor(states[:, :, :cond_len])  # N, m, T_cond
            decoder_feature = torch.Tensor(states[:, :, cond_len:]) # N, m, T_pred
            self.times_extrap = torch.linspace(0, 1, T2-T1)  
        else: # val, test set 
            cond_len = self.args.condition_length 
            self.times_extrap = torch.linspace(0, 1, T2-T1) 
            features = torch.Tensor(states[:, :, :cond_len])  # N, m, T_cond
            decoder_feature = torch.Tensor(states[:, :, cond_len:]) # N, m, T_pred
            self.times_extrap = torch.linspace(0, 1, T2-T1) 
            self.batch_size = self.args.test_bsize  
             
            if self.args.test_case == 'noise_obs':
                all_feat = []
                decoder_feat = []
                n_level = torch.linspace(0.01,0.2,10)
                for i in n_level: 
                    noise = torch.randn(features.shape) * i
                    all_feat.append(features * (1+noise))
                    decoder_feat.append(decoder_feature)
                features = torch.cat(all_feat,dim=0)
                decoder_feature = torch.cat(decoder_feat, dim=0) 

        N, m, T = features.shape
        Tp = decoder_feature.shape[-1] 
        print('decoder', decoder_feature.shape, '\n')
        if self.args.encoder in ['gru', 'gnn']:
            features = features.view(N, m, T, 1) 
         
        decoder_cat = np.zeros((decoder_feature.shape[0], decoder_feature.shape[1]+1,Tp,1))
        decoder_cat[:,:-1,:] = decoder_feature.reshape(decoder_feature.shape[0], decoder_feature.shape[1],Tp,1) 
        decoder_cat[:,-1,:,0] = self.times_extrap 
        decoder_feature = decoder_cat  
        
        self.times_observed = torch.linspace(0, self.args.condition_length/(T2-T1), self.args.condition_length) 
         
        return features, decoder_feature 
      
    def load_data(self,data_type = "train"): 
        if self.args.dataset in ['data_main', 'data_sample']:
            features, decoder_feature = self.load_graphs(data_type)  
        elif self.args.dataset in ["data_large", "data_large_1"]:
            features, decoder_feature = self.load_large_graph( data_type)    
        elif self.args.dataset == "motion":
            features, decoder_feature = self.load_motion_graphs(data_type) 
        self.num_features = features.shape[-1]
        
        print('data_type', data_type)
        print('num neuron', self.num_neurons)
        print('encoder features.shape', features.shape)
        print('decoder feature.shape', decoder_feature.shape) 
        
        if self.args.encoder != 'gnn':
            encoder_data_loader = Loader(features, batch_size=self.batch_size ) 
        else:
            num_samples = features.shape[0]
            data_list=[]
            for i in tqdm(range(num_samples)):
                data_per_graph = self.transfer_one_graph(features[i], np.ones((features.shape[2],features.shape[1], features.shape[1])))
                data_list.append(data_per_graph) 
            encoder_data_loader = DataLoader(data_list, batch_size=self.batch_size,shuffle=False,num_workers=3)
 
        decoder_data_loader = Loader(decoder_feature, batch_size=self.batch_size , shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(
                                         batch), num_workers=3)  # num_graph*num_ball [tt,vals,masks]
         
        num_batch = len(decoder_data_loader) 
         
        # Inf-Generator
        encoder_data_loader = utils.inf_generator(encoder_data_loader) 
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        return encoder_data_loader, decoder_data_loader, num_batch, self.num_neurons 
         
    def variable_time_collate_fn_activity(self,batch): 
        combined_vals = np.concatenate([np.expand_dims(ex[:-1],axis=0) for ex in batch],axis=0)
        #combined_vals_true = np.concatenate([np.expand_dims(ex[1],axis=0) for ex in batch], axis = 0)

        combined_vals = torch.FloatTensor(combined_vals) #[M,T2,D]
        #combined_vals_true = torch.FloatTensor(combined_vals_true)  # [M,T2,D]
        if len(batch[0].shape) > 2: 
            combined_tt = torch.FloatTensor(batch[0][-1,:,0])
        else:
            combined_tt = torch.FloatTensor(batch[0][-1,:])
        #print('batch', batch.shape) 
        data_dict = {
            "data": combined_vals,
            "time_steps": combined_tt 
            } 
        return data_dict
    
    def transfer_one_graph(self, feature, edge):
        '''f 
        :param feature: [N,T1,D]  D = number of features
        :param edge: [T, M]
        :param time: [T1]
        :return:
            1. x : [N*T1,D]: feature for each node.
            2. edge_index [2,num_edge]: edges including cross-time
            3. edge_state [num_edge]: edge states
            4. y: [N], value= num_steps: number of timestamps for each state node.
            5. x_pos 【N*T1】: timestamp for each node
            6. edge_time [num_edge]: edge relative time.
        '''
        #print('transfering one graph')

        ########## Getting and setting hyperparameters:
        time = self.times_observed
        num_states = feature.shape[0] # number of nodes
        T1 = edge.shape[0]
        each_gap = 1/edge.shape[0]
        time = np.reshape(time,(-1,1))
        
        #print('computing x_pos')
        ########## Compute Node related data:  x,y,x_pos
        # [Num_states],value is the number of timestamp for each state in the encoder, == args.condition_length
        y = self.args.condition_length*np.ones(num_states)
        # [Num_states*T1,D]
        x = np.reshape(feature,(-1,feature.shape[2]))
        # [Num_states*T1,1] node timestamp
        x_pos = np.concatenate([time for i in range(num_states)],axis=0)
        assert len(x_pos) == feature.shape[0]*feature.shape[1] # number of nodes * time length

        #print('edge_time_matrix')
        ########## Compute edge related data
        edge_time_matrix = np.concatenate([np.asarray(x_pos).reshape(-1, 1) for _ in range(len(x_pos))],
                                          axis=1) - np.concatenate(
            [np.asarray(x_pos).reshape(1, -1) for _ in range(len(x_pos))], axis=0)  # [N*T1,N*T1], SAME TIME = 0

        edge_exist_matrix = np.ones((len(x_pos), len(x_pos)))  # [N*T1,N*T1] NO-EDGE = 0, depends on both edge state and time matrix

        #print('compute edge_state_matrix')
        # Step1: Construct edge_state_matrix [N*T1,N*T1]
        edge_repeat = np.repeat(edge, self.args.condition_length, axis=2)  # [T1,N,NT1]
        edge_repeat = np.transpose(edge_repeat, (1, 0, 2))  # [N,T1,NT1]
        edge_state_matrix = np.reshape(edge_repeat, (-1, edge_repeat.shape[2]))  # [N*T1,N*T1]
        #print('\nedge_state_matrix', edge_state_matrix.shape, '\n')
        #print('compute edge_statematrix')
        # mask out cross_time edges of different state nodes.
        a = np.identity(T1)  # [T,T]
        b = np.concatenate([a for i in range(num_states)], axis=0)  # [N*T,T]
        c = np.concatenate([b for i in range(num_states)], axis=1)  # [N*T,N*T]

        a = np.ones((T1, T1))
        d = block_diag(*([a] * num_states))
        edge_state_mask = (1 - d) * c + d
        edge_state_matrix = edge_state_matrix * edge_state_mask  # [N*T1,N*T1]

        max_gap = each_gap

        #print('computing edge exist matrix')
        # Step2: Construct edge_exist_matrix [N*T1,N*T1]: depending on both time and state.
        edge_exist_matrix = np.where(
            (edge_time_matrix <= 0) & (abs(edge_time_matrix) <= max_gap) & (edge_state_matrix != 0),
            edge_exist_matrix, 0)


        edge_state_matrix = edge_state_matrix * edge_exist_matrix
        edge_index, edge_state_attr = utils.convert_sparse(edge_state_matrix)
        assert np.sum(edge_state_matrix!=0)!=0  #at least one edge state (one edge) exists.

        edge_time_matrix = (edge_time_matrix + 3) * edge_exist_matrix # padding 2 to avoid equal time been seen as not exists.
        _, edge_time_attr = utils.convert_sparse(edge_time_matrix)
        edge_time_attr -= 3

        # converting to tensor
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_state_attr = torch.FloatTensor(edge_state_attr)
        edge_time_attr = torch.FloatTensor(edge_time_attr)
        y = torch.LongTensor(y)
        x_pos = torch.FloatTensor(x_pos)
 
        graph_data = Data(x=x, edge_index=edge_index, edge_state=edge_state_attr, y=y, pos=x_pos, edge_time = edge_time_attr, topo=feature.shape[0])
        edge_num = edge_index.shape[1]

        # print('x.shape', x.shape)
        # print('edge_index.shape', edge_index.shape)
        # print('edge_state_attr.shape', edge_state_attr.shape)
        # print('edge_time_attr.shape', edge_time_attr.shape)
        # print('y.shape', y.shape)
        # print('x_pos.shape', x_pos.shape)

        #return graph_data
        return graph_data 