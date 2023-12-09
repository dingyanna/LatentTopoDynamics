import torch.nn as nn
import lib.utils as utils
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class TemporalEncoding(nn.Module):

    def __init__(self, d_hid):
        super(TemporalEncoding, self).__init__()
        self.d_hid = d_hid
        self.div_term = torch.FloatTensor([1 / np.power(10000, 2 * (hid_j // 2) / self.d_hid) for hid_j in range(self.d_hid)]) #[20]
        self.div_term = torch.reshape(self.div_term,(1,-1))
        self.div_term = nn.Parameter(self.div_term,requires_grad=False)

    def forward(self, t):
        '''

        :param t: [n,1]
        :return:
        '''
        t = t.view(-1,1)
        t = t   # scale from [0,1] --> [0,200], align with 'attention is all you need'
        position_term = torch.matmul(t,self.div_term)
        position_term[:,0::2] = torch.sin(position_term[:,0::2])
        position_term[:,1::2] = torch.cos(position_term[:,1::2])

        return position_term


class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super(FeedForward, self).__init__()  
        self.ff = nn.Sequential(
            nn.Linear( dim_in, dim_hid),
            nn.ReLU(),
            nn.Linear( dim_hid, dim_out)
        ) 
        utils.init_network_weights(self.ff) 
    def forward(self, x):
        return self.ff(x)
    
class Encoder_MLP(nn.Module):
    def __init__(self, input_dim, latent_dim, args):
        super(Encoder_MLP, self).__init__()  
        self.f_node =  FeedForward(input_dim, latent_dim*2, latent_dim)
        self.f_edge =  FeedForward(2*latent_dim, latent_dim, 1) 
        utils.init_network_weights(self.f_node) 
        utils.init_network_weights(self.f_edge) 
        # encoder = nn.Sequential(
        #     nn.Linear(input_dim, args.rec_dims),
        #     nn.ReLU(),
        #     nn.Linear(args.rec_dims,latent_dim),
        # )
        # utils.init_network_weights(encoder) 
        #self.encoder = encoder

    def forward(self, x):
        '''
        x: batch_size x num_nodes x d
        output: batch_size x num_nodes x d', batch_size x num_nodes x num_nodes
        '''
        N,m,d = x.shape
        x = self.f_node(x) 
        sender = x.view(N, -1, 1, x.shape[-1]) # N x m x 1 x d
        sender = sender.repeat(1, 1, m, 1) # N x m x m x d 
        receiver = x.view(N, -1, 1, x.shape[-1]) # N x m x 1 x d
        receiver = receiver.repeat(1, 1, m, 1) # N x m x m x d
        receiver = receiver.transpose(1,2)  
        edge = torch.cat((sender, receiver.transpose(1, 2)), dim=-1) # N x m x m x 2d
        edge = self.f_edge(edge) # N x m x m x 1
        edge /= m 
        #np.savetxt('edge.txt', edge[0].reshape(m,m).detach().cpu().numpy())
        #np.savetxt('edge.txt', torch.softmax(edge[0].reshape(m,m),dim=1).detach().cpu().numpy())
        return x, edge.squeeze(-1)
        #return self.encoder(x), None

class Encoder_NRI(nn.Module):
    def __init__(self, input_dim, latent_dim, args):
        
        super(Encoder_NRI, self).__init__()  
        self.f_node =  FeedForward(input_dim, latent_dim*2, latent_dim)
        self.f_edge =  FeedForward(2*latent_dim, latent_dim, latent_dim) 
        self.f_edge2node = FeedForward(latent_dim, latent_dim*2, latent_dim)
        self.f_node2edge = FeedForward(2*latent_dim, latent_dim, 1) 
         

    def forward(self, x):
        '''
        x: batch_size x num_nodes x d
        output: batch_size x num_nodes x d', batch_size x num_nodes x num_nodes
        '''
         
        N,m,d = x.shape
        x = self.f_node(x)  
        edge = self.f_edge(self.construct_pair(x)) # N x m x m x d
          
        off_diag = torch.ones([m, m]) - torch.eye(m) # m x m
        edge2node = edge * off_diag.view(1,m,m,1)
        edge2node = edge2node.sum(1) # N x m x d
        x = self.f_edge2node(edge2node) # N x m x d
        
        edge = self.f_node2edge(self.construct_pair(x)) # N x m x m x 1
        edge /= m
        return x, edge.squeeze(-1)

    def construct_pair(self,x):
        N, m, d = x.shape
        sender = x.view(N, -1, 1, d) # N x m x 1 x d
        sender = sender.repeat(1, 1, m, 1) # N x m x m x d 
        receiver = x.view(N, -1, 1, d) # N x m x 1 x d
        receiver = receiver.repeat(1, 1, m, 1) # N x m x m x d
        receiver = receiver.transpose(1,2)   
        return torch.cat((sender, receiver.transpose(1, 2)), dim=-1)

class Encoder_Transformer(nn.Module):
    def __init__(self, input_dim, latent_dim, args):
        
        super(Encoder_Transformer, self).__init__()  
        self.f_node =  FeedForward(input_dim, latent_dim*2, latent_dim)
        self.w_k = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for i in range(args.num_head)])
        self.w_q = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for i in range(args.num_head)])
        self.w_v = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for i in range(args.num_head)])
        self.f_v = FeedForward(latent_dim*args.num_head,latent_dim,latent_dim)
        self.f_node2edge = FeedForward(2*latent_dim, latent_dim, 1) 
          
        utils.init_network_weights(self.w_k )
        utils.init_network_weights(self.w_q )
        utils.init_network_weights(self.w_v )  
        self.relu = nn.LeakyReLU(0.2) 
        self.softmax = nn.Softmax(dim=-1)
        self.temporal_net = TemporalEncoding(1)
        self.t = torch.linspace(0,args.t_obs,args.condition_length)

    def forward(self, x ):
        '''
        x: batch_size x num_nodes x T 
        output: batch_size x num_nodes x d', batch_size x num_nodes x num_nodes
        ''' 
        N, m, T = x.shape 
        te = self.temporal_net(self.t.to(x.device))  
        x = self.f_node(x+te.reshape(1,1,T)) # N x m x d 
        heads = []  
        for i in range(len(self.w_k)):
            sender = self.w_k[i](x ) # N x m x d
            sender = sender.view(N, -1, 1, sender.shape[-1]) # N x m x 1 x d
            sender = sender.repeat(1, 1, m, 1) # N x m x m x d

            receiver = self.w_q[i](x ) # N x m x d
            receiver = receiver.view(N, -1, 1, receiver.shape[-1]) # N x m x 1 x d
            receiver = receiver.repeat(1, 1, m, 1) # N x m x m x d
            receiver = receiver.transpose(1,2) 
  
            alpha = (sender * receiver).sum(-1) 
            alpha = self.relu(alpha) # N x m*T x m*T
            alpha = self.softmax(alpha) # N x m*T x m*T   

            att = self.relu(alpha @ self.w_v[i](x )) # N x m*T x m*T  @ N x m*T x d
 
            heads.append(att)

        att = torch.cat(heads, dim=-1)
        x = self.f_v(att) + x
        edge = self.f_node2edge(utils.construct_pair(x)) # N x m x m x 1
        edge /= m
        return x, edge.squeeze(-1)
 
class Encoder_GRU1(nn.Module):
    def __init__(self, input_dim, latent_dim, args):
        super(Encoder_GRU1, self).__init__() 
        self.rec_layers = args.rec_layers
        self.rec_dims = args.rec_dims

        self.gru = nn.GRU(input_dim, args.rec_dims, args.rec_layers, batch_first=True)
        self.fc = nn.Linear(args.rec_dims, latent_dim)
         
        utils.init_network_weights(self.gru)  
        utils.init_network_weights(self.fc)  


    def forward(self, x):
        '''
        x: batch_size*num_nodes x T x d
        ''' 
        b,n,T,d = x.shape 
        x = x.view(-1, T, d)
        h0 =  torch.zeros(self.rec_layers, x.shape[0], self.rec_dims) 
        out, h = self.gru(x, h0.to(x.device))
        out = self.fc(out[:, -1, :])  # Using the final time step's output for prediction 
        out = out.view(b, n, -1)
        return out

class Encoder_GRU2(nn.Module):
    def __init__(self, input_dim, latent_dim, args):
        super(Encoder_GRU2, self).__init__()  
        self.rec_layers = args.rec_layers
        self.rec_dims = args.rec_dims

        self.gru = nn.GRU(input_dim, args.rec_dims, args.rec_layers, batch_first=True)
        self.w = nn.Linear(args.rec_dims, args.rec_dims)
        self.fc = nn.Linear(args.rec_dims, latent_dim)
         
        utils.init_network_weights(self.gru)  
        utils.init_network_weights(self.w)  
        utils.init_network_weights(self.fc)  


    def forward(self, x):
        '''
        x:  (batch_size x num_nodes) x T x d
        '''
        b,n,T,d = x.shape 
        x = x.view(-1, T, d)

        h0 =  torch.zeros(self.rec_layers, x.shape[0], self.rec_dims)
        out, h = self.gru(x, h0.to(x.device)) # out: b*n, T, d   
        out = out.reshape(b*n*T, out.shape[-1]) 
        
        batch_new = self.rewrite_batch(b,n,T,x.device) 
        alpha = F.gelu(self.w(global_mean_pool(out, batch_new))) # b*n x d 

        alpha = alpha.unsqueeze(1).repeat(1,T,1)
        alpha = alpha.view(b*n*T,-1) 
        
        alpha = torch.sigmoid(torch.bmm(out.unsqueeze(1), alpha.unsqueeze(2))).reshape(-1,1) # b*n*T x 1 x d @ b*n*T x d x 1 => b*n*T x 1 
        alpha = alpha * out # b*n*T x 1 * b*n*T x d => b*n*T x d  
        out = global_mean_pool(alpha, batch_new) # b*n x d  
        out = self.fc(out) # b*n x d_out 
        out = out.reshape(b, n, -1) 
        return out

    def rewrite_batch(self, b,n,T,device):  
        batch_new = torch.zeros(b*n*T, dtype=torch.int64).to(device)
        current_index = 0
        for i in range(b*n):
            batch_new[current_index:current_index + T] = i 
            current_index += T

        return batch_new

    def attention_expand(self,attention_ball, b, n, T):
        '''

        :param attention_ball: [num_ball, d]
        :param batch: [num_nodes,d]
        :param batch_new: [num_ball,d]
        :return:
        '''
        node_size = b*n*T
        dim = attention_ball.size()[1]
        new_attention = torch.ones(node_size, dim).to(attention_ball.device)
        # if attention_ball.device != torch.device("cpu"):
        #     new_attention = new_attention.cuda()
 
        current_index = 0
        for i in range(b*n):
            new_attention[current_index:current_index+T] = attention_ball[i] 
            current_index += T

        return new_attention


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim,decoder_network = None):
        super(Decoder, self).__init__()
        # decode data from latent space where we are solving an ODE back to the data space
        if decoder_network == None:
            decoder = nn.Sequential(
                nn.Linear(latent_dim, latent_dim//2),
                nn.ReLU(),
                nn.Linear(latent_dim//2,output_dim),
            )
            utils.init_network_weights(decoder)
        else:
            decoder = decoder_network

        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)

 