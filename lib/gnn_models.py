import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool  
import math
import lib.utils as utils 
#from torch_scatter import scatter_add  
from lib.encoder_decoder import FeedForward
from torch_geometric.utils import softmax  

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
        t = t *200  # scale from [0,1] --> [0,200], align with 'attention is all you need'
        position_term = torch.matmul(t,self.div_term)
        position_term[:,0::2] = torch.sin(position_term[:,0::2])
        position_term[:,1::2] = torch.cos(position_term[:,1::2])

        return position_term

 
class Node_GCN(nn.Module):
    """Node ODE function."""

    def __init__(self, dim_in , dim_hid, num_node, num_head, dropout=0., gamma=0):
        super(Node_GCN, self).__init__()
        print('creating gcn')
        #self.gat = GAT(dim_in, dim_hid, num_head)
        #self.decoder = FeedForward(dim_hid,dim_hid,dim_hid)  
        #utils.init_network_weights(self.decoder)
        self.f = FeedForward(dim_in, dim_hid//2, dim_hid)
        self.g = FeedForward(dim_in*2, dim_hid, dim_hid) 
         
    def forward(self, x, edge):  
        '''
        x: num_batch x num_node x d
        edge: num_batch x num_node x num_node
        '''
        N,m,d = x.shape
        self_dyn = self.f(x)  # N x m x d'
        sender = x.view(N, -1, 1, x.shape[-1]) # N x m x 1 x d
        sender = sender.repeat(1, 1, m, 1) # N x m x m x d 
        receiver = x.view(N, -1, 1, x.shape[-1]) # N x m x 1 x d
        receiver = receiver.repeat(1, 1, m, 1) # N x m x m x d
        receiver = receiver.transpose(1,2)  
        interaction = torch.cat((sender, receiver.transpose(1, 2)), dim=-1) # N x m x m x 2d
        interaction = self.g(interaction) # N x m x m x d'
        interaction = edge.unsqueeze(-1) * interaction  # N x m x m x d'
        interaction = interaction.sum(1) # N x m x d'  
        return self_dyn + interaction

class Node_GAT(nn.Module):
    """Node ODE function.""" 
    def __init__(self, dim_in , dim_hid, num_node, num_head, dropout=0., gamma=0):
        super(Node_GAT, self).__init__() 
        self.gat = GAT(dim_in, dim_hid, num_head)
        #self.decoder = FeedForward(dim_hid,dim_hid,dim_hid)   
        self.decoder = nn.Linear(dim_hid * num_head, dim_in)
        utils.init_network_weights(self.decoder)
    def forward(self, x, edge):  
        '''
        x: num_batch x num_node x d
        edge: num_batch x num_node x num_node
        '''
        x = self.gat(x)
        x = self.decoder(x)
        return x
 
 
class GAT(nn.Module):
    def __init__(self, dim_in, dim_out, num_head):
        super(GAT, self).__init__()
        print('\nhidden dimension', dim_out)
        #self.att_w = nn.ModuleList([nn.Parameter(torch.rand(dim_out*2)) for i in range(num_head)])
        self.w_k = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(num_head)])
        self.w_q = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(num_head)])
        self.w_v = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(num_head)])
        print('\n\n len(self.w_k)', len(self.w_k), '\n\n')
        #self.self_dynamics = FeedForward(dim_in, dim_out, dim_out)
        #std = 1
        utils.init_network_weights(self.w_k )
        utils.init_network_weights(self.w_q )
        utils.init_network_weights(self.w_v )
        #utils.init_network_weights(self.self_dynamics)

        #self.B = nn.Parameter(torch.rand(num_node,1))
        #nn.init.normal_(self.B , mean=0, std=std)
        #print(init_w.shape)
        #self.B = nn.Parameter(init_w)
        #self.B = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(num_head)])
        #utils.init_network_weights(self.B, std=0.1)
 
        self.relu = nn.LeakyReLU(0.2) 
        self.softmax = nn.Softmax(dim=-1)
         
    def forward(self, x):
        N, m, d = x.shape  
        heads = []
        for i in range(len(self.w_k)):
            sender = self.w_k[i](x) # N x m x d
            sender = sender.view(N, -1, 1, sender.shape[-1]) # N x m x 1 x d
            sender = sender.repeat(1, 1, m, 1) # N x m x m x d

            receiver = self.w_q[i](x) # N x m x d
            receiver = receiver.view(N, -1, 1, receiver.shape[-1]) # N x m x 1 x d
            receiver = receiver.repeat(1, 1, m, 1) # N x m x m x d
            receiver = receiver.transpose(1,2) 
 
            #x_att = torch.cat((sender, receiver.transpose(1, 2)), dim=-1)  
            #alpha = x_att @ self.att_w[i] # N x num_node x num_node x (2 x num_feat) @ (2 x num_feat)  

            alpha = (sender * receiver).sum(-1)
            #np.savetxt(f"att{i}.txt", alpha[0].detach().cpu().numpy())
            alpha = self.relu(alpha) # N x num_node x num_node
            #np.savetxt(f"att{i}.txt", alpha[0].detach().cpu().numpy())
            #print(alpha[0][0].std())
            alpha = self.softmax(alpha)   
            #np.savetxt(f"att{i}.txt", alpha.mean(0).detach().cpu().numpy())
            #dx = alpha @ self.w_v[i](x) # N x m x m @ N x m x d

            dx = self.relu(alpha @ self.w_v[i](x)) - x # gelu() + self.self_dynamics
            #dx = alpha @ self.w_v[i](x)
            #dx = self.relu(dx) # N x num_node x num_feat1
            heads.append(dx)

        dx = torch.cat(heads, dim=-1)
        
        return dx

class GTrans(MessagePassing):
    '''
    Multiply attention by edgeweight
    '''

    def __init__(self, n_heads=1,d_input=6, d_output=6,dropout = 0.1,**kwargs):
        super(GTrans, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.d_input = d_input
        self.d_k = d_output//n_heads
        self.d_q = d_output//n_heads
        self.d_e = d_output//n_heads
        self.d_sqrt = math.sqrt(d_output//n_heads)


        #Attention Layer Initialization
        self.w_k_list = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for _ in range(self.n_heads)])
        self.w_q_list = nn.ModuleList([nn.Linear(self.d_input, self.d_q, bias=True) for _ in range(self.n_heads)])
        self.w_v_list = nn.ModuleList([nn.Linear(self.d_input, self.d_e, bias=True) for _ in range(self.n_heads)])

        #initiallization
        utils.init_network_weights(self.w_k_list)
        utils.init_network_weights(self.w_q_list)
        utils.init_network_weights(self.w_v_list)


        #Temporal Layer
        self.temporal_net = TemporalEncoding(d_input)

        #Normalization
        self.layer_norm = nn.LayerNorm(d_input,elementwise_affine = False)

    def forward(self, x, edge_index, edge_weight,time_nodes,edge_time):
        '''

        :param x:
        :param edge_index:
        :param edge_wight: edge_weight
        :param time_nodes:
        :param edge_time: edge_time_attr
        :return:
        '''

        residual = x
        x = self.layer_norm(x) 
        return self.propagate(edge_index, x=x, edges_weight=edge_weight, edge_time=edge_time, residual=residual)

    def message(self, x_j,x_i,edge_index_i, edges_weight,edge_time):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :param edge_time: [num_edge,d]
           :return:
        '''


        messages = []
        for i in range(self.n_heads):
            k_linear = self.w_k_list[i]
            q_linear = self.w_q_list[i]
            v_linear = self.w_v_list[i]

            edge_temporal_vector = self.temporal_net(edge_time) #[num_edge,d]
            edges_weight = edges_weight.view(-1, 1)
            x_j_transfer = x_j + edge_temporal_vector

            attention = self.each_head_attention(x_j_transfer,k_linear,q_linear,x_i) #[N_edge,1]
            attention = torch.div(attention,self.d_sqrt)

            # Need to multiply by original edge weight
            attention = attention * edges_weight

            attention_norm = softmax(attention,edge_index_i) #[N_neighbors_,1]
            sender = v_linear(x_j_transfer)

            message  = attention_norm * sender #[N_nodes,d]
            messages.append(message)

        message_all_head  = torch.cat(messages,1) #[N_nodes, k*d] ,assuming K head

        return message_all_head

    def each_head_attention(self,x_j_transfer,w_k,w_q,x_i):
        '''

        :param x_j_transfer: sender [N_edge,d]
        :param w_k:
        :param w_q:
        :param x_i: receiver
        :return:
        '''

        # Receiver #[num_edge,d*heads]
        x_i = w_q(x_i)
        # Sender
        sender = w_k(x_j_transfer)
        # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender,1),torch.unsqueeze(x_i,2)) #[N,1]

        return torch.squeeze(attention,1)


    def update(self, aggr_out,residual):
        x_new = residual + F.gelu(aggr_out)

        return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class GeneralConv(nn.Module):
    '''
    general layers
    '''
    def __init__(self, conv_name, in_hid, out_hid, n_heads, dropout,args):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'GTrans':
            self.base_conv = GTrans(n_heads, in_hid, out_hid, dropout)
        elif self.conv_name == "Node":
            self.base_conv = Node_GCN(in_hid,out_hid,args.num_atoms,dropout)


    def forward(self, x, edge_index, edge_weight, x_time,edge_time):

        return self.base_conv(x, edge_index, edge_weight, x_time,edge_time)


class GNN(nn.Module):
    '''
    wrap up multiple layers
    '''
    def __init__(self, in_dim, out_dim, args, conv_name = 'GTrans', is_encoder = True):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        n_hid = args.rec_dims
        n_heads = args.z0_n_heads
        n_layers = args.rec_layers
        dropout = args.dropout
        self.n_hid = n_hid
        self.out_dim = out_dim
        self.drop = nn.Dropout(dropout)
        self.is_encoder = is_encoder
        self.f_node2edge = FeedForward(2*out_dim, out_dim, 1) 

        if is_encoder:
            # If encoder, adding 1.) sequence_W 2.)transform_W ( to 2*z_dim).
            self.sequence_w = nn.Linear(n_hid,n_hid) # for encoder
            self.hidden_to_z0 = nn.Sequential(
		        nn.Linear(n_hid, n_hid//2),
		        nn.Tanh(),
		        nn.Linear(n_hid//2, out_dim))
            self.adapt_w = nn.Linear(in_dim,n_hid)
            utils.init_network_weights(self.sequence_w)
            utils.init_network_weights(self.hidden_to_z0)
            utils.init_network_weights(self.adapt_w)
        else: # ODE GNN
            assert self.in_dim == self.n_hid 
        # first layer is input layer
        for l in range(0,n_layers):
            self.gcs.append(GeneralConv(conv_name, self.n_hid, self.n_hid,  n_heads, dropout, args))

        if conv_name in  ['GTrans'] :
            self.temporal_net = TemporalEncoding(n_hid)  #// Encoder, needs positional encoding for sequence aggregation.

    def forward(self, x, edge_weight=None, edge_index=None, x_time=None, edge_time=None,batch= None, batch_y = None, batch_size=1):  #aggregation part
        #print('\nx.shape', x.shape, '\n')

        if not self.is_encoder: #Encoder initial input node feature
            h_t = self.drop(x)
        else:
            h_t = self.drop(F.gelu(self.adapt_w(x)))  #initial input for encoder 

        for gc in self.gcs:
            h_t = gc(h_t, edge_index, edge_weight, x_time,edge_time)  #[num_nodes,d]

        ### Output
        if batch!= None:  ## for encoder
            batch_new = self.rewrite_batch(batch,batch_y) #group by balls 
            h_t += self.temporal_net(x_time)
            attention_vector = F.gelu(
                self.sequence_w(global_mean_pool(h_t, batch_new)))  # [num_ball,d] ,graph vector with activation Relu
            attention_vector_expanded = self.attention_expand(attention_vector, batch, batch_y)  # [num_nodes,d]
            attention_nodes = torch.sigmoid(
                torch.squeeze(torch.bmm(torch.unsqueeze(attention_vector_expanded, 1), torch.unsqueeze(h_t, 2)))).view(
                -1, 1)  # [num_nodes] 
            nodes_attention = attention_nodes * h_t  # [num_nodes,d]
            h_ball = global_mean_pool(nodes_attention, batch_new)  # [num_ball,d] without activation 
            h_out = self.hidden_to_z0(h_ball) #[num_ball,2*z_dim] Must ganrantee NO 0 ENTRIES!
             
            h_out = h_out.reshape(batch_size, -1, h_out.shape[-1])
             
            edge = utils.construct_pair(h_out)
            edge = self.f_node2edge(edge)
            return h_out, edge.squeeze(-1)
            #mean,mu = self.split_mean_mu(h_out)
            #mu = mu.abs()
            #return mean,mu  
        else:  # for ODE
            h_out = h_t
            return h_out

    def rewrite_batch(self,batch, batch_y):
        #print('\nrewrite batch batch.shape', batch.shape, 'batch_y.shape', batch_y.shape, '\n')
        assert (torch.sum(batch_y).item() == list(batch.size())[0])
        batch_new = torch.zeros_like(batch)
        group_num = 0
        current_index = 0 
        for ball_time in batch_y:
            batch_new[current_index:current_index + ball_time] = group_num
            group_num += 1
            current_index += ball_time.item() 
        return batch_new

    def attention_expand(self,attention_ball, batch,batch_y):
        ''' 
        :param attention_ball: [num_ball, d]
        :param batch: [num_nodes,d]
        :param batch_new: [num_ball,d]
        :return:
        '''
        node_size = batch.size()[0]
        dim = attention_ball.size()[1]
        new_attention = torch.ones(node_size, dim).to(attention_ball.device)
        # if attention_ball.device != torch.device("cpu"):
        #     new_attention = new_attention.cuda()

        group_num = 0
        current_index = 0
        for ball_time in batch_y:
            new_attention[current_index:current_index+ball_time] = attention_ball[group_num]
            group_num +=1
            current_index += ball_time.item()

        return new_attention