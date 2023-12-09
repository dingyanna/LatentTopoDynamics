import networkx as nx
import numpy as np   

class GeneralNet:
    '''
    A general network.
    '''
    def __init__(self, N, p, dyn, name, topo='er', m=4, directed=True, seed=42):
 
        self.N = N
        self.name = name
        self.topo = topo
        self.dyn = dyn
        if topo == 'er':
            G = nx.fast_gnp_random_graph(N, p, seed=seed, directed=directed)
        else:
            G = nx.barabasi_albert_graph(N, m, seed=seed)
        self.G = G
        self.A = nx.to_numpy_array(G, nodelist=range(N))
        self.degree = self.A.sum(axis=0) # in degree
        self.deg_unique, self.block_sizes = np.unique(self.degree, return_counts=True)
        self.beta = np.sum(self.A @ self.degree) / np.sum(self.A)
        self.H = np.std(self.degree) ** 2 / np.mean(self.degree)
        self.sampled = range(N)
        #self.Dmap = self.mfa_plus
        self.A_exist = True
        self.directed = directed
        self.int_step_size = 0.001

    def setTopology(self, A):
        self.A = A
        self.degree = self.A.sum(axis=1)
        self.out_degree = self.A.sum(0)
        binA = np.zeros((len(A),len(A)))
        binA[A != 0] = 1
        self.binA = binA 
        self.N = A.shape[0] 
    
   
      