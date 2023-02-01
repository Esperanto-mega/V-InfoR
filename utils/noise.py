# In[Import]
import torch
import random
import numpy as np

from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import dense_to_sparse

# In[Noise]
def add_noise(g, ratio):
    edge_dim = g.edge_attr.shape[1]
    noise_num = int(np.round(ratio * g.num_edges))
    noise_num = min(max(noise_num, 1),g.num_edges)
    
    adj = to_dense_adj(g.edge_index)
    adj = torch.squeeze(adj, dim = 0)
    
    row = random.sample(range(0,g.num_nodes),noise_num)
    col = random.sample(range(0,g.num_nodes),noise_num)
    for (i,j) in (row,col):
        # Flip the edge.
        adj[i,j] = 1 - adj[i,j]
        
    g.edge_index, g.edge_attr = dense_to_sparse(adj)
    g.edge_attr = torch.unsqueeze(g.edge_attr,dim = 1)
    g.edge_attr = g.edge_attr.repeat(1,edge_dim)
    return g