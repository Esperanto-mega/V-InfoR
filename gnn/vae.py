# In[Import]
import torch
import torch.nn as nn
from torch.nn import ReLU

from torch_geometric.nn import GINConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm

from collections import OrderedDict

# In[AutoEncoder]
class VAE(nn.Module):
    def __init__(self, node_dim, 
                 hidden_dim, out_dim, device):
        super().__init__()
        self.in_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device
        
        self.relu = ReLU()

        # GCNConv
        # ----------------------------------------------------------- #
        # ----------------------------------------------------------- #
        self.first_gcn = GCNConv(self.in_dim, self.hidden_dim)
        self.first_bn = BatchNorm(hidden_dim)
        self.second_gcn = GCNConv(self.hidden_dim, self.hidden_dim)
        self.second_bn = BatchNorm(hidden_dim)
        
        self.mean_gcn = GCNConv(self.hidden_dim,self.out_dim)
        self.log_stddev_gcn = GCNConv(self.hidden_dim,self.out_dim)
        # ----------------------------------------------------------- #
        
    def weight_init_(self,mode = 'kaiming'):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if mode == 'kaiming':
                    nn.init.kaiming_normal_(module.weight)
                elif mode == 'xavier':
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.normal_(module.weight)
    
    def encode(self, g, edge_weight = None):
        # Type of g.x should be torch.float
        g.x = g.x.float()
        # g.edge_index = g.edge_index.float()

        hidden_x = self.first_gcn(g.x,g.edge_index,edge_weight)
        hidden_x = self.relu(hidden_x)
        # hidden_x = self.first_bn(hidden_x)
        # print('first hidden x:',hidden_x)
        hidden_x = self.second_gcn(hidden_x,g.edge_index,edge_weight)
        hidden_x = self.relu(hidden_x)
        # hidden_x = self.second_bn(hidden_x)
        # print('second hidden x:',hidden_x)
        
        self.mean = self.mean_gcn(hidden_x,g.edge_index,edge_weight)
        self.mean = self.relu(self.mean)
        self.log_stddev = self.log_stddev_gcn(hidden_x,g.edge_index,edge_weight)
        self.log_stddev = self.relu(self.log_stddev)
        
        # print('g.x shape:',g.x.shape)
        # print('mean shape:',self.mean.shape)
        # print('log_stddev shape:',self.log_stddev.shape)
        # print('num_nodes:',g.num_nodes)
        
        gauss_noise = torch.randn(g.x.shape[0], self.out_dim,
                                  device = self.device)
        # print('noise shape:',gauss_noise.shape)
        
        # print('gauss noise dim:',gauss_noise.shape[0])
        # print('mean:',self.mean.device)
        # print('stddev:',self.log_stddev.device)
        # print('noise:',gauss_noise.device)
        
        # print('self.mean:',self.mean)
        # print('self.log_stddev:',self.log_stddev)
        
        self.sampled_z = gauss_noise * torch.exp(self.log_stddev) + self.mean
        # print('sampled_z dim:',self.sampled_z.shape)
        return self.sampled_z
    
    def decode(self, sampled_z):
        # print('sampled_z:',sampled_z)
        adj_pred = torch.matmul(sampled_z,sampled_z.t())
        adj_pred = torch.sigmoid(adj_pred)
        return adj_pred
    
    def forward(self, g, edge_weight = None):
        sampled_z = self.encode(g, edge_weight)
        adj_pred = self.decode(sampled_z)
        
        # upper triangular part of matrix. 
        # params: diagonal.
        adj_pred = torch.triu(adj_pred,diagonal = 1)
        return adj_pred
        
# In[AlphaNet]
class AlphaNet(nn.Module):
    def __init__(self, edge_dim, in_dim, hidden_dim, out_dim = 1):
        super().__init__()
        self.edge_dim = edge_dim
        
        if self.edge_dim > 1:
            self.linear_node_attr = nn.Linear(2 * in_dim, hidden_dim)
            self.linear_edge_attr = nn.Linear(self.edge_dim, hidden_dim)
            
            self.mlp = nn.Sequential(OrderedDict([
            ('Linear1', nn.Linear(2 * hidden_dim, hidden_dim)),
            ('Activate', nn.ReLU()),
            ('Linear2', nn.Linear(hidden_dim, out_dim))]))
        else:
            # 2 * in_dim: edge dim = 2 * node dim
            self.mlp = nn.Sequential(OrderedDict([
                ('Linear1', nn.Linear(2 * in_dim, hidden_dim)),
                ('Activate', nn.ReLU()),
                ('Linear2', nn.Linear(hidden_dim, out_dim))]))
        
        self.weight_init_(mode = 'kaiming')
        
    def weight_init_(self,mode = 'kaiming'):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if mode == 'kaiming':
                    nn.init.kaiming_normal_(module.weight)
                elif mode == 'xavier':
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.normal_(module.weight)
    
    def forward(self, g):
        x, edge_index, edge_attr = g.x, g.edge_index, g.edge_attr.float()
        # print('x.shape:',x.shape)
        # print('edge_index.shape:',edge_index.shape)
        e = torch.cat([x[edge_index[0,:]],
                       x[edge_index[1,:]]],
                      dim = 1)
        
        # print('e.shape',e.shape)
        # print('edge_attr.shape',edge_attr.shape)
        
        if self.edge_dim > 1:
            e1 = self.linear_node_attr(e)
            e2 = self.linear_edge_attr(edge_attr)
            e = torch.cat([e1,e2], dim = 1)
        
        # print('edge_attr.shape:',e.shape)
        alpha = self.mlp(e)
        
        '''
        parameter alpha.
        alpha.shape [edge_num,1]
        '''
        return alpha
        
        
        
        
        
        
        
        
        
        