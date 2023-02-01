'''
Architecture for Subgraph Generator in Explainers.
'''
# In[Import]
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import ModuleList
from torch.nn import Linear
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import ARMAConv

from collections import OrderedDict

from explainer.overload import overload

# In[MLP]
class MLP(nn.Module):
    def __init__(self,InChannels,Hidden,OutChannels,
                 Activate = nn.ReLU()):
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([
            ('Linear1',Linear(InChannels,Hidden)),
            ('Activate',Activate),
            ('Linear2',Linear(Hidden,OutChannels))]))
        
    def forward(self,x):
        return self.mlp(x)
    
# In[SubgraphNet]
class SubgraphNet(nn.Module):
    def __init__(self,NodeChannels,EdgeChannels,
                 Hidden = 72,Layers = 3):
        super().__init__()
        
        self.NodeLinear = Linear(NodeChannels,Hidden)
        
        self.CONV = ModuleList()
        self.BN = ModuleList()
        
        for _ in range(Layers):
            conv = ARMAConv(in_channels = Hidden, 
                            out_channels = Hidden)
            bn = BatchNorm(Hidden)
            self.CONV.append(conv)
            self.BN.append(bn)
            
        if EdgeChannels > 1:
            self.EdgeLinear1 = Linear(2 * Hidden, Hidden)
            self.EdgeLinear2 = Linear(EdgeChannels, Hidden)
            
        self.mlp = MLP(2 * Hidden, Hidden, 1)
        
        self.weight_init_(mode = 'kaiming')
        
    def weight_init_(self,mode = 'kaiming'):
        for module in self.modules():
            if isinstance(module, Linear):
                if mode == 'kaiming':
                    nn.init.kaiming_normal_(module.weight)
                elif mode == 'xavier':
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.normal_(module.weight)

    @overload
    def forward(self,x,EdgeID,EdgeAttr):
        x = torch.flatten(x,1,-1)
        '''
        Flattens input by reshaping it into a one-dimensional tensor. 
        If start_dim or end_dim are passed, only dimensions starting 
        with start_dim and ending with end_dim are flattened. The 
        order of elements in input is unchanged.
        '''
        x = F.relu(self.NodeLinear(x))
        
        for conv,bn in zip(self.CONV,self.BN):
            x = F.relu(conv(x,EdgeID))
            x = bn(x)
            
        e = torch.cat([x[EdgeID[0,:]],
                       x[EdgeID[1,:]]],
                      dim = 1)

        if EdgeAttr.size(-1) > 1:
            e1 = self.EdgeLinear1(e)
            e2 = self.EdgeLinear2(EdgeAttr)
            e = torch.cat([e1,e2], dim = 1)
            
        return self.mlp(e)