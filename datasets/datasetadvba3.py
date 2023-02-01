'''
Dateset: Adversarial BA-3Motif.
Create InMemoryDataset.
'''
# In[Import]
import os
import random
import numpy as np
import pickle as pkl
import os.path as op

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

# In[advba3]
class advba3(InMemoryDataset):
    # Split dataset to 3 parts.
    splits = ['train','valid','test']
    
    def __init__(self,root,mode = 'test',
                 transform = None,
                 pre_transform = None,
                 pre_filter = None):
        assert mode in self.splits
        self.mode = mode
        
        super().__init__(root,transform,
                         pre_transform,pre_filter)
        self.adv_paths = [r'./data/BA3/adv005/train.pt',
                          r'./data/BA3/adv005/valid.pt',
                          r'./data/BA3/adv005/test.pt']
        
        idx = self.processed_file_names.index('{}.pt'.format(mode))
        self.data,self.slices = torch.load(self.adv_paths[idx])
        
        '''
        In: Train.slices
        Out: 
        defaultdict(dict,
                    {'x': tensor([    0,    15,    30,  ..., 48318, 48334, 48350]),
                     'edge_index': tensor([    0,    24,    46,  ..., 64972, 64990, 65010]),
                     'edge_attr': tensor([    0,    24,    46,  ..., 64972, 64990, 65010]),
                     'y': tensor([   0,    1,    2,  ..., 2198, 2199, 2200]),
                     'Role': tensor([   0,    1,    2,  ..., 2198, 2199, 2200]),
                     'GTMask': tensor([   0,    1,    2,  ..., 2198, 2199, 2200]),
                     'name': tensor([   0,    1,    2,  ..., 2198, 2199, 2200])})
        
        '''
        
    @property
    def raw_file_names(self):
        return ['BA-3motif.npy']

    @property
    def processed_file_names(self):
        return ['train.pt','valid.pt','test.pt']
    
    def download(self):
        pass
            
    def process(self):
        pass