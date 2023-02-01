'''
Dateset: Adversarial Mutagenicity.
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

# In[advmut]
class advmut(InMemoryDataset):
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
        self.adv_paths = [r'./data/MUT/adv005/train.pt',
                          r'./data/MUT/adv005/valid.pt',
                          r'./data/MUT/adv005/test.pt']
        
        idx = self.processed_file_names.index('{}.pt'.format(mode))
        self.data,self.slices = torch.load(self.adv_paths[idx])
        
    @property
    def raw_file_names(self):
        return ['Mutagenicity/' + i \
                for i in [
                    'Mutagenicity_A.txt',
                    'Mutagenicity_edge_labels.txt',
                    'Mutagenicity_graph_indicator.txt',
                    'Mutagenicity_graph_labels.txt',
                    'Mutagenicity_node_labels.txt',
                    ]
                ]
                
    @property
    def processed_file_names(self):
        return ['train.pt','valid.pt','test.pt']
    
    def download(self):
        pass
            
    def process(self):
        pass