# In[Import]
import torch
import torch.nn as nn

import os
import os.path as op

from gnn import *
from datasets.datasetba3 import *
from datasets.datasetadvba3 import *

from datasets.datasetmut import *
from datasets.datasetadvmut import *

from ogb.graphproppred import PygGraphPropPredDataset

# In[TrainProxy]
def TrainProxy(TrainLoader,model,opt,device,LossF):
    model.train()
    LOSS = 0
    ACC = 0
    
    for data in TrainLoader:
        data.to(device)
        # pred = model(data.x,data.edge_index,
        #              data.edge_attr,data.batch)
        
        pred = model(data)
        # print('pred:',pred)
        # print('pred.shape:',pred.shape)
        
        data.y = torch.squeeze(data.y,dim = 1)
        # print('data.y:',data.y)
        # print('data.y.shape:',data.y.shape)
        
        yhat = pred.argmax(dim = 1)
        
        loss = LossF(pred,data.y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        LOSS += loss.item() * data.num_graphs
        ACC += yhat.eq(data.y).sum().item()
    
    return LOSS / len(TrainLoader.dataset), ACC / len(TrainLoader.dataset)

# In[TestProxy]
def TestProxy(TestLoader,model,device,LossF):
    model.eval()
    LOSS = 0
    ACC = 0
    
    with torch.no_grad():
        for data in TestLoader:
            # 'data' is a DataBatch.
            data = data.to(device)
            pred = model(data.x,data.edge_index,
                         data.edge_attr,data.batch)
            data.y = torch.squeeze(data.y,dim = 1)
            
            yhat = pred.argmax(dim = 1)
            
            LOSS += LossF(pred,data.y) * data.num_graphs
            ACC += yhat.eq(data.y).sum().item()
  
    ACC = float(ACC)
    
    return LOSS / len(TestLoader.dataset), ACC / len(TestLoader.dataset)

# In[DataLoader]
def get_dataset(name, root = '../data/'):
    
    if name == 'ba3':
        MODE = ['train','valid','test']
        folder = op.join(root, name.upper())
        Train = ba3motif(folder,MODE[0])
        Valid = ba3motif(folder,MODE[1])
        Test = ba3motif(folder,MODE[2])
    elif name == 'advba3':
        MODE = ['train','valid','test']
        folder = op.join(root,'BA3')
        Train = advba3(folder,MODE[0])
        Valid = advba3(folder,MODE[1])
        Test = advba3(folder,MODE[2])
    elif name == 'mut':
        MODE = ['training', 'evaluation', 'testing']
        folder = op.join(root,'MUT')
        Train = Mutagenicity(folder,MODE[0])
        Valid = Mutagenicity(folder,MODE[1])
        Test = Mutagenicity(folder,MODE[2])
    elif name == 'advmut':
        MODE = ['train','valid','test']
        folder = op.join(root,'MUT')
        Train = advmut(folder,MODE[0])
        Valid = advmut(folder,MODE[1])
        Test = advmut(folder,MODE[2])
    elif name == 'hiv':
        MODE = ['train','valid','test']
        molhiv = PygGraphPropPredDataset(name = 'ogbg-molhiv',
                                         root = './data/')
        split_idx = molhiv.get_idx_split()
        Train = molhiv[split_idx[MODE[0]]]
        Valid = molhiv[split_idx[MODE[1]]]
        Test = molhiv[split_idx[MODE[2]]]
    elif name == 'ppa':
        MODE = ['train','valid','test']
        ppa = PygGraphPropPredDataset(name = 'ogbg-ppa',
                                         root = './data/')
        split_idx = ppa.get_idx_split()
        Train = ppa[split_idx[MODE[0]]]
        Valid = ppa[split_idx[MODE[1]]]
        Test = ppa[split_idx[MODE[2]]]
    elif name == 'cod':
        MODE = ['train','valid','test']
        code2 = PygGraphPropPredDataset(name = 'ogbg-code2',
                                         root = './data/')
        split_idx = code2.get_idx_split()
        Train = code2[split_idx[MODE[0]]]
        Valid = code2[split_idx[MODE[1]]]
        Test = code2[split_idx[MODE[2]]]
    return Train,Valid,Test
