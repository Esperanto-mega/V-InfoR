'''
Attack graph classification datasets.
'''
# In[Import]
import os
import random
import pickle
import os.path as op
import numpy as np
import pandas as pd
from copy import deepcopy

import sys
father_path = '/home/data/yj/Anova/#CodeNew'
if father_path not in sys.path:
    sys.path.append(father_path)
# print(sys.path)

'''DGL & PyG.'''
import dgl
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader

from utils.train import get_dataset
from train.config import root
from utils.PyG2DGL import pyg2dgl, collate
# from attack.gcn import GCNGraphClassifier
from gcn import GCNGraphClassifier
from utils.seed import GlobalSeed
from attack.bayesopt_attack import BayesOptAttack
from attack.utils import nettack_loss

from ogb.graphproppred import PygGraphPropPredDataset

# In[GPU]
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    
# In[Random Seed]
seed = 42
GlobalSeed(seed)

# In[Dataset]
# dataset to be attacked.
dataset = 'mut'
batchsize = 20
# data_path = './data/'

# In[DATASET]
# filename = f'{dataset}-adv_example.pickle'
train, valid, test = get_dataset(dataset, root['data'])
# DATASET = [train,valid,test]

# dataset = PygGraphPropPredDataset(name = dataset,
#                                   root = './data/')
# split_idx = dataset.get_idx_split()

# train = dataset[split_idx['train']]
# valid = dataset[split_idx['valid']]
# test = dataset[split_idx['test']]

# In[PyG2DGL]
'''
Convert PyG data to DGL-graph.
Only need once per dataset.
'''
pyg2dgl(dataset = train,
        savepath = './data/',
        filename = 'train-mut-005-adv_example.pickle',
        info = True)

pyg2dgl(dataset = valid,
        savepath = './data/',
        filename = 'valid-mut-005-adv_example.pickle',
        info = True)
pyg2dgl(dataset = test,
        savepath = './data/',
        filename = 'test-mut-005-adv_example.pickle',
        info = True)

# # In[Adversarial-Set]
train_path = os.path.join(root['data'],'train-mut-005-adv_example.pickle')
valid_path = os.path.join(root['data'],'valid-mut-005-adv_example.pickle')
test_path = os.path.join(root['data'],'test-mut-005-adv_example.pickle')

train = pickle.load(open(train_path,'rb'))
valid = pickle.load(open(valid_path,'rb'))
test = pickle.load(open(test_path,'rb'))

train = [(graph,
           torch.IntTensor([graph.ndata['y'][0][0]])) for graph in train]
# print('Train num:',len(train))
valid = [(graph,
           torch.IntTensor([graph.ndata['y'][0][0]])) for graph in valid]
# print('Valid num:',len(valid))
test = [(graph,
           torch.IntTensor([graph.ndata['y'][0][0]])) for graph in test]
# print('Test num:',len(test))

graphs = []
graphs.extend(train)
graphs.extend(valid)
graphs.extend(test)

# # adversarial_path = os.path.join(root['data'],'test-mut-adv_example.pickle')
# # graphs = pickle.load(open(adversarial_path,'rb'))

# # graphs = [(graph,
# #            torch.IntTensor([graph.ndata['y'][0][0]])) for graph in graphs]

# # In[YJ]
# '''
# ****************** Start training attack surrogate... *******************
# *************************************************************************
# *************************************************************************
# *************************************************************************
# '''
# # In[Dataloader]
# # split = 0.8

# dataset = 'hiv'
# adversarial_loader_train = DataLoader(train,
#                                       batch_size = batchsize,
#                                       shuffle = True, collate_fn = collate)

# adversarial_loader_valid = DataLoader(valid, 
#                                       batch_size = batchsize,
#                                       shuffle = False, collate_fn = collate)

# # In[Model]
# feature_dim = valid[0][0].ndata['x'].shape[1]
# classes = len(np.unique([datapoint[1] for datapoint in valid]))
# lr = 1e-3
# weight_decay = 1e-4
# atk_surrogate = GCNGraphClassifier(feature_dim, classes).to(device)

# loss_fn = nn.functional.cross_entropy
# optimizer = opt.Adam(atk_surrogate.parameters(),
#                      lr = lr,
#                      weight_decay = weight_decay)
# best_val_acc = 0.
# best_model = None
# training_logs = []

# # In[Epochs]
# epochs = 50

# # # In[Train]
# for epoch in range(epochs):

#     # training step
#     atk_surrogate.train()
#     train_loss, train_acc = 0, 0
#     for i, (graphs, labels) in enumerate(adversarial_loader_train):
#         graphs, labels = graphs.to(device), labels.to(device)
#         labels = labels.long()
#         predictions = atk_surrogate(graphs)

#         loss = loss_fn(predictions, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.detach().item()
#         train_acc += (predictions.argmax(axis=1) == labels).sum().detach().item()
#     train_loss /= len(adversarial_loader_train)
#     train_acc /= len(adversarial_loader_train.dataset)

#     # evaluation step
#     atk_surrogate.eval()
#     valid_loss, valid_acc = 0, 0
#     with torch.no_grad():
#         for i, (graphs, labels) in enumerate(adversarial_loader_train):
#             graphs, labels = graphs.to(device), labels.to(device)
#             labels = labels.long()
#             predictions = atk_surrogate(graphs)

#             loss = loss_fn(predictions, labels)
#             valid_loss += loss.detach().item()
#             valid_acc += (predictions.argmax(axis=1) == labels).sum().detach().item()
#         valid_loss /= len(adversarial_loader_train)
#         valid_acc /= len(adversarial_loader_train.dataset)

#     # save best model
#     if valid_acc > best_val_acc:
#         print('Best val acc recorded at epoch ', epoch)
#         best_model = deepcopy(atk_surrogate)
#         best_val_acc = valid_acc

#     print(epoch, '{:.4f}'.format(train_loss), '{:.4f}'.format(valid_loss),
#           '{:.2f}'.format(train_acc), '{:.2f}'.format(valid_acc))
#     training_logs.append([epoch, train_loss, valid_loss, train_acc, valid_acc])

# # In[Save]
# # save model
# os.makedirs(op.join('atk-surrogate', 'models'), exist_ok=True)
# model_path = op.join('atk-surrogate', 'models', 
#                       f'atk_surrogate_{dataset}_{seed}.pt')
# torch.save(best_model.state_dict(), model_path)

# # save training information
# # os.makedirs(op.join('atk-surrogate', 'training_logs'), exist_ok=True)
# # training_logs_path = op.join('atk-surrogate', 'training_logs', 
# #                               f'atk_surrogate_{dataset}_{seed}.csv')

# # training_logs = pd.DataFrame(training_logs, 
# #                               columns=['epoch', 'train_loss', 
# #                                       'valid_loss', 'train_acc', 
# #                                       'valid_acc'])
# # training_logs.to_csv(training_logs_path)

# # In[YJ]
# '''
# ****************** Finish training attack surrogate... ******************
# *************************************************************************
# *************************************************************************
# *************************************************************************
# '''

# ---------------------------- Attack ----------------------------------- #
# ----------------------------------------------------------------------- #
# In[Save path]
save_path = f'./data/{dataset}_adv005/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# In[Load Model]
dataset = 'mut'
adversarial_loader = DataLoader(graphs,
                                batch_size = batchsize,
                                shuffle = False, collate_fn = collate)

feature_dim = graphs[0][0].ndata['x'].shape[1]
classes = len(np.unique([datapoint[1] for datapoint in graphs]))
model = GCNGraphClassifier(feature_dim, classes)

model_path = op.join('atk-surrogate', 'models', 
                     f'atk_surrogate_{dataset}_{seed}.pt')
model_stat = torch.load(model_path, map_location = 'cpu')

model.load_state_dict(model_stat)
# model = model.to(device)
model.eval()

# In[Correct]
all_graphs = []
all_labels = []

for i, (graphs, labels) in enumerate(adversarial_loader):
    with torch.no_grad():
        graphs = dgl.unbatch(graphs)
        all_graphs += graphs
        all_labels += labels.numpy().tolist()

correct_ids = []
        
for i in range(len(all_labels)):
    sample = all_graphs[i]
    label = all_labels[i]
    pred = model(sample).detach()
    
    if pred.argmax() == label:
        correct_ids.append(i)
        
print('Correct classified samples:',
      len(correct_ids),'/',len(all_labels),'.')

'''************************* 2986/3000 **************************'''

# In[Prepare]
num_success = 0
num_trials = 0
budget = 0.05
budget_by = 'node2'
dfs, adv_examples = [], []


all_labels = torch.Tensor(all_labels)

# Sample to be attacked.
atk_ids = np.arange(0, len(all_labels), 1)
L = len(atk_ids)

atk_mode = 'rewire'
# atk_mode = 'flip'

# In[Attack]
for trial in range(num_trials, num_trials + 1):
    print(f'Starting trial {trial}/{num_trials}.')
    is_successful = [0] * L
    is_attacked = [0] * L
    is_correct = [0] * L
    num_edges = [0] * L
    num_nodes = [0] * L
    
    for i, graph_id in enumerate(atk_ids):
        n_stagnation = 0
        best_loss = -np.inf
        query_per_atk = 1
        num_nodes[i] = int(all_graphs[graph_id].num_nodes())
        num_edges[i] = int(all_graphs[graph_id].num_edges() // 2)
        print(f'Starting graph {graph_id} ',
              f'(#nodes = {num_nodes[i]}, #edges = {num_edges[i]}).')
        
        if graph_id in correct_ids:
            is_attacked[i] = 1
            is_correct[i] = 1
            
            graph, label = all_graphs[graph_id], all_labels[graph_id]
            
            # graph, label = graph.to(device), label.to(device)
            
            if atk_mode == 'rewire':
                query_per_atk = 2 * query_per_atk
                if budget_by == 'node2':
                    edit = min(int(2e4 / query_per_atk),
                               np.round(budget * graph.num_nodes() ** 2 // 2).astype(int) // 2) + 1
                else:
                    edit = min(int(2e4 / query_per_atk),
                               np.round(budget * graph.num_edges() // 2 // 2).astype(int) // 2) + 1
            else:
                if budget_by == 'node2':
                    edit = min(int(2e4 / query_per_atk),
                               np.round(budget * graph.num_nodes() ** 2).astype(int)) + 1
                else:
                    edit = min(int(2e4 / query_per_atk),
                               np.round(budget * graph.num_edges() // 2).astype(int)) + 1
            
            attacker = BayesOptAttack(model, nettack_loss,
                                      batch_size = batchsize,
                                      mode = atk_mode)
            
            print(f'edit:{edit}, Budget:{edit * query_per_atk}')
            df, adv_example = attacker.attack(graph, label, edit,
                                              edit * query_per_atk)
            
            if adv_example is not None:
                num_success += 1
                is_successful[i] = 1
            
            dfs.append(df)
            adv_examples.append(adv_example)
        
        else:
            dfs.append(None)
            adv_examples.append(None)
        states = {'atk_ids':atk_ids,
                  'is_successful':is_successful,
                  'is_attacked':is_attacked,
                  'is_correct':is_correct,
                  '#nodes':num_nodes,
                  '#edges':num_edges}
        
        pickle.dump(dfs, open(op.join(save_path, f'trial-{trial}.pickle'), 'wb'))
        pickle.dump(adv_examples, open(op.join(save_path, f'trial-{trial}-adv_example.pickle'), 'wb'))
        pickle.dump(states, open(op.join(save_path, f'trial-{trial}-states.pickle'), 'wb'))
        
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #