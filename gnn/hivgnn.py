# In[Import]
import time
import argparse
from typing import overload
import os.path as osp

# import sys
# father_path = ''
# if father_path not in sys.path:
#     sys.path.append(father_path)

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ogb.graphproppred import PygGraphPropPredDataset

from gnn.overload import overload
from utils.seed import GlobalSeed
from utils.train import TrainProxy
from utils.train import TestProxy

# In[Args]
def parse_args():
    parser = argparse.ArgumentParser(description="Train molhiv Model")
    
    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'param', 'gnns'),
                        help='path for saving trained model.')
    parser.add_argument('--cuda', type=int, default=2,
                        help='GPU device.')
    parser.add_argument('--epoch', type=int, default=50, 
                        help='Number of epoch.')
    parser.add_argument('--lr', type=float, default= 1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size.')
    parser.add_argument('--test', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='ogbg-ppa')
    parser.add_argument('--node_dim', type=int, default=0)
    parser.add_argument('--edge_dim', type=int, default=7)
    parser.add_argument('--out_dim', type=int, default=37)
    parser.add_argument('--linear_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--layers', type=int, default=5)

    return parser.parse_args()

# In[HivNet]
class MolhivNet(torch.nn.Module):
    def __init__(self, node_dim = 9, edge_dim  = 3,
                 linear_dim = 32,out_dim = 2,
                 hidden_dim = 300, layers = 5):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.linear_dim = linear_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.out_dim = out_dim
        
        # need to concatenate
        self.node_emb = nn.Linear(self.node_dim,self.linear_dim)
        self.edge_emb = nn.Linear(self.edge_dim,self.linear_dim)
        
        # graph convolution layer
        self.first_gcn = GCNConv(self.linear_dim,self.hidden_dim)
        self.gcn = nn.ModuleList()
        for _ in range(0,self.layers - 1):
            gcn_conv = GCNConv(self.hidden_dim,self.hidden_dim)
            self.gcn.append(gcn_conv)
            
        # classiifer
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(self.hidden_dim,self.linear_dim)
        self.linear2 = nn.Linear(self.linear_dim,self.out_dim)
        self.softmax = nn.Softmax(dim = 1)
        
    @overload
    def forward(self,x,edge_index,edge_attr,batch):
        graph_x = self.get_graph_rap(x,edge_index,edge_attr,batch)
        pred = self.get_pred(graph_x)
        return pred
    
    @overload
    def NodeRep(self,x,edge_index,edge_attr,batch):
        # print('edge_attr shape:',edge_attr.shape)
        # print('edge_index shape:',edge_index.shape)
        
        num_edges = edge_index.shape[1]
        
        edge_attr = self.edge_emb(edge_attr.float())
        
        if self.node_dim != 0:
            attr = self.node_emb(x.float())
            for i in range(0,num_edges):
                src_idx = edge_index[0,i]
                attr[src_idx,:] = attr[src_idx,:] + edge_attr[i,:]
        
        # # [29,9]
        # print('x shape:',x.shape)
        # # [29,32]
        # print('attr shape:',attr.shape)
        # # [62,32]
        # print('edge_attr shape:',edge_attr.shape)
        # # [2,62]
        # print('edge_index shape:',edge_index.shape)
        
        attr = edge_attr
        # print('attr shape:',attr.shape)
        attr = self.first_gcn(attr,edge_index)
        attr = self.relu(attr)
        for gcn in self.gcn:
            attr = gcn(attr,edge_index)
            attr = self.relu(attr)
            
        # print('attr shape:',attr.shape)
            
        return attr
    
    @overload
    def get_graph_rap(self,x,edge_index,edge_attr,batch):
        attr = self.NodeRep(x,edge_index,edge_attr,batch)
        graph_x = torch.mean(attr, dim = 0, keepdim = True)
        # print('graph_x shape:',graph_x.shape)
        # graph_x = global_mean_pool(attr)
        return graph_x
    
    def get_pred(self,graph_x):
        graph_x = self.linear1(graph_x)
        graph_x = self.relu(graph_x)
        pred = self.linear2(graph_x)
        self.readout = self.softmax(pred)
        return pred
    
    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

if __name__ == "__main__":
    args = parse_args()
    GlobalSeed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.cuda))
    else:
        device = torch.device('cpu')
        
    # In[Dataset]
    dataset = PygGraphPropPredDataset(name = args.dataset,
                                      root = 'data/')
    
    # g = dataset[0]
    # print('Instance:',g)
    
    split_idx = dataset.get_idx_split()
        
    train_loader = DataLoader(dataset[split_idx['train']],
                              batch_size = args.batch_size,
                              shuffle = True)
    valid_loader = DataLoader(dataset[split_idx['valid']],
                              batch_size = args.batch_size,
                              shuffle = False)
    test_loader = DataLoader(dataset[split_idx['test']],
                              batch_size = args.batch_size,
                              shuffle = False)
    
    GNN = MolhivNet(args.node_dim, args.edge_dim,
                    args.linear_dim, args.out_dim).to(device)
    opt = torch.optim.Adam(GNN.parameters(),lr = args.lr)
    sdl = ReduceLROnPlateau(opt, mode = 'min', factor = 0.8,
                            patience = 10, min_lr = 1e-4)
    
    min_val_loss = None
    loss = nn.CrossEntropyLoss()
    
    # In[Train & Test]
    for epoch in range(1,args.epoch + 1):
        print('Epoch:',epoch,'is under training.')
        t1 = time.time()
        train_loss, train_acc = TrainProxy(train_loader, GNN, opt, device, loss)
        valid_loss, valid_acc = TestProxy(valid_loader, GNN, device, loss)
        if min_val_loss is None or valid_loss < min_val_loss:
            min_val_loss = valid_loss
        sdl.step(valid_loss)
        t2  = time.time()
        
        lr = sdl.optimizer.param_groups[0]['lr']
        if epoch % args.test == 0:
            test_loss, test_acc = TestProxy(test_loader, GNN, device, loss)
            t3 = time.time()
            print('Epoch{:4d}[{:.3f}]: LR:{:.5f},'.format(epoch,t3-t1,lr),
                  'Train Loss: {:.5f}, Train Acc: {:.5f}'.format(train_loss,train_acc),
                  'Valid Loss: {:.5f}, Valid Acc: {:.5f}'.format(valid_loss,valid_acc),
                  'Test Loss: {:.5f}, Test Acc: {:.5f}'.format(test_loss,test_acc))
        else:
            print('Epoch{:4d}[{:.3f}]: LR:{:.5f},'.format(epoch,t2-t1,lr),
                  'Train Loss: {:.5f}, Train Acc: {:.5f}'.format(train_loss,train_acc),
                  'Valid Loss: {:.5f}, Valid Acc: {:.5f}'.format(valid_loss,valid_acc))
            
    # In[Save model]
    save_path = '%snet.pt' % args.dataset[-3:]
    if not osp.exists(args.model_path):
        osp.makedirs(args.model_path)
    torch.save(GNN.cpu(),osp.join(args.model_path,save_path))
            