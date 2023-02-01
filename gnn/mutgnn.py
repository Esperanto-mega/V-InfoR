import sys
import time
import random
import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch.nn import Sequential as Seq, ReLU, Linear as Lin, Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, BatchNorm, global_mean_pool

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# if '..' not in sys.path:
#     sys.path.append('..')
    
from gnn.overload import overload
from utils.seed import GlobalSeed
from utils.train import TrainProxy
from utils.train import TestProxy
from datasets.datasetmut import Mutagenicity


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mutag Model")

    parser.add_argument('--data_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'data', 'MUTAG'),
                        help='Input data path.')
    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'param', 'gnns'),
                        help='path for saving trained model.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='GPU device.')
    parser.add_argument('--epoch', type=int, default=300, 
                        help='Number of epoch.')
    parser.add_argument('--lr', type=float, default= 1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    parser.add_argument('--num_unit', type=int, default=2,
                        help='number of Convolution layers(units)')
    parser.add_argument('--random_label', type=bool, default=False,
                        help='train a model under label randomization for sanity check')

    return parser.parse_args()


class MutagNet(torch.nn.Module):

    def __init__(self, conv_unit=2):
        super().__init__()

        self.node_emb = Lin(14, 32)
        self.edge_emb = Lin(3, 32)
        self.relu_nn = ModuleList([ReLU() for i in range(conv_unit)])

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        for i in range(conv_unit):
            conv = GINEConv(nn=Seq(Lin(32, 75), self.relu_nn[i], Lin(75, 32)))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(32))
            self.relus.append(ReLU())

        self.lin1 = Lin(32, 16)
        self.relu = ReLU()
        self.lin2 = Lin(16, 2)
        self.softmax = Softmax(dim=1)

    @overload
    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.NodeRep(x, edge_index, edge_attr, batch)
        # print('node_x.shape:',node_x.shape)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_pred(graph_x)

    @overload
    def NodeRep(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for conv, batch_norm, ReLU in \
                zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_attr)
            x = ReLU(batch_norm(x))
        node_x = x
        return node_x
        
    @overload
    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.NodeRep(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_pred(self, graph_x):
        pred = self.relu(self.lin1(graph_x))
        pred = self.lin2(pred)
        self.readout = self.softmax(pred)
        return pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)