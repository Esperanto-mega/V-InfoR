# In[Import]
import os
import pickle
'''DGL & PyG.'''
import dgl
import torch
from torch_geometric.utils.convert import to_networkx

# In[PyG2DGL]
def pyg2dgl(graph = None, dataset = None, info = False,
            savepath = r'./', filename = 'adv.pickle',mode = 'wb'):
    '''
    Parameters
    ----------
    graph : graph to be convert.
    dataset : dataset to be convert.
    
    ********************************************************
    User can choose convert single graph or whole dataset, 
    even several datasets.
    ********************************************************
    
    info : Whether to print prompt.
    savepath : Directory to save pickle file.
    filename : Name of pickle file.
    mode : Mode of pickle.dump().
    
    Returns
    -------
    None.
    '''
    
    assert graph is not None or dataset is not None
    if type(dataset) != list:
        DATASET = [dataset]
    else:
        DATASET = dataset
    total = 0
    DGL_G = []
    count = 0
    for data_set in DATASET:
        total = total + len(data_set)
        for g in iter(data_set):
            nx_g = to_networkx(g)
            dgl_g = dgl.from_networkx(nx_g)
            dgl_g.ndata['x'] = g.x
            dgl_g.edata['x'] = g.edge_attr
            dgl_g.ndata['y'] = torch.full(g.x.shape, g.y.item())
            
            if 'name' in dir(g):
                dgl_g.ndata['name'] = torch.full(g.x.shape, id(g.name))
            if 'GTMask' in dir(g):
                dgl_g.edata['GTMask'] = g.GTMask
                
            DGL_G.append(dgl_g)
            count = count + 1
            if info:
                print('finish/total:',count,'/',total)

    assert len(DGL_G) == total
    pickle.dump(DGL_G,open(os.path.join(savepath,filename),'wb'))
    print('Successful.')

# In[Collate]
def collate(samples, add_selfloops = True):
    '''Used to create DGL dataloaders.'''
    graphs, labels = map(list, zip(*samples))
    if add_selfloops:
        graphs = [dgl.add_self_loop(graph) for graph in graphs]
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)