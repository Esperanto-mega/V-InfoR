# In[Import]
import random
import torch
import numpy as np

# In[Seed]
def GlobalSeed(seed,opt = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = opt
    torch.backends.cudnn.deterministic = not opt
