# In[Import]
from functools import wraps

# In[Overload]
def overload(func):
    @wraps(func)
    def wrapper(*args,**kargs):
        if len(args) == 2:
            # For input like model(g).
            g = args[1]
            return func(args[0],
                        g.x,
                        g.edge_index,
                        g.edge_attr)
        elif len(args) == 4:
            # For input like model(x,edge_index,edge_attr).
            return func(*args)
        else:
            raise TypeError
    return wrapper