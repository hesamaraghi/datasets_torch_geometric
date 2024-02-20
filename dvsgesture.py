import os.path as osp
import numpy as np
import torch
from torch_geometric.data import  Data
from .base_dataset import BaseDataset

dir_path = osp.dirname(osp.realpath(__file__))
dataset_path = osp.join(dir_path,'DVSGESTURE','data')

class DVSGESTURE(BaseDataset):
    
    def __init__(self, root=dataset_path, name='all', transform=None,
            pre_transform=None, pre_filter=None, num_workers=4):
        if root is None:
            root = dataset_path
        super().__init__(root, name, transform, pre_transform, pre_filter, num_workers)

    def read_events(self,filename):
        
        with np.load(filename) as events:
            data_x = events['x'].astype(np.float32)
            data_y = events['y'].astype(np.float32)
            data_ts = events['t'].astype(np.float32)
            data_ts = data_ts - data_ts.min()
            data_p = events['p'].astype(np.float32) * 2 - 1.0

  
        pos = np.array([data_x,data_y,data_ts])
        pos = torch.from_numpy(pos)
        pos = pos.transpose(0,1)
        data_p = np.expand_dims(data_p, axis=1) 
        data_p = torch.from_numpy(data_p)
        data = Data(x=data_p,pos=pos)
        return data




if __name__ == '__main__':
    dataset  = DVSGESTURE(dataset_path, transform = None)
    print("Good bye!")

