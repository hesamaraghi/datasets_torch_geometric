import os.path as osp
import numpy as np
import torch
from torch_geometric.data import  Data
try:
    from .base_dataset import BaseDataset, BaseInMemoryDataset
except:
    from base_dataset import BaseDataset, BaseInMemoryDataset
    
from tonic.cached_dataset import load_from_disk_cache

dir_path = osp.dirname(osp.realpath(__file__))
dataset_path = osp.join(dir_path,'DVSGESTURE_TONIC','data')

def select_base_class(BaseClass = BaseDataset):
    
    class DerivedClass(BaseClass):
    
        def __init__(self, root=dataset_path, name='all', transform=None,
                pre_transform=None, pre_filter=None, num_workers=4):
            if root is None:
                root = dataset_path
            super().__init__(root, name, transform, pre_transform, pre_filter, num_workers)

        def read_events(self,filename):
            
            events, _ = load_from_disk_cache(filename)
            
            data_x = events['x'].astype(np.float32)
            data_y = events['y'].astype(np.float32)
            data_ts = events['t'].astype(np.float32)
            data_p = events['p'].astype(np.float32) * 2 - 1.0

            pos = np.array([data_x,data_y,data_ts])
            pos = torch.from_numpy(pos)
            pos = pos.transpose(0,1)
            data_p = np.expand_dims(data_p, axis=1) 
            data_p = torch.from_numpy(data_p)
            data = Data(x=data_p,pos=pos)
            return data

    return DerivedClass

DVSGESTURE_TONIC = select_base_class(BaseClass = BaseDataset)
DVSGESTURE_TONICInMemory = select_base_class(BaseClass = BaseInMemoryDataset)

if __name__ == '__main__':
    dataset  = DVSGESTURE_TONIC(dataset_path, transform = None)
    dataset_in_memory = DVSGESTURE_TONICInMemory(dataset_path, transform = None)
    print("Good bye!")

