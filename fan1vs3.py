import os.path as osp
import numpy as np
import torch
from torch_geometric.data import  Data
from .base_dataset import BaseDataset, BaseInMemoryDataset
from event_utils.lib.data_formats.read_events import read_h5_events_dict

dir_path = osp.dirname(osp.realpath(__file__))
dataset_path = osp.join(dir_path,'FAN1VS3','data')

def select_base_class(BaseClass = BaseDataset):
    
    class DerivedClass(BaseClass):
        
        def __init__(self, root=dataset_path, name='all', transform=None,
                pre_transform=None, pre_filter=None, num_workers=4):
            if root is None:
                root = dataset_path
            super().__init__(root, name, transform, pre_transform, pre_filter, num_workers)

        def read_events(self,filename):
            
            
            events = read_h5_events_dict(filename, read_frames=False)

            data_x = events['xs'].astype(np.float32)
            data_y = events['ys'].astype(np.float32)
            data_ts = events['ts'].astype(np.float32)    
            data_p = events['ps'].astype(np.float32)
    
            pos = np.array([data_x,data_y,data_ts])
            pos = torch.from_numpy(pos)
            pos = pos.transpose(0,1)
            data_p = np.expand_dims(data_p, axis=1) 
            data_p = torch.from_numpy(data_p)
            data = Data(x=data_p,pos=pos)
            return data

    return DerivedClass

FAN1VS3 = select_base_class(BaseClass = BaseDataset)
FAN1VS3InMemory = select_base_class(BaseClass = BaseInMemoryDataset)


if __name__ == '__main__':
    dataset  = FAN1VS3(dataset_path, transform = None)
    dataset_in_memory = FAN1VS3InMemory(dataset_path, transform = None)
    print("Good bye!")

