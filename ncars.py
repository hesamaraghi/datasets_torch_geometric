
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import  Data


try:
    from .NCARS.src.io.psee_loader import PSEELoader
except ModuleNotFoundError:
    from NCARS.src.io.psee_loader import PSEELoader

try:
    from .base_dataset import BaseDataset, BaseInMemoryDataset
except ModuleNotFoundError:
    from base_dataset import BaseDataset, BaseInMemoryDataset

dir_path = osp.dirname(osp.realpath(__file__))
dataset_path = osp.join(dir_path,'NCARS','data')

def select_base_class(BaseClass = BaseDataset):
       
    class DerivedClass(BaseClass):

        def __init__(self, root=dataset_path, name='all', transform=None,
                pre_transform=None, pre_filter=None, num_workers=4):
            if root is None:
                root = dataset_path
            super().__init__(root, name, transform, pre_transform, pre_filter, num_workers)


        def read_events(self,filename):
            """"Reads in the TD events contained in the NASL dataset file specified by 'filename'

                Python: import scipy.io as sio
                    sio.loadmat(filename)

                Each sample contains x, y addresses(x, y), timestamp(ts), polarity(pol).
                    x: 0-239    y: 0-179  (dtype=uint8)
                    ts: in microsecond  (dtype=int32)
                    pol: 1 or 0 (1 means ON polarity, while 0 means OFF polarity.) (dtype=uint8)"""
            video = PSEELoader(filename)
            events = video.load_n_events(video.event_count()) 
            data_x = events["x"].reshape(-1,1).astype(np.float32)
            data_y = events["y"].reshape(-1,1).astype(np.float32)
            data_ts = events["t"].reshape(-1,1).astype(np.float32)
            data_p = events["p"].reshape(-1,1).astype(np.float32) * 2 - 1.0
            pos = np.concatenate([data_x,data_y,data_ts], axis=1)
            pos = torch.from_numpy(pos)
            data_p = torch.from_numpy(data_p)
            data = Data(x=data_p,pos=pos)
            return data

    return DerivedClass

NCARS = select_base_class(BaseClass = BaseDataset)
NCARSInMemory = select_base_class(BaseClass = BaseInMemoryDataset)

if __name__ == '__main__':
    dataset  = NCARS(dataset_path, transform = None)
    dataset_in_memory = NCARSInMemory(dataset_path, transform = None)
    print("Good bye!")