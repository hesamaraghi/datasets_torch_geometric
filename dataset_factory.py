from .nmnist import NMNIST
from .ncaltech101 import NCALTECH101, NCALTECH101InMemory
from .nasl import NASL
from .ncars import NCARS, NCARSInMemory
from .fan1vs3 import FAN1VS3, FAN1VS3InMemory
from .dvsgesture import DVSGESTURE
from .dvsgesture_tonic import DVSGESTURE_TONIC, DVSGESTURE_TONICInMemory
import os


def create_dataset(dataset_path=None, 
                   dataset_name=None, 
                   dataset_type=None,
                   transform=None, 
                   pre_transform=None,
                   in_memory=False,
                   num_workers=4):
    
    # Define a dictionary mapping dataset names to their corresponding classes
    dataset_classes = {
        "NMNIST": NMNIST,
        "NCALTECH101": NCALTECH101,
        "NASL": NASL,
        "NCARS": NCARS,
        "FAN1VS3": FAN1VS3,
        "DVSGESTURE": DVSGESTURE,
        "DVSGESTURE_TONIC": DVSGESTURE_TONIC
        # Add more dataset options here as needed
    }
    
    # Define a dictionary for in-memory versions if they exist
    in_memory_dataset_classes = {
        "FAN1VS3": FAN1VS3InMemory,
        "DVSGESTURE_TONIC": DVSGESTURE_TONICInMemory,
        "NCARS": NCARSInMemory,
        "NCALTECH101": NCALTECH101InMemory
        # Add more in-memory dataset options here as needed
    }
    
    # Check if the dataset_name is valid and get the corresponding dataset class
    if dataset_name in dataset_classes:
        if in_memory and dataset_name in in_memory_dataset_classes:
            dataset_class = in_memory_dataset_classes[dataset_name]
        else:
            dataset_class = dataset_classes[dataset_name]
        return dataset_class(root=dataset_path, 
                             name=dataset_type, 
                             transform=transform,
                             pre_transform=pre_transform,
                             num_workers=num_workers)
    else:
        raise ValueError("Invalid dataset name")
