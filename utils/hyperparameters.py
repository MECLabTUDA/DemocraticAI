import torch
import torch.nn as nn
import torch.optim

from network.crc import CTCViT4SGD


def scenario_vars(data_type):
    if data_type == 'fetalAbdominal':
        num_clients = 5
        data_split_seed = 42
        num_test = 0.7
        num_rounds  = 200
    elif data_type == 'XRayMimic':
        num_clients = 5
        data_split_seed = 42
        num_test = 0.5
        num_rounds  = 200
    elif data_type == 'XRayMimic200':
        num_clients = 5
        data_split_seed = 42
        num_test = 0.25
        num_rounds  = 200
    elif data_type == 'crc':
        num_clients = 5
        data_split_seed = 42
        num_test = 1.0
        num_rounds  = 25
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    return num_clients, data_split_seed, num_test, num_rounds

def get_batch_size(data_type) -> int:
    return {'fetalAbdominal': 4, 'XRayMimic': 4, 'XRayMimic200': 4, 'crc': 4}[data_type]

def get_local_epochs(data_type) -> int:
    return 3 if data_type == 'fetalAbdominal' else 1

def is_multiclass_classification(data_type) -> bool:
    return data_type in ['crc']

def get_optimizer(data_type: str, network: nn.Module) -> torch.optim.Optimizer:
    if isinstance(network, CTCViT4SGD):
        return torch.optim.SGD(network.parameters(), lr=0.001, weight_decay=0)
    elif data_type == 'isic19':
        return torch.optim.Adam(network.parameters(), 5e-4)
    else:
        return torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=0)
    
def is_classification_task(data_type: str) -> bool:
    return data_type in ["crc"]