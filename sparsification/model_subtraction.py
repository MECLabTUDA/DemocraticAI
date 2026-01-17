
import torch
import torch.nn as nn


def subtract_model(global_model: nn.Module, local_model: nn.Module) -> nn.Module:
    # local -= global
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        local_param.data -= global_param.data
    return local_model


def add_model(global_model: nn.Module, local_model: nn.Module) -> nn.Module:
    # local += global
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        local_param.data += global_param.data
    return local_model