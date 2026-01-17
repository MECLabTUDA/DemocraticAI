
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
import json
import numpy as np
import torch

def print_model_summary(model: FLModel):
    """Prints the summary of the model."""

    def traverse_node(d: dict) -> dict:
        ret_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                ret_dict[key] = traverse_node(value)
            else:
                ret_dict[key] = str(type(value))
        return ret_dict

    model_dict = dict()
    for key in model.summary().keys():
        a = getattr(model, key)
        model_dict[key] = a

    summary_dict = traverse_node(model_dict)
    print("Model Summary:", json.dumps(summary_dict, indent=4))

def get_size_of_model(model: FLModel) -> int:
    """Returns the size of the model and meta etc. in bytes."""

    keys_to_ignore ={
        "NUM_STEPS_CURRENT_ROUND": int,
        "CLIENT_NAME": str,
        "client_name": str,
        "params_type": ParamsType,
        "nr_aggregated": int,
        "current_round": int,
        "quant_type": str,
        "blocksize": int,
        "dtype": str,
        "shape": tuple,
        "source_datatype": dict,
    }
    def size_of_dict(d):
        error_keys = []
        size = 0
        for key, value in d.items():
            if key in keys_to_ignore and isinstance(value, keys_to_ignore[key]):
                continue
            elif isinstance(value, np.ndarray):
                size += value.nbytes
            elif isinstance(value, torch.Tensor):
                size += value.element_size() * value.numel()
            elif isinstance(value, dict):
                size += size_of_dict(value)
            else:
                error_keys.append(key)
        if error_keys:
            raise ValueError(f"Unknown type for keys {error_keys}: {[type(d[key]) for key in error_keys]}")
        return size
    
    model_dict = dict()
    for key in model.summary().keys():
        a = getattr(model, key)
        model_dict[key] = a

    size = size_of_dict(model_dict)
    return size