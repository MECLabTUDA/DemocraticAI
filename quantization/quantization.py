import re
from typing import Union

import numpy as np
import torch
from bitsandbytes.functional import quantize_4bit, quantize_blockwise

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

DATA_TYPE = [
    "FLOAT64",
    "FLOAT32",
    "FLOAT16",
    "BFLOAT16",
]

QUANTIZATION_TYPE = [
    "FLOAT16",
    "BLOCKWISE8",
    "FLOAT4",
    "NORMFLOAT4",
]


NP_FP16_MIN = np.finfo(np.float16).min
NP_FP16_MAX = np.finfo(np.float16).max
TS_FP16_MIN = torch.finfo(torch.float16).min
TS_FP16_MAX = torch.finfo(torch.float16).max
def quantization(params: dict, quantization_type: str):
    n_params = len(params.keys())
    print(f"Running quantization on {n_params} variables")
    n_bytes_before = 0
    n_bytes_after = 0
    n_bytes_meta = 0
    n_quant_params = 0
    quant_state = {}
    source_datatype = {}
    for i, param_name in enumerate(params.keys()):
        values = params[param_name]
        quant_state[param_name] = {}

        # check the data type, numpy or torch
        # otherwise error
        if isinstance(values, np.ndarray):
            # if numpy, convert to torch
            source_data_format = "numpy"
        elif isinstance(values, torch.Tensor):
            source_data_format = "torch"
        else:
            raise ValueError(f"Invalid source data type of {param_name}: {type(values)}, valid: numpy or torch")

        # get the data type of the values
        if source_data_format == "numpy":
            source_data_type = values.dtype.name
        elif source_data_format == "torch":
            source_data_type = str(values.dtype).split(".")[1]
        source_datatype[param_name] = source_data_type

        skip = False
        # check if the data type is valid
        if source_data_type.upper() not in DATA_TYPE:
            skip = True
            #raise ValueError(f"Invalid source data type of {param_name}: {source_data_type}, valid: {DATA_TYPE}")

        # get the bits information
        source_data_bits = int(re.findall(r"\d+", source_data_type)[0])
        quantization_bits = int(re.findall(r"\d+", quantization_type)[0])

        # add the number of bytes of the values
        n_bytes_before += values.nbytes
        # only quantize if the quantization type is lower than the source data type
        if quantization_bits >= source_data_bits or skip:
            print(
                f"Skipping quantization for {param_name}, quantization bit {quantization_type} >= source data bit {source_data_type}",
            )
            continue
        else:
            n_quant_params += 1
            if quantization_type == "float16":
                if source_data_format == "numpy":
                    # first clamp the values to the range of float16
                    values = np.clip(values, NP_FP16_MIN, NP_FP16_MAX)
                    # then convert to float16
                    values = values.astype(np.float16)
                elif source_data_format == "torch":
                    # first clamp the values to the range of float16
                    values = torch.clamp(values, TS_FP16_MIN, TS_FP16_MAX)
                    # then convert to float16
                    values = values.to(torch.float16)
                params[param_name] = values
            elif quantization_type in ["blockwise8", "float4", "normfloat4"]:
                # use bitsandbytes to quantize the values
                # input is a tensor, output is a tuple of (quantized tensor, quantized_state)
                if quantization_type == "blockwise8":
                    if source_data_format == "numpy":
                        # if numpy, first convert numpy array to tensor
                        values_tensor = torch.as_tensor(values)
                    elif source_data_format == "torch":
                        values_tensor = values

                    # then quantize the tensor
                    quantized, quantized_state = quantize_blockwise(values_tensor)
                    # add the quantization state and values, keep source data format
                    if source_data_format == "numpy":
                        quant_state[param_name]["absmax"] = quantized_state.absmax.numpy()
                        quant_state[param_name]["code"] = quantized_state.code.numpy()
                        values = quantized.numpy()
                    elif source_data_format == "torch":
                        quant_state[param_name]["absmax"] = quantized_state.absmax
                        quant_state[param_name]["code"] = quantized_state.code
                        values = quantized
                    n_bytes_meta += quant_state[param_name]["absmax"].nbytes
                    n_bytes_meta += quant_state[param_name]["code"].nbytes
                else:
                    if source_data_format == "numpy":
                        # if numpy, first convert numpy array to tensor, need to use GPU
                        values_tensor = torch.as_tensor(values).cuda()
                    elif source_data_format == "torch":
                        # if torch, directly use the tensor, need to use GPU
                        values_tensor = values.cuda()
                    # then quantize the tensor
                    if quantization_type == "float4":
                        quantized, quantized_state = quantize_4bit(values_tensor, quant_type="fp4")
                    else:
                        quantized, quantized_state = quantize_4bit(values_tensor, quant_type="nf4")
                    # add the quantization state and values, keep source data format
                    quantized_state = quantized_state.as_dict()

                    for state_name, state in quantized_state.items():
                        if isinstance(state, torch.Tensor):
                            if source_data_format == "numpy":
                                # if the state is a tensor, convert it to numpy array
                                quant_state[param_name][state_name] = state.cpu().numpy()
                            elif source_data_format == "torch":
                                # if the state is a tensor, keep it as tensor
                                quant_state[param_name][state_name] = state.cpu()
                            n_bytes_meta += state.nbytes
                        else:
                            quant_state[param_name][state_name] = state
                    # add values
                    if source_data_format == "numpy":
                        values = quantized.cpu().numpy()
                    elif source_data_format == "torch":
                        values = quantized.cpu()
                params[param_name] = values
            n_bytes_after += params[param_name].nbytes
            #self.log_info(fl_ctx, f"Quantized {param_name} with {source_data_type} to {self.quantization_type}")

    print(
        f"Quantized {n_quant_params}/{n_params} params."
        f" Before quantization: {n_bytes_before / (1024 ** 2):.2f} MB."
        f" After quantization: {n_bytes_after / (1024 ** 2):.2f} MB with meta: {n_bytes_meta / (1024 ** 2):.2f} MB.",
    )
    return params, quant_state, source_datatype

from nvflare.app_common.abstract.fl_model import FLModel
def quantize_model(model: FLModel, quantization_type: str) -> FLModel:
    if quantization_type == "none":
        return model
    quantized_params, quant_state, source_datatype = quantization(model.params, quantization_type)
    model.params = quantized_params
    assert not "quant_state" in model.meta.keys()
    assert not "source_datatype" in model.meta.keys()
    model.meta["quant_state"] = quant_state
    model.meta["source_datatype"] = source_datatype
    return model
