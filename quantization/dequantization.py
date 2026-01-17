import re
from typing import Union

import numpy as np
import torch
from bitsandbytes.functional import QuantState, dequantize_4bit, dequantize_blockwise

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


QUANTIZATION_TYPE = [
    "FLOAT16",
    "BLOCKWISE8",
    "FLOAT4",
    "NORMFLOAT4",
]
def dequantization(
        params: dict, quant_state: dict, quantization_type: str, source_datatype: dict
    ):
        n_params = len(params.keys())
        print(f"Running dequantization on {n_params} variables")
        n_bytes_before = 0
        n_bytes_after = 0
        n_bytes_meta = 0
        n_quant_params = 0
        for i, param_name in enumerate(params.keys()):
            source_data_type = source_datatype[param_name]

            # get the bits information
            source_date_bits = int(re.findall(r"\d+", source_data_type)[0])
            quantization_bits = int(re.findall(r"\d+", quantization_type)[0])

            # only dequantize if the quantization type is lower than the source data type
            if "int" in source_data_type:
                if source_data_type == "int64" and source_data_format == "numpy":
                    params[param_name] = params[param_name].astype(np.int64)
                elif source_data_type == "int64" and source_data_format == "torch":
                    params[param_name] = params[param_name].to(torch.int64)
                else:
                    raise ValueError(f"Invalid source data type: {source_data_type}, source_data_format: {source_data_format}")
            elif quantization_bits >= source_date_bits:
                print(
                    f"Skipping dequantization for {param_name}, quantization bit {quantization_type} >= source data bit {source_data_type}",
                )
                continue
            else:
                values = params[param_name]
                n_bytes_before += values.nbytes
                for item in quant_state[param_name].values():
                    if isinstance(item, np.ndarray) or isinstance(item, torch.Tensor):
                        n_bytes_meta += item.nbytes

                if isinstance(values, np.ndarray):
                    # if numpy, convert to torch
                    source_data_format = "numpy"
                elif isinstance(values, torch.Tensor):
                    source_data_format = "torch"
                else:
                    raise ValueError(f"Invalid source data type: {type(values)}, valid: numpy or torch")

                n_quant_params += 1
                if quantization_type == "float16":
                    # direct assign and convert back to higher precision
                    params[param_name] = values
                elif quantization_type in ["blockwise8", "float4", "normfloat4"]:
                    # use bitsandbytes to dequantize the values
                    # extract quantization state
                    if quantization_type == "blockwise8":
                        if source_data_format == "numpy":
                            # first convert numpy array to tensor if numpy
                            quantized = torch.as_tensor(values)
                            absmax = torch.as_tensor(quant_state[param_name]["absmax"])
                            code = torch.as_tensor(quant_state[param_name]["code"])
                        elif source_data_format == "torch":
                            quantized = values
                            absmax = quant_state[param_name]["absmax"]
                            code = quant_state[param_name]["code"]
                        # de-quanitze
                        dequantized = dequantize_blockwise(quantized, absmax=absmax, code=code)
                    else:
                        if source_data_format == "numpy":
                            # first convert numpy array to tensor, need to use GPU
                            quantized = torch.as_tensor(values).cuda()
                            # create QuantState object
                            quantize_state = QuantState(
                                quant_type=quant_state[param_name]["quant_type"],
                                absmax=torch.as_tensor(quant_state[param_name]["absmax"]).cuda(),
                                blocksize=quant_state[param_name]["blocksize"],
                                code=torch.as_tensor(quant_state[param_name]["quant_map"]).cuda(),
                                dtype=getattr(torch, quant_state[param_name]["dtype"]),
                                shape=torch.Size(quant_state[param_name]["shape"]),
                            )
                        elif source_data_format == "torch":
                            quantized = values.cuda()
                            quantize_state = QuantState(
                                quant_type=quant_state[param_name]["quant_type"],
                                absmax=torch.as_tensor(quant_state[param_name]["absmax"]).cuda(),
                                blocksize=quant_state[param_name]["blocksize"],
                                code=torch.as_tensor(quant_state[param_name]["quant_map"]).cuda(),
                                dtype=getattr(torch, quant_state[param_name]["dtype"]),
                                shape=torch.Size(quant_state[param_name]["shape"]),
                            )
                        # de-quanitze
                        if quantization_type == "float4":
                            dequantized = dequantize_4bit(quantized, quantize_state, quant_type="fp4")
                        else:
                            dequantized = dequantize_4bit(quantized, quantize_state, quant_type="nf4")
                    if source_data_format == "numpy":
                        params[param_name] = dequantized.cpu().numpy()
                    elif source_data_format == "torch":
                        params[param_name] = dequantized.cpu()

                # assign back
                if source_data_format == "numpy":
                    # convert back to original data type
                    if source_data_type == "float32":
                        params[param_name] = params[param_name].astype(np.float32)
                    elif source_data_type == "float64":
                        params[param_name] = params[param_name].astype(np.float64)
                    elif source_data_type == "float16":
                        params[param_name] = params[param_name].astype(np.float16)
                elif source_data_format == "torch":
                    # convert back to original data type
                    if source_data_type == "float32":
                        params[param_name] = params[param_name].float()
                    elif source_data_type == "float64":
                        params[param_name] = params[param_name].double()
                    elif source_data_type == "float16":
                        params[param_name] = params[param_name].half()
                    elif source_data_type == "bfloat16":
                        params[param_name] = params[param_name].bfloat16()

            n_bytes_after += params[param_name].nbytes

        print(
            f"Dequantized {n_quant_params}/{n_params} params."
            f" Before dequantization: {n_bytes_before / (1024 ** 2):.2f} MB with meta: {n_bytes_meta / (1024 ** 2):.2f} MB."
            f" After dequantization: {n_bytes_after / (1024 ** 2):.2f} MB.",
        )
        return params

from nvflare.app_common.abstract.fl_model import FLModel
def dequantize_model(model: FLModel, quantization_type: str) -> FLModel:
    if quantization_type == "none":
        return model
    quant_state = model.meta.pop("quant_state")
    source_datatype = model.meta.pop("source_datatype")
    model.params = dequantization(model.params, quant_state, quantization_type, source_datatype)
    return model