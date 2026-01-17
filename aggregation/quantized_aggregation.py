import json
from typing import List

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

from quantization.dequantization import dequantize_model
from quantization.quantization import quantize_model

from sparsification.model_sparsifier import ModelSparsifier
from torch import nn
import copy
import torch
import numpy as np
import os
import inspect

from sia.source_inference_attack import SIA
import tenseal as ts
from utils.fl_utils import print_model_summary, get_size_of_model

class QuantizedAggregator(FedAvg):
    def __init__(self, 
                 quantization_type: str, 
                 model_type: str,
                 sparsification_mode: str,
                 sparsification_parameter: float, 
                 random_init_path: str, 
                 run_sia: bool,
                 data_type: str,
                 sia_results_path: str,
                 num_test: float,
                 data_split_seed: int,
                 apply_homomorphic_encryption: bool,
                 public_seal_context_path: str,

                 downlink_modelsize_path: str,
                 checkpoint_dir: str,

                 *args, num_clients = 3, num_rounds = 5, start_round = 0, **kwargs):
        super().__init__(*args, num_clients=num_clients, num_rounds=num_rounds, start_round=start_round, **kwargs)
        self.quantization_type = quantization_type

        self.model_type = model_type
        self.sparsification_mode = sparsification_mode
        self.sparsification_parameter = sparsification_parameter
        self.random_init_path = random_init_path

        state_dict = torch.load(random_init_path)
        self.model_sparsifier = ModelSparsifier(state_dict, sparsification_mode, sparsification_parameter)

        self.global_model = FLModel(params=state_dict, meta={})
        self.run_sia = run_sia
        self.data_type = data_type
        self.sia_results_path = sia_results_path
        self.data_split_seed = data_split_seed
        self.num_test = num_test
        if self.run_sia:
            self.sia = SIA(self.num_clients, self.data_type, self.model_type, self.num_test, self.data_split_seed)

        self.apply_homomorphic_encryption = apply_homomorphic_encryption
        self.public_seal_context_path = public_seal_context_path
        with open(public_seal_context_path, "rb") as f:
            self.public_seal_context = ts.context_from(f.read())

        self.downlink_modelsize_path = downlink_modelsize_path
        self.checkpoint_dir = checkpoint_dir

    def maybe_save_checkpoint(self, model: FLModel):
        if self.checkpoint_dir.lower() != "none":
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.checkpoint_dir, f"global_model_round_{self.current_round}.pth")
            torch.save(model, checkpoint_path)

    def aggregate(self, models: List[FLModel], aggregate_fn=None) -> FLModel:
        # Dequantize the models

        uplink_modelsizes = []
        for m in models:
            for param_name in m.params.keys():
                assert isinstance(m.params[param_name], np.ndarray), f"Expected numpy array, got {type(m.params[param_name])}"

        models = [dequantize_model(model, self.quantization_type) for model in models]
        models = [self.model_sparsifier.unsparsify_model(model, self.global_model) for model in models]

        if self.run_sia:
            sia_results = self.sia.attack(models)
            sia_results_header = "\t".join([k for k in sia_results.keys()])
            sia_results_str = "\t".join([f"{v}" for k, v in sia_results.items()])
            if not os.path.exists(self.sia_results_path):
                with open(self.sia_results_path, "w") as f:
                    f.write(sia_results_header + "\n")
            
            with open(self.sia_results_path, "a") as f:
                f.write(sia_results_str + "\n")


        for param_name in models[0].params.keys():
            param_type = models[0].params[param_name].dtype
            for m in models:
                assert m.params[param_name].dtype == param_type, f"Data type mismatch for {param_name}: {m.params[param_name].dtype} != {param_type}"

        if self.apply_homomorphic_encryption:
            for m in models:
                encrypted_params = m.meta["encrypted_weights"]
                for param_name in encrypted_params.keys():
                    assert isinstance(encrypted_params[param_name], (torch.Tensor, np.ndarray)), f"Expected torch tensor, got {type(encrypted_params[param_name])}"
                    if isinstance(encrypted_params[param_name], torch.Tensor):
                        m.params[param_name] = encrypted_params[param_name].cpu().numpy()
                    m.params[param_name] = ts.ckks_vector_from(self.public_seal_context, bytes(m.params[param_name]))

            def custom_encrypted_aggregate_fn(models):
                # perform manual aggregation
                agg_model = FLModel(params={}, meta={})
                normalization = 1 / len(models)
                encrypted_params = {}
                for param_name in models[0].params.keys():
                
                    encrypted_params[param_name] = sum([m.params[param_name] for m in models]) * normalization
                    encrypted_params[param_name] = torch.from_numpy(np.frombuffer(encrypted_params[param_name].serialize(), dtype=np.uint8))

                assert self.quantization_type == "none"
                assert self.sparsification_mode == "none"
                agg_model.params = {"dummy": np.zeros(1)}
                agg_model.meta["encrypted_weights"] = encrypted_params
                return agg_model
            
            # just ignore this. this.aggregate_fn is passed as argument
            #assert aggregate_fn is None, f"Custom aggregate function is not supported with homomorphic encryption, got function {inspect.getsource(aggregate_fn)}"

            agg_model = super().aggregate(models, custom_encrypted_aggregate_fn)

            with open(self.downlink_modelsize_path, "a") as f:
                downlink_modelsize = get_size_of_model(agg_model)
                f.write(f"{downlink_modelsize}\n")
            return agg_model
        else:
            # Aggregate the models
            agg_model = super().aggregate(models, aggregate_fn)
        
        self.global_model = copy.deepcopy(agg_model)
        self.maybe_save_checkpoint(agg_model)


        for param_name in agg_model.params.keys():
            if not isinstance(agg_model.params[param_name], np.ndarray):
                agg_model.params[param_name] = np.array(agg_model.params[param_name])

            assert isinstance(agg_model.params[param_name], np.ndarray), f"Expected numpy array, got {type(agg_model.params[param_name])}"


        for param_name in agg_model.params.keys():
            if isinstance(models[0].params[param_name], np.ndarray):
                param_type = models[0].params[param_name].dtype
            else:
                assert isinstance(models[0].params[param_name], torch.Tensor), f"Expected numpy array or torch tensor, got {type(models[0].params[param_name])}"
                param_type = models[0].params[param_name].numpy().dtype
            if not isinstance(agg_model.params[param_name], np.ndarray):
                agg_model.params[param_name] = np.array(agg_model.params[param_name])
            agg_model.params[param_name] = agg_model.params[param_name].astype(param_type)

        # Quantize the aggregated model
        agg_model = quantize_model(agg_model, self.quantization_type)
        with open(self.downlink_modelsize_path, "a") as f:
            downlink_modelsize = get_size_of_model(agg_model)
            f.write(f"{downlink_modelsize}\n")
        return agg_model