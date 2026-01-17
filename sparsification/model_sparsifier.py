
from typing import OrderedDict
import numpy as np
import torch, collections

from nvflare.app_common.abstract.fl_model import FLModel
class ModelSparsifier:
    def __init__(self, params: OrderedDict, sparsification_mode: str, sparsification_parameter: float):
        assert isinstance(params, OrderedDict), f"Expected OrderedDict, got {type(params)}"
        self.param_names = list(params.keys())
        self.params_shape = {param_name: params[param_name].shape for param_name in self.param_names}

        self.num_params = sum([params[param_name].numel() for param_name in self.param_names])

        self.sparsification_mode = sparsification_mode
        self.sparsification_parameter = sparsification_parameter

    def get_all_parameters(self, model: FLModel) -> np.ndarray:
        state_dict = model.params
        all_consecutive_parameters = [state_dict[param_name].flatten() for param_name in self.param_names]
        all_consecutive_parameters = torch.cat(all_consecutive_parameters).numpy()
        return all_consecutive_parameters

    def sparsify_model(self, local_model: FLModel, global_model: FLModel) -> FLModel:
        if self.sparsification_mode == "none":
            return local_model

        all_local_parameters = self.get_all_parameters(local_model)
        all_global_parameters = self.get_all_parameters(global_model)
        assert all_local_parameters.shape == all_global_parameters.shape
        assert len(all_local_parameters) == self.num_params

        param_diff = all_local_parameters - all_global_parameters

        print(f"Sparsifying {all_local_parameters.size} parameters with {self.sparsification_mode} with parameter {self.sparsification_parameter}")

        if self.sparsification_mode == "top-k":
            num_to_sparse = int(self.sparsification_parameter * len(param_diff))
            sparse_indices = np.argpartition(np.abs(param_diff), -num_to_sparse)[-num_to_sparse:]
        elif self.sparsification_mode == "threshold":
            sparse_indices = np.argwhere(np.abs(param_diff) > self.sparsification_parameter).ravel()
        elif self.sparsification_mode == "random":
            num_to_sparse = int(self.sparsification_parameter * len(param_diff))
            sparse_indices = np.random.choice(len(param_diff), num_to_sparse, replace=False)
        else:
            raise ValueError(f"Unknown sparsification mode: {self.sparsification_mode}")    
            
        sparse_params = all_local_parameters[sparse_indices]

        local_model.params = {"sparse_params": torch.from_numpy(sparse_params)}
        local_model.meta["sparsification_indices"] = sparse_indices
    
        print(f"Sparsified {all_local_parameters.size} parameters to {sparse_params.size} parameters."
              f" Before sparsification: {all_local_parameters.size * 4 / (1024 ** 2):.2f} MB."
              f" After sparsification: {sparse_params.size * 4 / (1024 ** 2):.2f} MB with meta: {sparse_indices.size * 4 / (1024**2):.2f} MB.")

        return local_model


    def unsparsify_model(self, local_model: FLModel, global_model: FLModel) -> FLModel:
        if self.sparsification_mode == "none":
            return local_model

        sparse_indices = local_model.meta.pop("sparsification_indices")

        unsparse_params = self.get_all_parameters(global_model)
        unsparse_params[sparse_indices] = local_model.params['sparse_params']


        local_model.params = collections.OrderedDict()
        i = 0
        for param_name in self.param_names:
            param_shape = self.params_shape[param_name]
            param_size = int(np.prod(param_shape))
            local_model.params[param_name] = torch.from_numpy(unsparse_params[i:i + param_size].reshape(param_shape))
            i += param_size



        return local_model