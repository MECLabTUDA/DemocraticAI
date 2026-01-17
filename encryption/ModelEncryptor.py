import tenseal as ts
from nvflare.app_common.abstract.fl_model import FLModel

import numpy as np
import torch
import tqdm, time


class ModelEncryptor:
    """
    Encrypts and decrypts FLModel parameters using homomorphic encryption.
    Now stacks all model weights into a single encrypted vector for efficiency.
    """

    def __init__(self, apply_homomorphic_encryption: bool, private_seal_context: str, enable_loadingbar: bool = False):
        self.apply_homomorphic_encryption = apply_homomorphic_encryption
        self.enable_loadingbar = enable_loadingbar
        
        with open(private_seal_context, "rb") as f:
            self.context = ts.context_from(f.read())
        self.poly_modulus_degree = 32768 #for some reason this cannot be read from the context, so we have to hard code
        #self.poly_modulus_degree = 2* 1_000_000 #for some reason this cannot be read from the context, so we have to hard code

        self.shapes = None
        self.sizes = None

    def encrypt(self, model: FLModel) -> FLModel:
        if not self.apply_homomorphic_encryption:
            return model

        params = model.params
        if self.shapes is None:
            self.shapes = {k: v.shape for k, v in params.items()}
            self.sizes = {k: v.numel() for k, v in params.items()}


        # Flatten and stack all weights
        flat_weights = torch.cat([v.view(-1) for v in params.values()])

        patch_size = self.poly_modulus_degree // 2

        encrypted_tensors = {}
        for i in tqdm.trange(0, flat_weights.size(0), patch_size, disable=not self.enable_loadingbar):
            patch = flat_weights[i:i + patch_size]
            encrypted_vector = ts.ckks_vector(self.context, patch.numpy())
            serialized = encrypted_vector.serialize()
            encrypted_tensor = torch.from_numpy(np.frombuffer(serialized, dtype=np.uint8))
            encrypted_tensors[i] = (encrypted_tensor)

        model.params = {"dummy": np.zeros(1)}
        model.meta["encrypted_weights"] = encrypted_tensors
        #model.meta["shapes"] = self.shapes
        #model.meta["sizes"] = self.sizes

        return model

    def decrypt(self, model: FLModel) -> FLModel:
        if not self.apply_homomorphic_encryption:
            return model

        encrypted_tensor = model.meta["encrypted_weights"]
        assert isinstance(encrypted_tensor, dict)
        shapes = self.shapes
        sizes = self.sizes

        patch_size = self.poly_modulus_degree // 2
        
        decrypted_vector =[]
        for i in tqdm.trange(0, sum(self.sizes.values()), patch_size, disable=not self.enable_loadingbar):
            assert isinstance(encrypted_tensor[i], torch.Tensor), f"got {type(encrypted_tensor[i])}"
            _bytes = encrypted_tensor[i].cpu().numpy().tobytes()
            encrypted_vector = ts.ckks_vector_from(self.context, _bytes)
            decrypted_values = np.array(encrypted_vector.decrypt())
            decrypted_vector.append(decrypted_values)


        decrypted_vector = np.concatenate(decrypted_vector)

        # Recover original tensors
        params = {}
        offset = 0
        for k in shapes:
            size = sizes[k]
            flat_vals = decrypted_vector[offset:offset + size]
            params[k] = torch.from_numpy(flat_vals).reshape(shapes[k])
            offset += size

        model.params = params
        return model
