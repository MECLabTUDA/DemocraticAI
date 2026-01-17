import torch
import torch.nn as nn
import random
import numpy as np
import copy

class Malfunction:
    def __init__(self, scale=None):
        if scale is None:
            self.scale = 100.5
        else:
            self.scale = scale

    def _ana(self, update):
        malf_update = copy.deepcopy(update)
        for key in update:
            dtype = update[key].dtype
            noise       = torch.randn(update[key].size()) * self.scale / 100.0 * malf_update[key]
            malf_update[key] = (noise + malf_update[key].float()).type(dtype)
        return malf_update

    def _sfa(self, update):
        malf_update = copy.deepcopy(update)
        scale = -1
        for key in update:
            dtype = update[key].dtype
            malf_update[key] = (scale * malf_update[key].float()).type(dtype)
        return malf_update


    def _corrupt(self, update, malfunction):
        if malfunction == 'ana':
            return self._ana(update)
        elif malfunction == 'sfa':
            return self._sfa(update)


    def __call__(self, update, malfunction):
        corrupted_update = self._corrupt(update, malfunction)
        return corrupted_update
