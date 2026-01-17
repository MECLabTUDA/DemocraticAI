import torch
from torch import nn

import einops 
import tqdm
import os
import losses.dice as dice

from network.MedNCA import MedNCA

from network.model import get_model
import skimage.measure
import numpy as np

import wandb
from utils import hyperparameters as hyp
import torch.nn.functional as F


class Trainer:

    def __init__(self, data_loader, data_type, model_type:str, client_name: str):
        self.device  = 'cuda:0'
        
        self.model = get_model(data_type, model_type, self.device)

        self.optimizer = hyp.get_optimizer(data_type, self.model)

        print(f'optimizer: {self.optimizer.__class__.__name__}')

        self.ldr     = data_loader
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss = dice.DiceLoss()
        self.client_name = client_name
        self.data_type = data_type

    def get_bboxes(self, label):
        gt = einops.rearrange(label, 'b 1 h w-> b h w').cpu().numpy().astype(int)
        bbox_mask = np.zeros_like(gt)
        for b in range(gt.shape[0]):
            rp = skimage.measure.regionprops(gt[b])
            assert len(rp) == 1
            rp = rp[0]
            bbox_mask[b, rp.bbox[0]:rp.bbox[2], rp.bbox[1]:rp.bbox[3]] = 1
        
        bbox_mask = einops.rearrange(bbox_mask, 'b h w-> b 1 h w')
        return torch.from_numpy(bbox_mask).to(self.device)

    def train_epoch(self, supervision_scenario="full", loading_bar=False):
        assert supervision_scenario in ["full", "weak", "none"]

        mean_loss = 0
        mean_dice = 0

        self.model.to(self.device)
        self.model.train()
        for img, label in tqdm.tqdm(self.ldr, disable=not loading_bar):
            assert isinstance(img, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            img, label = img.to(self.device, torch.float32), label.to(self.device, torch.float32)
            if isinstance(self.model, MedNCA):
                pred, _ = self.model(img, label)
                pred = einops.rearrange(pred, 'b h w c-> b c h w')
            else:
                pred       = self.model(img)

            if supervision_scenario == "none":
                with torch.no_grad():
                    pred2, _ = self.model(img, label)
                    pred2 = einops.rearrange(pred2, 'b h w c-> b c h w')
                    pesudo_label = torch.logical_and(pred > 0, pred2 > 0)
                    label = pesudo_label
            elif supervision_scenario == "weak":
                bbox_mask = self.get_bboxes(label)
                label = bbox_mask * pred
                label = (label > 0).float()
                

            if hyp.is_multiclass_classification(self.data_type):
                bce_loss = F.cross_entropy(pred, label.long())
                loss = bce_loss
                with torch.no_grad():
                    preds = torch.argmax(pred, dim=1)
                    correct = (preds == label).float()
                    dice_loss = correct.mean()
            else:
                bce_loss = self.loss_fn(pred, label.float())
                dice_loss = self.dice_loss.compute_dice(torch.sigmoid(pred), label)
                loss = bce_loss #- dice_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            mean_loss += loss.item()
            mean_dice += dice_loss.item()
            del bce_loss, dice_loss, loss, pred, img, label

        print(f'{self.client_name} trained epoch, mean loss: {mean_loss / len(self.ldr)}, dice: {mean_dice / len(self.ldr)}')
        wandb.log({
            f'{self.client_name}/loss': mean_loss / len(self.ldr),
            f'{self.client_name}/dice': mean_dice / len(self.ldr)
        })

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)
