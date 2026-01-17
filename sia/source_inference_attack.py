import numpy as np
import torch
from torch import nn
from nvflare.app_common.abstract.fl_model import FLModel

from dataset.FetalAbdominal import FetalAbdominal_loading

from network.ultrasound import get_ultrasound_model
from network.model import get_model

import einops 
import scipy.stats

from dataset.loader import get_data_loader
import losses.dice as dice

class SIA(object):
    def __init__(self, num_clients: int, data_type: str, model_type: str, num_test:float, data_split_seed: int):
        self.num_clients = num_clients
        self.data_type = data_type
        self.model_type = model_type
        self.data_split_seed = data_split_seed
        self.num_test = num_test

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"SIA running on {self.device}")
        self.dtype = torch.float32

        self.prediction_of_lowest_entropy = {}
        self.lowest_entropy = {}

    def instantiate_dataloader(self, idx: int):
        ldr = get_data_loader(self.data_type, f"client-{idx+1}", self.num_clients, self.num_test, data_split_seed=self.data_split_seed, batch_size=1, shuffle=False)
        
        N = len(ldr.dataset)

        if f"client-{idx}" not in self.prediction_of_lowest_entropy.keys():
            self.prediction_of_lowest_entropy[f"client-{idx}"] = np.zeros((N,))
            self.lowest_entropy[f"client-{idx}"] = np.ones((N,)) * np.inf
        
        return ldr

    def instantiate_model(self):
        return get_model(self.data_type, self.model_type, self.device).to(self.device)


    @torch.no_grad()
    def attack(self, w_locals: list[FLModel]):
        net = self.instantiate_model()
        min_losses = []
        entropies = []
        correct_total_ent = 0
        correct_total = 0
        correct_total_dice_loss = 0
        len_set = 0
        for idx in range(self.num_clients):
            dataloader_local = self.instantiate_dataloader(idx)
            y_loss_all = []
            dice_loss_all = []
            # evaluate the selected training data on each local model
            for local in range(self.num_clients):
                y_loss_party = []
                dice_loss_party = []
                idx_tensor = torch.tensor(idx)
                w_local = {k: torch.from_numpy(v) for k, v in w_locals[local].params.items()}
                net.load_state_dict(w_local)
                net.eval()
                for id, (data, target) in enumerate(dataloader_local):
                    data, target = data.to(device=self.device, dtype=self.dtype), target.to(device=self.device, dtype=self.dtype)
                    idx_tensor = idx_tensor.to(device=self.device)
                    log_prob = net(data) # B, C, H, W

                    if self.model_type == "mednca":
                        log_prob = einops.rearrange(log_prob, 'b h w c -> b c h w')

                    # prediction loss based attack: get the prediction loss of the target training sample
                    loss = nn.BCEWithLogitsLoss(reduction='none')
                    y_loss: torch.Tensor = loss(log_prob, target)
                    dice_loss = 1 - dice.DiceLoss().compute_dice(log_prob > 0, target, dim=(1, 2, 3))
                    assert y_loss.dim() == 4, f"y_loss.dim()={y_loss.dim()} should be BCHW"
                    y_loss = y_loss.mean(dim=(1, 2, 3))
                    y_loss_party.append(y_loss.cpu().detach().numpy())
                    dice_loss_party.append(dice_loss.cpu().detach().numpy())

                y_loss_party = np.concatenate(y_loss_party).reshape(-1)
                y_loss_all.append(y_loss_party)

                dice_loss_all.append(dice_loss_party)



            y_loss_all = torch.tensor(y_loss_all).to(self.device)
            dice_loss_all = torch.tensor(dice_loss_all).to(self.device)
            # y_loss_all.shape = (C, N) where C is the number of clients and N is the number of samples in the dataset
            assert torch.all(y_loss_all >= 0), "y_loss_all should be non-negative"
            
            #y_loss_all_softmax = nn.functional.softmax(y_loss_all, dim=0)
            y_loss_all_softmax = y_loss_all / y_loss_all.sum(dim=0)

            entropy = scipy.stats.entropy(y_loss_all_softmax.cpu().detach().numpy(), axis=0) #(N,)
            min_loss, index_of_min_loss = y_loss_all.min(0, keepdim=True)
            # index_of_min_loss.shape = (1, N) where N is the number of samples in the dataset
            # min_loss.shape = (1, N)
            assert min_loss.shape[0] == 1
            min_loss = min_loss.cpu().detach()[0]

            _, index_of_min_dice_loss = dice_loss_all.min(0, keepdim=True)


            mask = entropy < self.lowest_entropy[f"client-{idx}"]

            self.prediction_of_lowest_entropy[f"client-{idx}"][mask] = index_of_min_loss.cpu().detach().numpy()[0, mask]
            self.lowest_entropy[f"client-{idx}"][mask] = entropy[mask]

            correct_local_ent = self.prediction_of_lowest_entropy[f"client-{idx}"] == idx
            correct_local_ent = correct_local_ent.sum()

            correct_local = index_of_min_loss == idx
            correct_local = correct_local.long().cpu().sum()

            correct_local_dice_loss = index_of_min_dice_loss == idx
            correct_local_dice_loss = correct_local_dice_loss.long().cpu().sum()

            min_losses.append(min_loss)
            entropies.append(entropy)

            correct_total_ent += correct_local_ent
            correct_total += correct_local
            correct_total_dice_loss += correct_local_dice_loss
            len_set += index_of_min_loss.shape[1]

        # calculate source inference attack accuracy
        accuracy_sia = 100.00 * correct_total / len_set
        accuracy_sia_ent = 100.00 * correct_total_ent / len_set
        accuracy_sia_dice_loss = 100.00 * correct_total_dice_loss / len_set
        min_losses = torch.cat(min_losses).cpu().numpy()
        min_losses_mean = min_losses.mean()

        entropies = np.concatenate(entropies)

        mean_entropy_all = np.mean(np.concatenate(list(self.lowest_entropy.values())))
        mean_entropy_epoch = np.mean(entropies)

        print(
            f'\nPrediction loss based source inference attack accuracy: {correct_total}/{len_set} ({accuracy_sia:.2f}%) with loss {min_losses_mean:.2f}\n')
        return {"acc. SIA": accuracy_sia, 
                "acc. SIA ent": accuracy_sia_ent,
                "acc. SIA dice loss": accuracy_sia_dice_loss,
                "mean min loss": min_losses_mean,
                "mean entropy all": mean_entropy_all,
                "mean entropy epoch": mean_entropy_epoch}
