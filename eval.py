

from dataset.FetalAbdominal import FetalAbdominal_loading

import torch
from unet import UNet3D, UNet2D
import tqdm
import einops
from encryption.ModelEncryptor import ModelEncryptor
import losses.dice as dice

from network.MedNCA import MedNCA

import nvflare.client as flare
from quantization.dequantization import dequantize_model

from utils.root_path import get_root_path, results_workingdir_path, results_single_path, results_tempdir_path
from network.model import get_model
from dataset.loader import get_data_loader

import tenseal as ts
import os
import numpy as np
import click
import utils.hyperparameters as hyp
import pandas as pd

class FetalAbdominalValidator:

    def __init__(self, val_client_name, data_type, model_type, algorithm,
                 quantize_mode, sparsification_mode, sparsification_parameter, apply_homomorphic_encryption,
                 malfclients, malf, malf_prob):
        path = get_root_path()
        self.device = 'cuda:0'
        self.supervision_scenario = "full"

        num_clients, data_split_seed, num_test, num_rounds = hyp.scenario_vars(data_type)

        self.classification = hyp.is_classification_task(data_type)
        
        self.ldr  = get_data_loader(data_type, val_client_name, num_clients, num_test, data_split_seed, batch_size=1, shuffle=False)
        
        self.quantize_mode = quantize_mode
        self.sparsification_parameter = sparsification_parameter
        self.apply_homomorphic_encryption = apply_homomorphic_encryption

        self.sparsification_mode = sparsification_mode
        if self.sparsification_mode == "none":   
            self.sparsification_parameter = 0.0


        self.model = get_model(data_type, model_type, self.device)

        
        self.setup_name = f"{data_type}_{num_clients}_{data_split_seed}_{num_test}_{self.supervision_scenario}"
        self.exp_name = f"exp_{algorithm}_malf_{malf}_malfclients_{'_'.join(malfclients)}_{self.quantize_mode}_{self.sparsification_mode}_{self.sparsification_parameter}_{self.apply_homomorphic_encryption}_{model_type}"
        self.workingdir_path = os.path.join(results_workingdir_path, self.setup_name, self.exp_name)

        self.path_single = os.path.join(results_single_path, self.setup_name, f"exp_{model_type}")
        self.data_type = data_type

    @torch.no_grad()
    def eval(self, model_path=None):
        if model_path is None:
            model_path = "server/simulate_job/app_server/FL_global_model.pt"

        dices = []
        tp, fp, fn, tn = 0, 0, 0, 0


        self.load_global_model(os.path.join(self.workingdir_path, model_path))
        
        
        self.model.to(self.device)
        self.model.eval()
        for img, label in tqdm.tqdm(self.ldr):
            img, label = img.to(self.device, torch.float32), label.to(self.device, torch.float32)
            outputs     = self.model(img)
            if isinstance(self.model, MedNCA):
                outputs = einops.rearrange(outputs, 'b h w c -> b c h w')

            if self.classification and hyp.is_multiclass_classification(self.data_type):
                preds = torch.argmax(outputs, dim=1)
                if not hasattr(self, 'confusion_matrix'):
                    self.confusion_matrix = torch.zeros((outputs.shape[1], outputs.shape[1]), dtype=torch.int64)
                for t, p in zip(label.view(-1), preds.view(-1)):
                    self.confusion_matrix[t.long(), p.long()] += 1
            elif self.classification:
                outputs = (outputs > 0).float()
                preds = outputs.long()
                tp += ((preds == label) & (label == 1)).sum().item()
                fp += ((preds == 1) & (label == 0)).sum().item()
                fn += ((preds == 0) & (label == 1)).sum().item()
                tn += ((preds == label) & (label == 0)).sum().item()
            else:
                outputs = (outputs > 0).float()
                dices.append(dice.DiceLoss.compute_dice(outputs, label))

        if self.classification and hyp.is_multiclass_classification(self.data_type):
            return self.confusion_matrix
        elif self.classification:
            return tp, fp, fn, tn
        else:
            dices = torch.tensor(dices)
            return torch.mean(dices), torch.std(dices)

    def load_parameters(self, params: dict):
        params = {k: torch.tensor(v) for k, v in params.items()}
        self.model.load_state_dict(params)

    def load_global_model(self, model_path):
        _dict = torch.load(model_path, weights_only=False)
        if isinstance(_dict, flare.FLModel):
            model = _dict
        else:
            model = flare.FLModel(
                params = _dict['model'],
                meta   = _dict['meta_props'])

        if self.apply_homomorphic_encryption:
            # we need the private key to decrypt the model parameters
            #with open(os.path.join(results_tempdir_path, self.setup_name, self.exp_name, "private_context.seal"), 'rb') as f:
            #    private_seal_context = ts.context_from(f.read())
            model_encryptor = ModelEncryptor(True, os.path.join(results_tempdir_path, self.setup_name, self.exp_name, "private_context.seal"))
            dummy_model = flare.FLModel(params = self.model.cpu().state_dict())
            model_encryptor.encrypt(dummy_model)


            decrypted_model: flare.FLModel = model_encryptor.decrypt(model)

            self.load_parameters(decrypted_model.params)
        elif self.quantize_mode != 'none':
            model = dequantize_model(model, self.quantize_mode)
            self.load_parameters(model.params)
        else:
            self.load_parameters(model.params)

@click.command()
@click.option('--data', default='fetalAbdominal', type=str, help='Type of data to validate on', required=True)
@click.option('--model', default='mednca', type=str, help='Type of model to validate', required=True)
@click.option('--algorithm', default='FedAvg', help='Aggregation rule', required=True)
@click.option('--quantize_mode', default='none', type=str, help='Quantization mode to use', required=False)
@click.option('--sparsification_mode', default='none', type=str, help='Sparsification mode to use', required=False)
@click.option('--sparsification_parameter', default=0.01, type=float, help='Sparsification parameter to use', required=False)
@click.option('--apply_homomorphic_encryption', default=False, type=bool, is_flag=True, help='Apply homomorphic encryption', required=False)
@click.option('--malfclients', multiple=True, type=str, help='Ids of malfunctioning clients', required=False)
@click.option('--malf', type=str, help='possible values: ana sfa', required=False)
@click.option('--malf_prob', type=float, help='possible values: number between zero and one', default=0.0, required=False)
@click.option('--checkpoints','eval_checkpoints', default=False, type=bool, is_flag=True, help='Evaluate checkpoints', required=False)
def main(data, model, algorithm, quantize_mode, sparsification_mode, sparsification_parameter, apply_homomorphic_encryption,
         malfclients, malf, malf_prob, eval_checkpoints):

    split = "val"
    def create_validator():
        return FetalAbdominalValidator(split, data, model, algorithm, quantize_mode, sparsification_mode, sparsification_parameter, apply_homomorphic_encryption,
                                        malfclients, malf, malf_prob)

    if eval_checkpoints:
        workingdir_path = create_validator().workingdir_path
        checkpoints = os.listdir(os.path.join(workingdir_path, "checkpoints"))
        checkpoints = sorted([int(checkpoint.split('_')[-1].split('.')[0]) for checkpoint in checkpoints])
    else:
        checkpoints = [None]

    for checkpoint_round in checkpoints:
        validator = create_validator()

        if checkpoint_round is not None:
            out_file = os.path.join(validator.workingdir_path, "checkpoint_val", f"dice_{checkpoint_round}.txt")
            if os.path.exists(out_file):
                print(f"Output file {out_file} already exists, skipping...")
                continue
            eval_res = validator.eval(model_path=os.path.join(workingdir_path, "checkpoints", f"global_model_round_{checkpoint_round}.pth"))
        else:
            eval_res = validator.eval()
            out_file = os.path.join(validator.workingdir_path, "dice.txt")

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        assert not os.path.exists(out_file), f"Output file {out_file} already exists"
        if isinstance(eval_res, torch.Tensor): #multiclass classification case
            confusion_matrix = eval_res
            num_classes = confusion_matrix.shape[0]
            df = pd.DataFrame(
                confusion_matrix,
                index=[f"GT_{i}" for i in range(num_classes)],
                columns=[f"P_{i}" for i in range(num_classes)]
            )
            df.to_csv(out_file, sep='\t')



            
        elif len(eval_res) == 4:  # classification case
            tp, fp, fn, tn = eval_res
            with open(out_file, 'a') as out_file:
                out_file.write(f'{split}\t{tp}\t{fp}\t{fn}\t{tn}\n')
            print(f'Classification results - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')
        else:
            dice_mean, dice_std = eval_res
            with open(out_file, 'a') as out_file:
                out_file.write(f'{split}\t{dice_mean}\t{dice_std}\n')
            print(f'Dice score: {dice_mean} \u00b1 {dice_std}')


if __name__ == '__main__':
    main()