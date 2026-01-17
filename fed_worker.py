from training.Trainer import Trainer

import nvflare.client as flare
import torch
import os
import random

import click
import wandb

from quantization.dequantization import dequantize_model
from quantization.quantization import quantize_model

from sparsification.model_sparsifier import ModelSparsifier
from encryption.ModelEncryptor import ModelEncryptor
from dataset.loader import get_data_loader
from utils.fl_utils import get_size_of_model, print_model_summary
from training.malf_update import Malfunction
from utils import hyperparameters as hyp

@click.command()
@click.option('--num_clients', default=3, help='Number of clients', required=True)
@click.option('--malf', type=str, help='possible values: ana sfa')
@click.option('--malf_prob', type=float, default=0.0, help='possible values: numbet between zero and one')
@click.option('--model', default='mednca', help='Model to use', required=True)
@click.option('--quantization_type', default='none', help='Quantization type', required=True)
@click.option('--sparsification_mode', default='none', help='Sparsification mode', required=True)
@click.option('--sparsification_parameter', default=None, help='Sparsification parameter', type=float, required=True)
@click.option('--data_split_seed', default=42, help='Data split seed', required=True)
@click.option('--data_type', default='fetalAbdominal', help='Data type', required=True)
@click.option('--num_test', default=0.2, help='Number of test samples', required=True)
@click.option('--supervision_scenario', default="full", help='Unsupervised scenario', required=True)
@click.option('--group_name', default="single", help='Group name', required=True)
@click.option('--apply_homomorphic_encryption', default=False, help='Apply homomorphic encryption', required=True)
@click.option('--private_seal_context', default=None, help='Private seal context', required=True)
@click.option('--enable_wandb', default=False, help='Enable wandb logging', required=True)
@click.option('--uplink_modelsize_path', default=None, help='Path to save uplink model size', required=True)
def main(num_clients, model, quantization_type, sparsification_mode, 
         sparsification_parameter, data_split_seed, data_type, num_test, 
         supervision_scenario, group_name, apply_homomorphic_encryption,
         private_seal_context, enable_wandb, uplink_modelsize_path, malf, malf_prob):
    flare.init()
    sys_info        = flare.system_info()
    client_name     = sys_info['site_name']
    batch_size      = hyp.get_batch_size(data_type)
    ldr             = get_data_loader(data_type, client_name, num_clients, num_test, data_split_seed, batch_size=batch_size, shuffle=True)
    trainer         = Trainer(ldr, data_type, model, client_name)
    local_epochs    = hyp.get_local_epochs(data_type)
    corrupt         = Malfunction()

    full_supervision    = supervision_scenario == "full"
    if client_name == 'client-1': # this client is always fully supervised
        full_supervision = True
        supervision_scenario = "full"

    
    wandb.init(project='fedNCA', name=client_name, group=group_name,
               config={
                     'num_clients': num_clients,
                        'model': model,
                        'quantization_type': quantization_type,
                        'sparsification_mode': sparsification_mode,
                        'sparsification_parameter': sparsification_parameter,
                        'data_split_seed': data_split_seed,
                        'data_type': data_type,
                        'num_test': num_test,
                        'supervision_scenario': supervision_scenario,
                        'malf': malf,
                        'malf_prob': malf_prob,

                        'local_epochs': local_epochs,
               },
               mode='online' if enable_wandb else 'disabled')
    
    first_round = True

    assert sparsification_mode == 'none' or quantization_type == 'none', 'Cannot use both sparsification and quantization'
    assert int(sparsification_mode != 'none') + int(quantization_type != 'none') + int(apply_homomorphic_encryption == True) <= 1, ' can only use one of sparsification, quantization and homomorphic encryption'
    model_sparsifier = ModelSparsifier(trainer.model.cpu().state_dict(), sparsification_mode, sparsification_parameter)
    model_encryptor = ModelEncryptor(apply_homomorphic_encryption, private_seal_context)

    while flare.is_running():
        global_model = flare.receive()

        if first_round:
            # not the most clever way to simulate the first round, but it should work
            first_round = False
            global_model = quantize_model(global_model, quantization_type)
        else:
            global_model = model_encryptor.decrypt(global_model)
        global_model = dequantize_model(global_model, quantization_type)

        if not full_supervision and global_model.current_round < 5:
            print(f"{client_name} skipping round {global_model.current_round} for {client_name}")
        else:
            print(f"{client_name} current_round={global_model.current_round}")
            trainer.load_model(global_model.params)
            print(f'{client_name} Training starts')
            for epoch in range(local_epochs):
                trainer.train_epoch(supervision_scenario=supervision_scenario)

        local_model_params = trainer.model.cpu().state_dict()
        if random.random() < malf_prob:
            print(f"{client_name} is committing a {malf} attack")
            local_model_params = corrupt(local_model_params, malf)


        local_model = flare.FLModel(
                params = local_model_params,
                meta   = {'NUM_STEPS_CURRENT_ROUND': local_epochs * len(ldr),
                          'CLIENT_NAME'            : client_name})
       

        if local_model is None:
            print(f"{client_name}: Model is None after creating FLModel object")
        local_model = model_sparsifier.sparsify_model(local_model, global_model)
        if local_model is None:
            print(f"{client_name}: Model is None after sparsification")
        local_model = quantize_model(local_model, quantization_type)
        if local_model is None:
            print(f"{client_name}: Model is None after quantification")

        local_model = model_encryptor.encrypt(local_model)
        if local_model is None:
            print(f"{client_name}: Model is None after encryption")

        with open(uplink_modelsize_path, 'a') as f:
            f.write(f"{get_size_of_model(local_model)}\n")

        flare.send(local_model)
    
    print(f'{client_name} finished training')

if __name__ == '__main__':
    main()
