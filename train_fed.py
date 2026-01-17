from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner

import torch
from torch import nn
import click

from nvflare import FedJob
from nvflare.app_common.workflows.fedavg import FedAvg
from aggregation.quantized_aggregation import QuantizedAggregator
from aggregation.robust_aggregation import Aggregator
import os

from utils.root_path import get_root_path, results_tempdir_path, results_workingdir_path
from network.model import get_model
from datetime import datetime

import tenseal as ts
from utils import hyperparameters as hyp

@click.command()
@click.option('--data', 'data_type', default='fetalAbdominal', help='Data type', required=True)
@click.option('--model', 'model_type', default='new_mednca', help='Model type', required=True)
@click.option('--algorithm', default='FedAvg', help='Aggregation rule', required=True)
@click.option('--quantize_mode', default='none', help='Quantization mode', required=False)
@click.option('--sparsification_mode', default='none', help='Sparsification mode', required=False)
@click.option('--sparsification_parameter', default=0.25, type=float, help='Sparsification parameter', required=False)
@click.option('--enable_wandb', default=False, type=bool, is_flag=True, help='Enable wandb logging', required=False)
@click.option('--apply_homomorphic_encryption', default=False, type=bool, is_flag=True, help='Apply homomorphic encryption', required=False)
@click.option('--no_rerun', default=False, type=bool, is_flag=True, help='Do not ask to rerun the job if it already exists', required=False)
@click.option('--malfclients', multiple=True, type=str, help='Ids of malfunctioning clients', required=False)
@click.option('--malf', type=str, help='possible values: ana sfa', required=False)
@click.option('--malf_prob', type=float, help='possible values: number between zero and one', default=0.0, required=False)
@click.option('--save_checkpoints', default=False, type=bool, is_flag=True, help='Save a global model checkpoint after each round', required=False)
def main(data_type, model_type, quantize_mode, sparsification_mode, sparsification_parameter, enable_wandb, apply_homomorphic_encryption, no_rerun, algorithm, malf, malfclients, malf_prob, save_checkpoints):
    path = get_root_path()

    num_clients, data_split_seed, num_test, num_rounds = hyp.scenario_vars(data_type)


    model = get_model(data_type, model_type)

    run_sia = False     # source inferecnce attack
    train_script = 'fed_worker.py'
    supervision_scenario = "full" # "full", "weak", "none"

    if sparsification_mode == "none":
        sparsification_parameter = 0.0

    setup_name = f"{data_type}_{num_clients}_{data_split_seed}_{num_test}_{supervision_scenario}"
    exp_name = f"exp_{algorithm}_malf_{malf}_malfclients_{'_'.join(malfclients)}_{quantize_mode}_{sparsification_mode}_{sparsification_parameter}_{apply_homomorphic_encryption}_{model_type}"

    workingdir_path = os.path.join(results_workingdir_path, setup_name, exp_name)
    temp_dir_path = os.path.join(results_tempdir_path, setup_name, exp_name)
    random_init_path = os.path.join(temp_dir_path, "random_init.pth")
    sia_results_path = os.path.join(workingdir_path, "sia_results.txt")

    if os.path.exists(os.path.join(workingdir_path, "dice.txt")):
        if no_rerun or input("Dice file exists. Overwrite? (y/n)") != "y":
            exit()

    job = BaseFedJob(name='Ultrasound', min_clients=num_clients, initial_model=model)

    os.makedirs(workingdir_path, exist_ok=True)
    os.makedirs(os.path.dirname(random_init_path), exist_ok=True)
    torch.save(model.state_dict(), random_init_path)

    if not os.path.exists(os.path.join(temp_dir_path, "private_context.seal")) or not os.path.exists(os.path.join(temp_dir_path, "public_context.seal")):
        #context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384, coeff_mod_bit_sizes=[60, 40, 40, 60])
        #context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40])
        context.global_scale = 2**40
        context.generate_galois_keys()
        context.generate_relin_keys()
        with open(os.path.join(temp_dir_path, "private_context.seal"), "wb") as f:
            f.write(context.serialize(save_secret_key=True))

        with open(os.path.join(temp_dir_path, "public_context.seal"), "wb") as f:
            f.write(context.serialize(save_secret_key=False))

    if algorithm == 'FedAvg':
        controller = QuantizedAggregator(
            quantization_type = quantize_mode,
            model_type = model_type,
            sparsification_mode = sparsification_mode,
            sparsification_parameter = sparsification_parameter,
            random_init_path = random_init_path,
            num_clients  = num_clients,
            num_rounds   = num_rounds,
            run_sia = run_sia,
            data_type = data_type,
            sia_results_path=sia_results_path,
            num_test=num_test,
            data_split_seed=data_split_seed,
            apply_homomorphic_encryption=apply_homomorphic_encryption,
            public_seal_context_path=os.path.join(temp_dir_path, "public_context.seal"),
            downlink_modelsize_path=os.path.join(workingdir_path, "downlink_modelsize.txt"),
            checkpoint_dir=os.path.join(workingdir_path, "checkpoints") if save_checkpoints else "None",
        )
    elif algorithm == 'ASMR':
        assert quantize_mode == 'none', "ASMR currently only supports no quantization"
        assert sparsification_mode == 'none', "ASMR currently only supports no sparsification"
        assert apply_homomorphic_encryption == False, "ASMR currently only supports no homomorphic encryption"
        assert not save_checkpoints, "ASMR currently does not support saving checkpoints"
        controller = Aggregator(
            num_clients  = num_clients,
            num_rounds   = num_rounds,
            apply_homomorphic_encryption=apply_homomorphic_encryption,
            persistor_id = job.comp_ids["persistor_id"],
            algo         = 'ASMR'
            )
    else:
        raise ValueError(f"Unknown federated algorithm: {algorithm}")

    job.to_server(controller)

    time_stamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    group_name = f"{time_stamp}"

    # Add clients
    for i in range(num_clients):
        prob = 0.0
        if str(i+1) in malfclients:
            prob = malf_prob
            
        executor = ScriptRunner(script=train_script, 
                                script_args=f"--num_clients {num_clients} \
                                --model {model_type} \
                                --malf {malf} \
                                --malf_prob {prob} \
                                --quantization_type {quantize_mode} \
                                --sparsification_mode {sparsification_mode} \
                                --sparsification_parameter {sparsification_parameter} \
                                --data_split_seed {data_split_seed} \
                                --data_type {data_type} \
                                --num_test {num_test} \
                                --supervision_scenario {supervision_scenario} \
                                --group_name {group_name} \
                                --apply_homomorphic_encryption {apply_homomorphic_encryption}\
                                --private_seal_context {os.path.join(temp_dir_path, 'private_context.seal')} \
                                --enable_wandb {enable_wandb} \
                                --uplink_modelsize_path {os.path.join(workingdir_path, f'uplink_modelsize_{i+1}.txt')}")
        job.to(executor, f"client-{i+1}", tasks=["train"])

    #run the job
    job.simulator_run(workingdir_path, gpu="0")

if __name__ == '__main__':
    main()
