import os

failed_commands = []

def run_command(command):
    print(f"Running command: {command}")
    result = os.system(command)
    if result != 0:
        failed_commands.append(command)
    return result

def train_and_eval(args):
    run_command(f"python train_fed.py {args} --no_rerun")
    run_command(f"python eval.py {args}")



for model in ["vit4sgd"]:
    #train_and_eval(f"--data crc --model {model}")
    train_and_eval(f"--data crc --model {model} --quantize_mode float4")
    for sparsification_parameter in [0.25, 0.01]:
        train_and_eval(f"--data crc --model {model} --sparsification_mode top-k --sparsification_parameter {sparsification_parameter}")




print(f"Failed commands:", end='')
for command in failed_commands:
    print(f"\n{command}")