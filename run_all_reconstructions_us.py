import os
import numpy as np
from utils.root_path import ultrasound_path
from utils.get_reconstruction_imgs import get_images_for_reconstruction_fetal_abdominal

failed_commands = []

def run_command(command):
    print(f"Running command: {command}")
    result = os.system(command)
    if result != 0:
        failed_commands.append(command)
    return result


seg_models = [
    ("unet", "l2_normalized"),
    ("segformer", "l2"),
    ("transunet_b16", "l2"),
]


images = get_images_for_reconstruction_fetal_abdominal()
for img_name in images:

    #for model, distance in seg_models:
    #    for known_label in [True, False]:
    #        run_command(f"python reconstruction.py --dataset fetalAbdominal --case_name {img_name} --model {model} --known_label {known_label} --distance_name {distance} --no_rerun")
    
    #CUDA_VISIBLE_DEVICES=1
    run_command(f"CUDA_VISIBLE_DEVICES=1 python reconstruction.py --dataset fetalAbdominal --case_name {img_name} --model deterministic_mednca --known_label True --distance_name l2_normalized --no_rerun")

print(f"Failed commands:", end='')
for command in failed_commands:
    print(f"\n{command}")