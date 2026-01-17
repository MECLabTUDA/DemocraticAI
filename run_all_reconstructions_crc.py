import os
import numpy as np
import pandas as pd
from utils.get_reconstruction_imgs import get_images_for_reconstruction_crc

failed_commands = []

def run_command(command):
    print(f"Running command: {command}")
    result = os.system(command)
    if result != 0:
        failed_commands.append(command)
    return result


seg_models = [
    #("vit4sgd", "l2"),
    #("densenet", "l2"),
    #("maxmednca_nobn", "l2_normalized"),
    ("deterministic_maxmednca_nobn", "l2_normalized"),
]

images = get_images_for_reconstruction_crc()


for i, path in enumerate(images):
    for model, distance in seg_models:
        for known_label in [True, False]:
            run_command(f"CUDA_VISIBLE_DEVICES=7 python reconstruction.py --dataset crc --case_name {path} --model {model} --known_label {known_label} --distance_name {distance} --no_rerun")

print(f"Failed commands:", end='')
for command in failed_commands:
    print(f"\n{command}")