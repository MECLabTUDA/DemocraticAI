import os
import pandas as pd
from utils.root_path import ultrasound_path, xray_mimic200_path, crc_path
import numpy as np

def get_images_for_reconstruction_fetal_abdominal():
    patient_to_image = {}
    for file in os.listdir(os.path.join(ultrasound_path, "ARRAY_FORMAT")):
        patient_id = file.split("_")[0]
        assert patient_id.startswith("P")
        patient_id = int(patient_id[1:])
        if not patient_id in patient_to_image:
            patient_to_image[patient_id] = []
        patient_to_image[patient_id].append(file[:-len(".npy")])

    images = []

    random_generator = np.random.default_rng(seed=42)
    for patient_id in sorted(patient_to_image.keys()):
        img_name = random_generator.choice(patient_to_image[patient_id])
        images.append(img_name)
    return images

def get_images_for_reconstruction_xray_mimic200():
    random_generator = np.random.default_rng(seed=42)
    images = sorted(os.listdir(os.path.join(xray_mimic200_path, "images")))
    images = random_generator.choice(images, size=100, replace=False)
    images = [img[:-len(".png")] for img in images]  # Remove the .png extension
    return images

def get_images_for_reconstruction_crc():
    random_generator = np.random.default_rng(seed=42)
    _classes = sorted(os.listdir(os.path.join(crc_path, "NCT-CRC-HE-100K")))
    all_images = []
    for c in _classes:
        class_images = sorted(os.listdir(os.path.join(crc_path, "NCT-CRC-HE-100K", c)))
        all_images.extend([os.path.join(c, img) for img in class_images])
    all_images = sorted(all_images)
    images = random_generator.choice(all_images, size=100, replace=False)
    images = [img[:-len(".tif")] for img in images]  # Remove the .tif extension
    return images