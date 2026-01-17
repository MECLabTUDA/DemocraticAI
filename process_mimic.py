import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.root_path import xray_mimic200_path

os.makedirs(os.path.join(xray_mimic200_path, "images"), exist_ok=True)
os.makedirs(os.path.join(xray_mimic200_path, "labels"), exist_ok=True)

img_path = "/local/scratch/jkalkhof/Data/MICCAI24/MIMIC_200/images"
lbl_path = "/local/scratch/jkalkhof/Data/MICCAI24/MIMIC_200/labels"

for file in os.listdir(img_path):
    img = nib.load(os.path.join(img_path, file))
    img_data: np.ndarray = np.array(img.get_fdata())
    
    if img_data.ndim == 3:
        img_data = img_data[:, :, 0]  # take first slice

    assert img_data.shape == (256, 256), f"Image {file} has shape {img_data.shape}, expected (256, 256)"
    
    img_data = 255 * (img_data - np.min(img_data)) / (np.ptp(img_data) + 1e-8)
    img_data = img_data.astype(np.uint8)
    
    img = Image.fromarray(img_data, mode='L')
    img.save(os.path.join(xray_mimic200_path, "images", file.replace(".nii", ".png")))


for file in os.listdir(lbl_path):
    img = nib.load(os.path.join(lbl_path, file))
    img_data: np.ndarray = np.array(img.get_fdata())
    assert img_data.shape == (256, 256, 3), f"Image {file} has shape {img_data.shape}, expected (256, 256, 3)"
    img_data = 255 * (img_data - np.min(img_data)) / (np.ptp(img_data) + 1e-8)
    img_data = img_data.astype(np.uint8)
    img = Image.fromarray(img_data, mode='RGB')
    img.save(os.path.join(xray_mimic200_path, "labels", file.replace(".nii", ".png")))