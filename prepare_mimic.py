import os
from getpass import getpass
import pandas as pd
import requests
from getpass import getpass
import os
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
from utils.PhysioNetClient import PhysioNetClient
from urllib.parse import urljoin


MIMIC_CXR = "https://physionet.org/files/mimic-cxr/2.1.0"
MIMIC_JPG = "https://physionet.org/files/mimic-cxr-jpg/2.1.0"
CHEXMASK = "https://physionet.org/files/chexmask-cxr-segmentation-data/1.0.0"

output_path = '/local/scratch/clmn1/data/xray/MIMIC_new/'
output_size = (256, 256) # tuple (64, 64) or 'original'
num_images = 10
username = input("PhysioNet Username: ")
password = getpass("PhysioNet Password: ")


def find_path_to_img(dicom_id, image_filenames):
    for line in image_filenames:
        if dicom_id in line:
            return line.strip()
    return None

def get_mask_from_RLE(rle, height, width):
    mask = np.zeros(height * width, dtype=np.uint8)
    if pd.isna(rle):
        return mask.reshape((height, width))
    rle_numbers = [int(num) for num in rle.split()]
    for i in range(0, len(rle_numbers), 2):
        start = rle_numbers[i] - 1
        length = rle_numbers[i + 1]
        mask[start:start + length] = 1
    return mask.reshape((height, width))



client = PhysioNetClient(username, password)
client.login()


image_filenames_file = os.path.join(output_path, "image_filenames.txt")
if not os.path.exists(image_filenames_file):
    client.download_file(MIMIC_JPG + "/IMAGE_FILENAMES", save_path=image_filenames_file)
with open(image_filenames_file, 'r') as file:
    image_filenames = file.readlines()

mimic_segmentation_file = os.path.join(output_path, "MIMIC-CXR-JPG.csv")
if not os.path.exists(mimic_segmentation_file):
    client.download_file(CHEXMASK + "/OriginalResolution/MIMIC-CXR-JPG.csv", save_path=mimic_segmentation_file)
df = pd.read_csv(mimic_segmentation_file)
df.sort_values(by='Dice RCA (Mean)', ascending=False, inplace=True)
df = df.reset_index(drop=True)

images_out_path = os.path.join(output_path, "images/")
masks_out_path = os.path.join(output_path, "labels/")
os.makedirs(images_out_path, exist_ok=True)
os.makedirs(masks_out_path, exist_ok=True)

for i, row in df.iterrows():
    if i > num_images:
        break
    assert row["Dice RCA (Mean)"] >= 0.7
    dicom_id = row['dicom_id']
    img_path = find_path_to_img(dicom_id, image_filenames)
    client.download_file(
        url=urljoin(MIMIC_JPG + "/", img_path),
        save_path=os.path.join(images_out_path, f"{dicom_id}.jpg")
    )
    if output_size != 'original':
        img = Image.open(os.path.join(images_out_path, f"{dicom_id}.jpg"))
        img = img.resize(output_size, resample=Image.BILINEAR)
        img.save(os.path.join(images_out_path, f"{dicom_id}.png"))
        os.remove(os.path.join(images_out_path, f"{dicom_id}.jpg"))

    masks = [get_mask_from_RLE(row[anatomy], int(row['Height']), int(row['Width'])) for anatomy in ['Left Lung', 'Right Lung', 'Heart']]
    mask = np.stack(masks, axis=-1)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    if output_size != 'original':
        mask_img = mask_img.resize(output_size, resample=Image.NEAREST)
    # save as PNG
    mask_img.save(os.path.join(masks_out_path, f"{dicom_id}.png"))
    
    
