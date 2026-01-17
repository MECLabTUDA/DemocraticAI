from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image
import pandas as pd

import numpy as np
import einops 
import pickle as pkl
import cv2
from utils.root_path import get_root_path, xray_path

import matplotlib.pyplot as plt

class XRayDataset(Dataset):

    def __init__(self, client_name, num_clients: int, num_test: float, data_split_seed: int, root_path = '/gris/scratch-gris-filesrv'):
        self.root_path   = root_path
        assert num_clients == 3

        if client_name == 'val':
            client_idx = 3
        else:
            assert client_name.startswith("client-")
            client_idx = int(client_name.split("-")[1]) - 1

        self.data_path = xray_path



        
        patients_x8 = os.listdir(os.path.join(self.data_path, "_Converted_ChestX8_png", "images"))
        patients_mimic = os.listdir(os.path.join(self.data_path, "MIMIC_50", "images"))
        patients_padchest = os.listdir(os.path.join(self.data_path, "Padchest_50", "images"))



        patients_x8 = list(set([i.split('_')[0] for i in patients_x8])) #X8 has multiple images per patient

        patients_x8_train, patients_x8_test = self.split_patients(patients_x8, num_test, data_split_seed) #ChestX8 has only lung segmentations
        images_mimic_train, images_mimic_test = self.split_patients(patients_mimic, num_test, data_split_seed)
        images_padchest_train, images_padchest_test = self.split_patients(patients_padchest, num_test, data_split_seed)

        images_x8_train = self.get_images_x8(patients_x8_train)
        images_x8_test = self.get_images_x8(patients_x8_test)

        images_x8_train = [(i, "_Converted_ChestX8_png") for i in images_x8_train]
        images_x8_test = [(i, "_Converted_ChestX8_png") for i in images_x8_test]
        images_mimic_train = [(i, "MIMIC_50") for i in images_mimic_train]
        images_mimic_test = [(i, "MIMIC_50") for i in images_mimic_test]
        images_padchest_train = [(i, "Padchest_50") for i in images_padchest_train]
        images_padchest_test = [(i, "Padchest_50") for i in images_padchest_test]


        if client_idx == 0: #client-1
            self.images = images_x8_train #only binary labels!
        elif client_idx == 1: #client-2
            self.images = images_mimic_train
        elif client_idx == 2: #client-3
            self.images = images_padchest_train
        else:
            assert client_name == 'val'
            self.images = images_x8_test + images_mimic_test + images_padchest_test
            #self.images = images_padchest_test

        self.client_idx = client_idx

        print(f"Client {client_name} has {len(self.images)} images")


    def get_images_x8(self, patients):
        images = []
        for file in os.listdir(os.path.join(self.data_path, "_Converted_ChestX8_png", "images")):
            patient_id = file.split("_")[0]
            if patient_id in patients:
                images.append(file)
        return images

    def split_patients(self, patients: list, num_test: float, data_split_seed: int) -> tuple:
        assert num_test > 0 and num_test < 1, f"num_test must be in (0, 1), but is {num_test}"
        patients.sort()
        rng = np.random.default_rng(data_split_seed)
        rng.shuffle(patients)

        num_test = int(num_test * len(patients))
        return patients[num_test:], patients[:num_test]


    
    def __len__(self):
        return len(self.images)

    
    def rgb2gray(self, rgb: np.ndarray) -> np.ndarray:
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



    def __getitem__(self, idx):
        img_name, dataset = self.images[idx]

        img_path = os.path.join(self.data_path, dataset, "images", img_name)
        lbl_path = os.path.join(self.data_path, dataset, "labels", img_name)

        
        img = Image.open(img_path).convert('L')
        lbl = Image.open(lbl_path).convert('RGB')

        img = np.array(img).astype(float)
        lbl = np.array(lbl)
        #discard blue channel
        lbl = lbl[:,:,0] + lbl[:,:,1]
        lbl = (lbl > 0).astype(float)

        img /= 255.0
        img -= 0.5

        img = einops.rearrange(img, 'h w -> 1 h w')
        lbl = einops.rearrange(lbl, 'h w -> 1 h w')

        return img, lbl


def XRay_loading(client_name, num_clients, num_test: float, data_split_seed, batch_size, shuffle):
    root_dir = get_root_path()

    
    ds = XRayDataset(client_name, num_clients, num_test, data_split_seed, root_dir)
    ldr = DataLoader(ds, batch_size = batch_size, shuffle=shuffle)

    return ldr
