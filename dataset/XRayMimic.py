from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image
import pandas as pd

import numpy as np
import einops 
import pickle as pkl
import cv2
from utils.root_path import get_root_path, xray_mimic_path, xray_mimic200_path

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

class XRayMimicDataset(Dataset):

    def __init__(self, client_name, num_clients: int, num_test: float, data_split_seed: int, root_path = '/gris/scratch-gris-filesrv', version=50):
        self.root_path   = root_path

        if client_name == 'val':
            client_idx = 3
        else:
            assert client_name.startswith("client-")
            client_idx = int(client_name.split("-")[1]) - 1

        if version == 50:
            self.data_path = xray_mimic_path
        elif version == 200:
            self.data_path = xray_mimic200_path

        images = list(os.listdir(os.path.join(self.data_path, "images")))
        rng = np.random.default_rng(data_split_seed)
        rng.shuffle(images)

        if client_name == 'val':
            client_idx = num_clients
        else:
            assert client_name.startswith("client-")
            client_idx = int(client_name.split("-")[1]) - 1

        num_test = int(num_test * len(images))
        

        images_val = images[:num_test]
        images = images[num_test:]

        if client_name == "val":
            self.images = images_val
        else:
            num_splits = num_clients
            self.images = images[client_idx * len(images) // num_splits : (client_idx + 1) * len(images) // num_splits]

        self.images.sort()
        print(f"Client {client_name} has {len(self.images)} images: {self.images}")
        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5,std=0.5) #normalize to [-1, 1]
        ])


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.data_path, "images", img_name)
        lbl_path = os.path.join(self.data_path, "labels", img_name)

        img = Image.open(img_path).convert('L')
        lbl = Image.open(lbl_path)
        img = self.transforms(img)

        lbl = np.array(lbl)
        lbl = lbl[:, :, :3]  # Keep only the first three channels
        lbl[:,:,2] = 0
        lbl = lbl.max(axis=2)  # Convert to grayscale
        lbl = torch.from_numpy(lbl)
        lbl = F.interpolate(lbl[None, None, :, :].float(), size=(64, 64), mode='nearest')[0, 0]
        lbl = lbl.bool().float()


        #img = einops.rearrange(img, 'h w -> 1 h w')
        lbl = einops.rearrange(lbl, 'h w -> 1 h w')

        return img, lbl


def XRayMimic_loading(client_name, num_clients, num_test: float, data_split_seed, batch_size, shuffle, version=50):
    root_dir = get_root_path()

    
    ds = XRayMimicDataset(client_name, num_clients, num_test, data_split_seed, root_dir, version=version)
    ldr = DataLoader(ds, batch_size = batch_size, shuffle=shuffle)

    return ldr
