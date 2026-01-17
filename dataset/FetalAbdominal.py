from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image

import numpy as np
import einops 
import pickle as pkl
import cv2
from utils.root_path import get_root_path, ultrasound_path

import torch
import torch.nn.functional as F

class FetalAbdominalDataset(Dataset):

    def __init__(self, client_name, num_clients: int, num_test: float, data_split_seed: int, root_path = '/gris/scratch-gris-filesrv', split_by_vendor = False):
        
        self.client_name = client_name
        self.root_path   = root_path

        self.data_path = ultrasound_path

        
        patient_to_image = {}

        for file in os.listdir(os.path.join(self.data_path, "IMAGES")):
            patient_id = file.split("_")[0]
            assert patient_id.startswith("P")
            patient_id = int(patient_id[1:])
            if not patient_id in patient_to_image:
                patient_to_image[patient_id] = []
            patient_to_image[patient_id].append(file[:-len(".png")])


        if split_by_vendor:
            assert num_clients == 2

            rgb_scanner_patients = []
            non_rgb_scanner_patients = []
            for patient_id in patient_to_image.keys():
                image = patient_to_image[patient_id][0]
                img = Image.open(os.path.join(self.data_path, "IMAGES", image + ".png"))
                img = np.array(img)
                assert len(img.shape) == 2 or len(img.shape) == 3
                is_rgb = len(img.shape) == 3

                if is_rgb:
                    rgb_scanner_patients.append(patient_id)
                else:
                    non_rgb_scanner_patients.append(patient_id)

                for image in patient_to_image[patient_id]:
                    img = Image.open(os.path.join(self.data_path, "IMAGES", image + ".png"))
                    img = np.array(img)
                    if is_rgb:
                        assert len(img.shape) == 3, f"Image {image} has shape {img.shape}"
                    else:
                        assert len(img.shape) == 2

            if client_name == "client-1":
                patients = rgb_scanner_patients
            elif client_name == "client-2":
                patients = non_rgb_scanner_patients
            else:
                assert client_name == "val"

            patients.sort()
            print(f"Client {client_name} has patients: {patients}")
            exit()

        else:

            patients = list(patient_to_image.keys())
            patients.sort()
            rng = np.random.default_rng(data_split_seed)
            rng.shuffle(patients)

            if client_name == 'val':
                client_idx = num_clients
            else:
                assert client_name.startswith("client-")
                client_idx = int(client_name.split("-")[1]) - 1

            num_test = int(num_test * len(patients))

            patients_val = patients[:num_test]
            patients = patients[num_test:]

            if client_name == 'val':
                patients = patients_val
            else:
                num_splits = num_clients
                patients = patients[client_idx * len(patients) // num_splits : (client_idx + 1) * len(patients) // num_splits]

            patients.sort()
            print(f"Client {client_name} has {len(patients)} patients: {patients}")

            self.images = []
            for p in patients:
                self.images.extend(patient_to_image[p])


        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5,std=0.5) #normalize to [-1, 1]
        ])

    
    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = np.load(os.path.join(self.data_path, "ARRAY_FORMAT", img_name + ".npy"), allow_pickle=True)
        img = img.item()
        segmentation = img['structures']
        img = img['image']
        img = Image.fromarray(img)
        img = img.crop((100, 0, 900, img.height))
        img = self.transform(img)
        img = img[0:1]

        seg_liver = segmentation['liver']
        label = torch.from_numpy(seg_liver[:, 100:900])
        label = F.interpolate(label[None, None, :, :].float(), size=(64, 64), mode='nearest')[0, 0]
        label = einops.rearrange(label, 'h w -> 1 h w')

        return img, label


def FetalAbdominal_loading(client_name, num_clients, num_test:float, data_split_seed: int, root_dir=None, batch_size=1, split_by_vendor = False, shuffle=True):

    root_dir = get_root_path()

    ds = FetalAbdominalDataset(client_name, num_clients, num_test, data_split_seed, root_dir, split_by_vendor)
    ldr = DataLoader(ds, batch_size = batch_size, shuffle=shuffle)

    return ldr
