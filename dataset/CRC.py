import os
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from utils.root_path import crc_path


class CRC(Dataset):
    def __init__(
        self,
        client_name,
        num_clients: int, 
        num_test: float, 
        data_split_seed: int, 
        root_path=crc_path,
    ):
        """
        Args:
            csv_path (str): Path to dataset CSV
            client_name (int): Client ID to load (0-4 for training, 5 for test)
            transform (callable, optional): Image transformations
            return_path (bool): If True, also return image path
        """
        assert num_clients == 5
        assert data_split_seed == 42, "Hardcoded data split used"
        assert num_test == 1.0, "Hardcoded data split used"
        self.root_path = root_path
        self.df = pd.read_csv(os.path.join(root_path, "fednca_meta.csv"))
        if client_name == 'val':
            client_name = 5
        else:
            client_name = int(client_name.split("-")[-1]) - 1
            assert client_name < 5, "Client name must be client-0 to client-4 for training, or 'val' for test"
            
        self.df = self.df[self.df["client_id"] == client_name].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No samples found for client_name={client_name}")

        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.root_path, row["image"])
        label = int(row["label"])

        # Load .tif image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def CRC_loading(client_name: str, num_clients: int, num_test:float, data_split_seed: int, batch_size=1, shuffle=True):

    ds  = CRC(client_name, num_clients, num_test, data_split_seed)
    ldr = DataLoader(ds, batch_size = batch_size, shuffle=shuffle)
    return ldr
