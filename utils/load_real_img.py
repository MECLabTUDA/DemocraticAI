from PIL import Image
import torchvision.transforms as transforms
from utils.root_path import ultrasound_path, xray_mimic200_path, crc_path
import torch, os
import numpy as np
import torch.nn.functional as F


def load_real_img_fetalAbdominal(case_name) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(os.path.join(ultrasound_path, "ARRAY_FORMAT", f"{case_name}.npy"), allow_pickle=True).item()
    img = Image.fromarray(data['image'])
    img = img.crop((100, 0, 900, img.height))
    t = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    img = t(img)
    img = img[0:1]

    label = data["structures"]["liver"]
    label = torch.from_numpy(label[:, 100:900])
    label = F.interpolate(label[None, None, :, :].float(), size=(64, 64), mode='nearest')[0, 0].long()
    return img[0], label


def load_real_img_mimic200(case_name) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(os.path.join(xray_mimic200_path, "images", case_name + ".png")).convert('L')
        t = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        img = t(img)
        label = Image.open(os.path.join(xray_mimic200_path, "labels", case_name + ".png"))
        label = np.array(label)
        label = label[:, :, :3]  # Keep only the first three channels
        label[:,:,2] = 0
        label = label.max(axis=2)  # Convert to grayscale
        label = torch.from_numpy(label)
        label = F.interpolate(label[None, None, :, :].float(), size=(64, 64), mode='nearest')[0, 0].long()
        label = label.bool().long()

        return img[0], label


def load_real_img_crc(case_name) -> tuple[torch.Tensor, torch.Tensor]:
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
        ])
    img = transform(Image.open(os.path.join(crc_path, "NCT-CRC-HE-100K", case_name + ".tif")).convert("RGB"))
    return img, None
        