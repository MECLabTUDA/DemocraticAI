import sys, os

import click
import torch
from inversion_attacks import GradientInversion_Attack

from PIL import Image
import torchvision.transforms as transforms

import numpy as np
import torch.nn.functional as F
import einops

import matplotlib.pyplot as plt

from utils.root_path import xray_mimic_path, ultrasound_path, results_reconstructions_path, xray_mimic200_path, crc_path
from network.model import get_model
import utils.hyperparameters as hyp



import hashlib

def compute_model_fingerprint(model):
    # Get the model's state_dict
    state_dict = model.state_dict()
    
    # Create a single bytes object from all tensor data
    hash_input = b""
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].cpu().numpy().tobytes()
        hash_input += tensor

    # Compute SHA256 hash
    fingerprint = hashlib.sha256(hash_input).hexdigest()
    return fingerprint

def compute_image_fingerprint(image_tensor: torch.Tensor) -> str:
    # Ensure tensor is on CPU and in a consistent dtype
    tensor = image_tensor.detach().cpu()
    
    # Optional: enforce consistent dtype and shape (e.g., float32, uint8, etc.)
    # tensor = tensor.to(torch.uint8)  # or torch.float32, depending on your needs

    # Convert tensor to bytes
    byte_data = tensor.numpy().tobytes()
    
    # Compute SHA256 hash
    fingerprint = hashlib.sha256(byte_data).hexdigest()
    return fingerprint



def get_seed_from_image(image_tensor: torch.Tensor) -> int:
    # Ensure it's on CPU and detached for hashing
    image_bytes = image_tensor.detach().cpu().numpy().tobytes()
    
    # Hash the bytes
    hash_digest = hashlib.sha256(image_bytes).hexdigest()
    
    # Use part of the hash to create a 32-bit seed
    seed = int(hash_digest[:16], 16) % (2**32)
    return seed


@click.command()
@click.option("--dataset", type=str, required=True, help="Name of the dataset")
@click.option("--case_name", type=str, required=True, help="Name of the case")
@click.option("--model", type=str, required=True, help="Name of the model")
@click.option("--known_label", type=bool, required=True, help="Whether the label is known")
@click.option("--distance_name", type=str, required=True, help="Name of the distance metric")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option('--no_rerun', default=False, type=bool, is_flag=True, help='Do not rerun the job if it already exists', required=False)
def main(dataset, case_name, model, known_label, distance_name, seed, no_rerun):

    exp_name = f"{dataset}/{model}/{known_label}_{distance_name}/{case_name.replace('/', '_')}"
    output_dir = os.path.join(results_reconstructions_path, exp_name)
    if os.path.exists(output_dir) and no_rerun:
        print(f"Output directory {output_dir} already exists. Skipping...")
        return
    os.makedirs(output_dir, exist_ok=True)

    if seed is not None:
        np.random.seed(seed+1)
        torch.manual_seed(seed+22)
        torch.cuda.manual_seed(seed+33)
        torch.cuda.manual_seed_all(seed+4444)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Using seed {seed}")

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    setup = dict(device=device, dtype=torch.float)


    if dataset in ["XRayMimic", "XRayMimic200"]:
        if dataset == "XRayMimic":
            path = xray_mimic_path
        elif dataset == "XRayMimic200":
            path = xray_mimic200_path
        img = Image.open(os.path.join(path, "images", case_name + ".png")).convert('L')
        t = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5,std=0.5) #normalize to [-1, 1]
        ])
        img = t(img).to(device)
        label = Image.open(os.path.join(path, "labels", case_name + ".png"))
        label = np.array(label)
        label = label[:, :, :3]  # Keep only the first three channels
        label[:,:,2] = 0
        label = label.max(axis=2)  # Convert to grayscale
        label = torch.from_numpy(label).to(device)
        label = F.interpolate(label[None, None, :, :].float(), size=(64, 64), mode='nearest')[0, 0].long().to(device)
        label = label.bool().long()
        labels = einops.rearrange(label, 'h w -> 1 1 h w').float()
    elif dataset == "fetalAbdominal":
        data = np.load(os.path.join(ultrasound_path, "ARRAY_FORMAT", f"{case_name}.npy"), allow_pickle=True).item()
        img = Image.fromarray(data['image'])
        img = img.crop((100, 0, 900, img.height))
        t = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5,std=0.5) #normalize to [-1, 1]
        ])
        img = t(img).to(device)
        img = img[0:1]

        label = data["structures"]["liver"]
        label = torch.from_numpy(label[:, 100:900])
        label = F.interpolate(label[None, None, :, :].float(), size=(64, 64), mode='nearest')[0, 0].long().to(device)

        img, label = img, label
        labels = einops.rearrange(label, 'h w -> 1 1 h w').float()
        del data
    elif dataset == "crc":
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor()
            ])
        img = transform(Image.open(os.path.join(crc_path, "NCT-CRC-HE-100K", case_name + ".tif")).convert("RGB")).to(device)
        labels = torch.as_tensor([0], device=device, dtype=torch.long)
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    ground_truth = img.to(**setup).unsqueeze(0)
    shape_img=tuple(ground_truth[0].shape)
    img = ground_truth.clone().detach().to(**setup)
    if dataset != "crc":
        img.mul_(.5).add_(.5).clamp_(0, 1)


    def reinitialize_all_parameters(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                with torch.no_grad():
                    param.uniform_(-0.05, 0.05)

    #print(torch.randint(0, 10000, (1,)).item(), np.random.randint(0, 10000, (1,)).item())

    net = get_model(dataset, model, device=device).to(device)
    
    if hyp.is_multiclass_classification(dataset):
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    net.eval()
    net.zero_grad()

    ground_truth.requires_grad = True
    pred = net(ground_truth)
    loss = loss_fn(pred, labels)
    received_gradients = torch.autograd.grad(loss, net.parameters())
    received_gradients = [cg.detach() for cg in received_gradients]

    ground_truth.requires_grad = False

    attack_params = dict(
                mean_std=(torch.as_tensor([0.5], **setup), torch.as_tensor([0.5], **setup)),
                num_iteration=10_000,#20000,
                lr=0.1,#0.1,
                optimizer_class=torch.optim.Adam,
                log_interval=100,
                distancename=distance_name,
                bn_reg_layers=[],
                group_num=1,
                tv_reg_coef=0,
                l2_reg_coef=0,
                bn_reg_coef=0,
                gc_reg_coef=0,
                lr_decay= True,
                device=device,
                early_stopping=1000,
                save_file_path='./',
                image_prior_model=None,
                ip_bn_reg_layers=[],
                ip_reg_coef=0,
                pp_reg_coef=0,
                patch_size=16,
                ep_reg_coef=0,
                loss_scheduler=False,
                seed=seed,
                group_seed=[None],

                optimize_label=True,
                lossfunc=loss_fn,
                binary_classification=not hyp.is_multiclass_classification(dataset),
            )
    if known_label:
        attack_params['labels'] = labels
    else:
        if dataset in ["crc"]:
            pass
        elif dataset in ["XRayMimic", "XRayMimic200", "fetalAbdominal"]:
            attack_params['labels'] = torch.randn((1, 1, 64, 64), device=device, requires_grad=True, generator=torch.Generator(device=device).manual_seed(get_seed_from_image(-1 * ground_truth[0])))
        else:
            raise ValueError(f"Unsupported dataset {dataset} for unknown label attack")
    attack_params['optimize_label'] = not known_label


    attacker = GradientInversion_Attack(
            net,
            shape_img,
            **attack_params,
            init_x = torch.randn((1,) + (shape_img), requires_grad=True, device=device, generator=torch.Generator(device=device).manual_seed(get_seed_from_image(ground_truth[0]))),
    )
    best_fake_x, best_fake_label, _ = attacker.group_attack(received_gradients, batch_size=1)

    
    plt.clf()

    img_mode = 'RGB' if dataset in ["crc"] else 'L'
    reconstruction = attacker.convert_to_save(best_fake_x[0][0])
    if dataset in ["XRayMimic", "XRayMimic200", "fetalAbdominal"]:
        reconstruction = einops.rearrange(reconstruction, 'h w 1 -> h w')
    elif dataset == "crc":
        pass
    reconstruction = (reconstruction.cpu().numpy() * 255).astype(np.uint8)
    reconstruction = Image.fromarray(reconstruction, mode=img_mode)
    reconstruction.save(f"{output_dir}/reconstruction.png")
    if dataset == "crc":
        with open(f"{output_dir}/recovered_label.txt", "w") as f:
            f.write(f"{torch.argmax(best_fake_label[0][0]).item() == 0}")
    elif dataset in ["XRayMimic", "XRayMimic200", "fetalAbdominal"]:
        if not known_label:
            recovered_label = best_fake_label[0][0].permute(1, 2, 0).detach().sigmoid().cpu().numpy() > 0.5
            recovered_label = einops.rearrange(recovered_label, 'h w 1 -> h w')
            recovered_label = Image.fromarray((recovered_label * 255).astype(np.uint8), mode='L')
            recovered_label.save(f"{output_dir}/recovered_label.png")
    else:
        raise ValueError(f"Unsupported dataset {dataset}")



if __name__ == "__main__":
    main()