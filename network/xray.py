from torch import nn
from network import nca_models

from unet import UNet2D
class XRayUNet(UNet2D):
    def __init__(self, in_channels:int=1, out_classes:int=1, padding:bool=True):
        super().__init__(in_channels=in_channels, 
                         out_classes=out_classes,
                         padding=padding)

from network.MedNCA import create_med_nca_xray      

from network.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from network.vit_seg_modeling import VisionTransformer as ViT_seg
class XRayTransUNet(ViT_seg):
    def __init__(self):
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_skip = 3
        config_vit.n_classes = 1
        config_vit.patches.grid = (int(64 / 16), int(64 / 16))
        super().__init__(config_vit, img_size=(64, 64), num_classes=config_vit.n_classes)

class XRayMedNCA(nca_models.MedNCA):
    def __init__(self, fire_rate=0.5):
        super().__init__(input_shape=(64, 64),
                         return_full=False,
                         num_input_channels=1,
                         num_classes=1,
                         fire_rate=fire_rate,
                         num_steps=[40, 20],
                         hidden_size=64,
                         bn=True,
                         conv_sizes=[7, 3])

from network.ultrasound import UltrasoundSegFormer

def get_xray_model(model_type: str, device="cpu") -> nn.Module:
    if model_type.lower() == 'unet':
        model = XRayUNet()
    elif model_type.lower() == 'mednca':
        model = create_med_nca_xray(device)
    elif model_type == "new_mednca":
        model = XRayMedNCA()
    elif model_type == "deterministic_mednca":
        model = XRayMedNCA(fire_rate=1.0)
    elif model_type == "transunet_b16":
        model = XRayTransUNet()
    elif model_type == "segformer":
        model = UltrasoundSegFormer() # same implementation
    else:
        raise ValueError(f'Unknown model: {model_type}')

    return model
