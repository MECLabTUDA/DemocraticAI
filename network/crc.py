import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from network.nca_models import ClassificationMaxNCA
import torchvision.models as torch_classifiers
from vit_pytorch import ViT

class EffNet(nn.Module):
    """Baseline model
    We use the EfficientNets architecture that many participants in the ISIC
    competition have identified to work best.
    See here the [reference paper](https://arxiv.org/abs/1905.11946)
    Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
    [pytorch reimplementation of EfficientNets]
    (https://github.com/lukemelas/EfficientNet-PyTorch).
    """

    def __init__(self, pretrained=True, arch_name="efficientnet-b0"):
        super(EffNet, self).__init__()
        self.pretrained = pretrained
        self.base_model = (
            EfficientNet.from_pretrained(arch_name)
            if pretrained
            else EfficientNet.from_name(arch_name)
        )
        # self.base_model=torchvision.models.efficientnet_v2_s(pretrained=pretrained)
        nftrs = self.base_model._fc.in_features
        print("Number of features output by EfficientNet", nftrs)
        self.base_model._fc = nn.Linear(nftrs, 9)

    def forward(self, image):
        out = self.base_model(image)
        return out
    

class DenseNetCRC(nn.Module):
    def __init__(self, num_classes=9):
        super(DenseNetCRC, self).__init__()
        # Load the pre-trained DenseNet-121 model
        self.base_model = torch_classifiers.densenet121()

        # Get the number of input features for the classifier
        num_features = self.base_model.classifier.in_features

        # Replace the classifier with a custom fully connected layer
        self.base_model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Forward pass through the base model
        return self.base_model(x)

class CRCClassificationMaxNCA(ClassificationMaxNCA): 
    def __init__(self):
        super().__init__(input_shape=(96, 96), 
                         num_classes=9, 
                         num_input_channels=3, 
                         fire_rate=0.5,
                         num_steps=[40,20], 
                         hidden_size=64, 
                         conv_sizes=[7,3], 
                         bn=True)

class CRCClassificationMaxNCANoBN(ClassificationMaxNCA): 
    def __init__(self):
        super().__init__(input_shape=(96, 96), 
                         num_classes=9, 
                         num_input_channels=3, 
                         fire_rate=0.5,
                         num_steps=[40,20], 
                         hidden_size=64, 
                         conv_sizes=[7,3], 
                         bn=False)

class DeterministicCRCClassificationMaxNCANoBN(ClassificationMaxNCA):
    def __init__(self):
        super().__init__(input_shape=(96, 96), 
                         num_classes=9, 
                         num_input_channels=3, 
                         fire_rate=1.0,
                         num_steps=[40,20], 
                         hidden_size=64, 
                         conv_sizes=[7,3], 
                         bn=False)

class CTCViT(ViT):
    def __init__(self):
        super().__init__(
            image_size=96,
            patch_size=16,
            num_classes=9,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
            emb_dropout=0,
            channels=3
        )

class CTCViT4SGD(CTCViT):
    pass


def get_crc_model(model_type: str, device="cpu") -> nn.Module:
    if model_type == "maxmednca":
        return CRCClassificationMaxNCA()
    elif model_type == "maxmednca_nobn":
        return CRCClassificationMaxNCANoBN()
    elif model_type == "deterministic_maxmednca_nobn":
        return DeterministicCRCClassificationMaxNCANoBN()
    elif model_type == "vit":
        return CTCViT()
    elif model_type == "vit4sgd":
        return CTCViT4SGD()
    elif model_type == "effnet":
        return EffNet()
    elif model_type.lower() == 'densenet':
        return DenseNetCRC()
    else:
        raise ValueError(f'Unknown model: {model_type}')