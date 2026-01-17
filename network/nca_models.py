import torch
import torch.nn as nn
import torch.nn.functional as F

class NCA(nn.Module):
    def __init__(self, num_input_channels=1, fire_rate=0.5, num_steps=16, hidden_size=32, conv_size=3, bn=False):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.fire_rate = fire_rate
        self.num_steps = num_steps
        self.conv = nn.Conv2d(16, 16, kernel_size=conv_size, stride=1, padding='same', groups=16, padding_mode="reflect")
        self.fc0 = nn.Conv2d(2*16, hidden_size, kernel_size=1, stride=1, padding=0, bias=not bn)
        self.fc1 = nn.Conv2d(hidden_size, 16-self.num_input_channels, kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU()
        if bn:
            self.norm = nn.BatchNorm2d(hidden_size, track_running_stats=False)
        else:
            self.norm = nn.Identity()

    def step(self, x_in):
        dx = self.conv(x_in)
        dx = torch.cat([x_in, dx], dim=1)
        dx = self.fc0(dx)
        dx = self.norm(dx)
        dx = self.activation(dx)
        dx = self.fc1(dx)
        if self.fire_rate < 1.0:
            with torch.no_grad():
                mask = torch.zeros((dx.shape[0], 1, dx.shape[2], dx.shape[3]), device=dx.device)
                mask.bernoulli_(self.fire_rate).float()
            dx = dx * mask
        return dx + x_in[:, self.num_input_channels:]


    def forward(self, x):
        input_channels = x[:, :self.num_input_channels]
        for _ in range(self.num_steps):
            x = self.step(x)
            x = torch.concat([input_channels, x], dim=1)
        # END: for
        return x

class StandaloneNCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = NCA()

    def forward(self, x):
        input_channels = x
        x = torch.concat([input_channels, torch.zeros(1,16-self.num_input_channels,32,32, device=input_channels.device)], dim=1)
        x = self.backbone(x)
        return x[:, -4:-1]

    
class MedNCA(nn.Module):
    def __init__(self, input_shape=(32,32), return_full=False, num_input_channels=1, num_classes=1, fire_rate=0.5, num_steps=[16, 16], hidden_size=32, conv_sizes=[3, 3], bn=False):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.nca1 = NCA(num_input_channels=num_input_channels, fire_rate=fire_rate, num_steps=num_steps[0], hidden_size=hidden_size, conv_size=conv_sizes[0], bn=bn)
        self.nca2 = NCA(num_input_channels=num_input_channels, fire_rate=fire_rate, num_steps=num_steps[1], hidden_size=hidden_size, conv_size=conv_sizes[1], bn=bn)
        self.return_full = return_full
        self.input_shape = input_shape
        self.num_classes = num_classes
    def forward(self, x):
        input_channels = x
        x = torch.concat([F.avg_pool2d(input_channels, 2), torch.zeros(x.shape[0],
                                                                       16-self.num_input_channels, 
                                                                       self.input_shape[0] // 2,
                                                                       self.input_shape[0] // 2,
                                                                       device=input_channels.device)], dim=1)
        x = self.nca1(x)[:, self.num_input_channels:]
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.concat([input_channels, x], dim=1)
        x = self.nca2(x)
        if self.return_full:
            return x
        else:
            x = x[:, self.num_input_channels:] # trim input channels
            return x[:, :self.num_classes]  # return only the first num_classes channels
    

class ClassificationNCA(nn.Module):
    def __init__(self, input_shape=(96,96), num_classes=1, num_input_channels=3, fire_rate=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.nca = MedNCA(input_shape=input_shape, return_full=True, num_input_channels=num_input_channels, fire_rate=fire_rate)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.nca(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.fc(x)
        return x
    
class ClassificationMaxNCA(nn.Module):
    def __init__(self, input_shape=(96,96), num_classes=1, num_input_channels=3, fire_rate=0.5, num_steps=[16,16], hidden_size=32, conv_sizes=[3,3], bn=False):
        super().__init__()
        self.num_classes = num_classes
        self.nca = MedNCA(input_shape=input_shape, return_full=True, num_input_channels=num_input_channels, fire_rate=fire_rate, num_steps=num_steps, hidden_size=hidden_size, conv_sizes=conv_sizes, bn=bn)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.nca(x)
        x = torch.amax(x, dim=[2, 3])  # Global max pooling
        x = self.fc(x)
        return x