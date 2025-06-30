import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.net = models.resnet18(pretrained=pretrained)
        self.net.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2, padding=3, bias=False)
        self.net.fc = nn.Linear(self.net.fc.in_features, 2)

    def forward(self, x):
        return self.net(x)


