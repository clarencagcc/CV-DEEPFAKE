import torch
import torch.nn as nn
from torchvision import transforms
from timm.models import xception

# Define the Channel Attention Module (CAM)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):  # Fix typo: "__init__"
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# Define the modified Xception model
class Xception_SingleCAM(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, dropout_prob=0.4):  # Fix typo: "__init__"
        super(Xception_SingleCAM, self).__init__()
        self.xception = xception(pretrained=pretrained)
        self.xception.conv1 = nn.Conv2d(
            5, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.feature_extractor = nn.Sequential(*list(self.xception.children())[:-2])
        self.channel_attention = ChannelAttention(in_channels=2048)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.channel_attention(x)
        x = x.mean([2, 3])
        x = self.dropout(x)
        x = self.fc(x)
        return x

