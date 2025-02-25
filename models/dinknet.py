import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        resnet = models.resnet34(pretrained=True)
        
        # Encoder
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        # Dilated Conv Block
        self.dblock = DilatedBlock(512)
        
        # Decoder
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)
        
        # Final layers
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 3, padding=1)
        )

    def forward(self, x):
        # Encoder
        x = self.initial(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        c = self.dblock(e4)
        
        # Decoder with skip connections
        d4 = self.decoder4(c) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        return torch.sigmoid(self.final(d1))

class DilatedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(channels, channels, 3, dilation=4, padding=4)
        self.conv4 = nn.Conv2d(channels, channels, 3, dilation=8, padding=8)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        return x + x1 + x2 + x3 + x4

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels//4, in_channels//4, 3, stride=2, 
                               padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)