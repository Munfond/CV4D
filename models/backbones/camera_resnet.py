"""
Camera Backbone: Trích xuất đặc trưng 6 camera góc rộng sử dụng ResNet + FPN
"""
import torch
import torch.nn as nn
import torchvision.models as models

class CameraBackbone(nn.Module):
    def __init__(self, out_channels=256, pretrained=True):
        super().__init__()
        # Sử dụng ResNet-18 hoặc ResNet-50 (ResNet-18 nhẹ hơn cho Kaggle T4)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512
        
        # FPN 1x1 convs
        self.latlayer1 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.latlayer2 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: [B, N_cams, 3, H, W] -> [B*N_cams, 3, H, W]
        """
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # Top-down FPN
        p5 = self.latlayer1(c5)
        p4 = self.latlayer2(c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p4 = self.smooth(p4)  # [B*N, out_channels, H/16, W/16]
        
        _, C_out, H_out, W_out = p4.shape
        return p4.view(B, N, C_out, H_out, W_out)
