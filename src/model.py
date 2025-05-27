import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)

# DownSample block
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p

# UpSample block
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Resize in case of mismatch
        if x1.size()[2:] != x2.size()[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

# Final UNet model
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)

        self.bottleneck = DoubleConv(128, 256, dilation=2)  # Atrous bottleneck
        
        self.up1 = UpSample(256, 128)
        self.up2 = UpSample(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        d1, p1 = self.down1(x)
        d2, p2 = self.down2(p1)

        b = self.bottleneck(p2)
        
        u1 = self.up1(b, d2)
        u2 = self.up2(u1, d1)

        out = self.out(u2)

        return out
