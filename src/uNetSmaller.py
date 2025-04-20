import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Match size if needed due to rounding
        if x1.size()[2:] != x2.size()[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class UNet_Modified(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels=16):
        super().__init__()

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2  # bottleneck

        self.down1 = DownSample(in_channels, c1)
        self.down2 = DownSample(c1, c2)
        self.down3 = DownSample(c2, c3)
        self.down4 = DownSample(c3, c4)

        self.bottleneck = DoubleConv(c4, c5, dilation=2)

        self.up1 = UpSample(c5, c4)
        self.up2 = UpSample(c4, c3)
        self.up3 = UpSample(c3, c2)
        self.up4 = UpSample(c2, c1)

        self.out = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        b = self.bottleneck(d4)

        u1 = self.up1(b, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        out = self.out(u4)

        # Upsample output to match input resolution
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out
