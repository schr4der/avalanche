import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet3(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()

        # Encoder (Downsampling)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.enc5 = DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(2)

        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)  # (B, 64, 1500, 1500)
        x2 = self.enc2(self.pool(x1))  # (B, 128, 750, 750)
        x3 = self.enc3(self.pool(x2))  # (B, 256, 375, 375)
        x4 = self.enc4(self.pool(x3))  # (B, 512, 187, 187)
        x5 = self.enc5(self.pool(x4))  # (B, 1024, 93, 93)

        # Decoder
        x = self.up1(x5)
        x = self.dec1(torch.cat([x, x4], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x3], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, x2], dim=1))

        x = self.up4(x)
        x = self.dec4(torch.cat([x, x1], dim=1))

        return self.final(x)