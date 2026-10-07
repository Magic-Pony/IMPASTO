"""IMPASTO-UNet reconstructed from Appendix B Fig. 2."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(.2, inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(.2, inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ImpastoUNet(nn.Module):
    """Two-channel image encoder + six-dimensional action encoder + U-Net decoder."""
    def __init__(self, action_dim=6, channels=(32, 64, 128, 256),
                 action_embedding_channels=32, dropout=.1):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.enc1 = ConvBlock(2, c1, dropout=dropout)
        self.enc2 = ConvBlock(c1 + action_embedding_channels, c2, dropout=dropout)
        self.enc3 = ConvBlock(c2, c3, dropout=dropout)
        self.enc4 = ConvBlock(c3, c4, dropout=dropout)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_embedding_channels), nn.LeakyReLU(.2, inplace=True),
            nn.Linear(action_embedding_channels, action_embedding_channels), nn.LeakyReLU(.2, inplace=True),
        )
        self.dec3 = DecodeBlock(c4, c3, c3)
        self.dec2 = DecodeBlock(c3, c2, c2)
        self.dec1 = DecodeBlock(c2, c1, c1)
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1), nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(c1, 1, 1), nn.Sigmoid(),
        )

    def forward(self, image, action):
        e1 = self.enc1(image)
        pooled = F.max_pool2d(e1, 2)
        action_map = self.action_encoder(action).unsqueeze(-1).unsqueeze(-1)
        action_map = action_map.expand(-1, -1, pooled.shape[-2], pooled.shape[-1])
        e2 = self.enc2(torch.cat([pooled, action_map], dim=1))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        x = self.dec3(e4, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        x = F.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(x)
