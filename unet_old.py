import torch
import torch.nn as nn
import torch.nn.functional as F


def downconv(in_channels, out_channels, kernel_size):
    padding = int(kernel_size / 2)
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
    )


def downsamp(channels):
    return nn.Sequential(
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(channels)
    )


def up(in_channels, out_channels, kernel_size):
    padding = int(kernel_size / 2)
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.Upsample(scale_factor=2)
    )
# %%


class Model(nn.Module):
    def __init__(self, in_channels=3, channels=128, kernel_size=3, dropout=0.):
        super(Model, self).__init__()
        self.dropout = dropout
        self.downconv1 = downconv(in_channels, channels, kernel_size)
        self.downsamp1 = downsamp(channels)
        self.downconv2 = downconv(channels, 2 * channels, kernel_size)
        self.downsamp2 = downsamp(2 * channels)
        self.downconv3 = downconv(2 * channels, 4 * channels, kernel_size)
        self.downsamp3 = downsamp(4 * channels)
        self.up1 = up(4 * channels, 4 * channels, kernel_size)
        self.up2 = up(8 * channels, 2 * channels, kernel_size)
        self.up3 = up(4 * channels, channels, kernel_size)
        padding = int(kernel_size / 2)
        self.last = nn.Sequential(
            nn.Conv2d(in_channels=2 * channels, out_channels=channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        d1 = self.downconv1(x)
        ds1 = self.downsamp1(d1)
        d2 = self.downconv2(ds1)
        ds2 = self.downsamp2(d2)
        d3 = self.downconv3(ds2)
        ds3 = self.downsamp3(d3)
        u = self.up1(ds3)
        u = torch.cat((d3, u), dim=1)
        u = nn.Dropout2d(self.dropout)(u)
        u = self.up2(u)
        u = torch.cat((d2, u), dim=1)
        u = nn.Dropout2d(self.dropout)(u)
        u = self.up3(u)
        u = torch.cat((d1, u), dim=1)
        u = nn.Dropout2d(self.dropout)(u)
        u = self.last(u)
        u = u.reshape(-1, res[1], res[0])
        return u
