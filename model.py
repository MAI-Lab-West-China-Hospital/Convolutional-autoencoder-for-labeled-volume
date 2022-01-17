#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 02:47:03 2021

@author: sun
"""

from torch import nn
import torch
from torchsummary import summary


class Encoder(nn.Module):

    def __init__(self, z=354):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, stride=2, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=64, out_channels=128, stride=2, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=128, out_channels=256, stride=2, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            # nn.MaxPool3d(kernel_size=2, stride=2)
            nn.Conv3d(in_channels=256, out_channels=512, stride=2, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2)
        )
        self.encoder_lin = nn.Linear(512 * 6 * 8 * 6, z)

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = torch.flatten(x, 1)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    def __init__(self, z):
        super().__init__()

        self.decoder_lin = nn.Linear(z, 512 * 6 * 8 * 6)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 5, kernel_size=4, stride=2, padding=1),
            nn.Softmax(dim=1)
        )

        # self.unflatten = nn.Unflatten(1, (256,6,8,6))

    def forward(self, x):
        x = self.decoder_lin(x)
        x = x.view(-1, 512, 6, 8, 6)
        x = self.decoder_cnn(x)

        return x


if __name__ == '__main__':

    model = Encoder(354)
    summary(model, (1, 96, 128, 96), device='cpu')
