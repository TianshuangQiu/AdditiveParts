#!/usr/bin/env python
# coding: utf-8

# Preparation

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import time
import argparse
from tqdm import tqdm

from prettytable import PrettyTable
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, kernel_size=3, activation_fn=nn.ReLU()):
        super().__init__()

        self.max_pooling_2 = nn.MaxPool3d(kernel_size=2)

        self.up_sampling_2 = nn.Upsample(scale_factor=2)

        self.conv64_1_8 = nn.Sequential(
            nn.Conv3d(
                in_channels=1, out_channels=8, kernel_size=kernel_size, padding="same"
            ),
            nn.BatchNorm3d(num_features=8),
            activation_fn,
        )

        self.conv64_8_8 = nn.Sequential(
            nn.Conv3d(
                in_channels=8, out_channels=8, kernel_size=kernel_size, padding="same"
            ),
            nn.BatchNorm3d(num_features=8),
            activation_fn,
        )

        self.conv32_8_32 = nn.Sequential(
            nn.Conv3d(
                in_channels=8, out_channels=32, kernel_size=kernel_size, padding="same"
            ),
            nn.BatchNorm3d(num_features=32),
            activation_fn,
        )

        self.conv32_32_32 = nn.Sequential(
            nn.Conv3d(
                in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"
            ),
            nn.BatchNorm3d(num_features=32),
            activation_fn,
        )

        self.conv16_32_128 = nn.Sequential(
            nn.Conv3d(
                in_channels=32,
                out_channels=128,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm3d(num_features=128),
            activation_fn,
        )

        self.conv16_128_128 = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=128,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm3d(num_features=128),
            activation_fn,
        )

        self.conv8_128_256 = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=256,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm3d(num_features=256),
            activation_fn,
        )

        self.conv8_256_256 = nn.Sequential(
            nn.Conv3d(
                in_channels=256,
                out_channels=256,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm3d(num_features=256),
            activation_fn,
        )

        self.conv16_384_128 = nn.Sequential(
            nn.Conv3d(
                in_channels=384,
                out_channels=128,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm3d(num_features=128),
            activation_fn,
        )

        self.conv32_160_32 = nn.Sequential(
            nn.Conv3d(
                in_channels=160,
                out_channels=32,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm3d(num_features=32),
            activation_fn,
        )

        self.conv64_40_8 = nn.Sequential(
            nn.Conv3d(
                in_channels=40, out_channels=8, kernel_size=kernel_size, padding="same"
            ),
            nn.BatchNorm3d(num_features=8),
            activation_fn,
        )

        self.conv64_8_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=8, out_channels=1, kernel_size=kernel_size, padding="same"
            ),
            activation_fn,
        )

    def forward(self, x):
        x = self.conv64_1_8(x)
        x = self.conv64_8_8(x)
        feature_map_64 = x.detach()
        x = self.max_pooling_2(x)
        x = self.conv32_8_32(x)
        x = self.conv32_32_32(x)
        feature_map_32 = x.detach()
        x = self.max_pooling_2(x)
        x = self.conv16_32_128(x)
        x = self.conv16_128_128(x)
        feature_map_16 = x.detach()
        x = self.max_pooling_2(x)
        x = self.conv8_128_256(x)
        x = self.conv8_256_256(x)
        x = self.up_sampling_2(x)
        x = torch.cat((feature_map_16, x), dim=1)
        x = self.conv16_384_128(x)
        x = self.conv16_128_128(x)
        x = self.up_sampling_2(x)
        x = torch.cat((feature_map_32, x), dim=1)
        x = self.conv32_160_32(x)
        x = self.conv32_32_32(x)
        x = self.up_sampling_2(x)
        x = torch.cat((feature_map_64, x), dim=1)
        x = self.conv64_40_8(x)
        x = self.conv64_8_1(x)
        return x


class ConvNetScalarLabel256(nn.Module):
    arch = "3DCNN_256_stride1_new"

    def __init__(self, kernel_size=3, activation_fn=nn.ReLU()):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(self.create_conv_set(1, 2, kernel_size, activation_fn))
        # self.layers.append(self.create_conv_set(2, 2, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size=2))
        self.layers.append(self.create_conv_set(2, 4, kernel_size, activation_fn))
        # self.layers.append(self.create_conv_set(4, 4, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size=2))
        self.layers.append(self.create_conv_set(4, 8, kernel_size, activation_fn))
        # self.layers.append(self.create_conv_set(8, 8, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size=2))
        self.layers.append(self.create_conv_set(8, 32, kernel_size, activation_fn))
        # self.layers.append(self.create_conv_set(32, 32, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size=2))
        self.layers.append(self.create_conv_set(32, 128, kernel_size, activation_fn))
        # self.layers.append(self.create_conv_set(128, 128, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size=2))
        self.layers.append(self.create_conv_set(128, 256, kernel_size, activation_fn))
        # self.layers.append(self.create_conv_set(256, 256, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size=8))

        self.linear_1 = nn.Linear(256, 16)
        self.linear_2 = nn.Linear(16, 1)

    def create_conv_set(self, in_channels, out_channels, kernel_size, activation_fn):
        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm3d(num_features=out_channels),
            activation_fn,
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.squeeze(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return torch.squeeze(x)


class ConvNetScalarLabel64(nn.Module):
    arch = "3DCNN_64_stride1"

    def __init__(self, kernel_size=3, activation_fn=nn.ReLU()):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(self.create_conv_set(1, 2, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size=2))
        # 32 res here
        self.layers.append(self.create_conv_set(2, 4, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size=2))
        # 16 res here
        self.layers.append(self.create_conv_set(4, 8, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size=2))
        # 8 res here
        self.layers.append(self.create_conv_set(8, 16, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size=2))
        # 4 res here
        self.layers.append(self.create_conv_set(16, 32, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size=4))
        # 1 res here

        self.linear_1 = nn.Linear(32, 8)
        self.linear_2 = nn.Linear(8, 1)

    def create_conv_set(self, in_channels, out_channels, kernel_size, activation_fn):
        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm3d(num_features=out_channels),
            activation_fn,
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.squeeze(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return torch.squeeze(x)


class MLPBaseline64(nn.Module):
    arch = "MLPBaseline64"

    def __init__(self, kernel_size=3, activation_fn=nn.ReLU()):
        super().__init__()
        self.linear1 = nn.Linear(64**3, 8)
        self.linear2 = nn.Linear(8, 1)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = torch.reshape(x, (-1, 64**3))
        x = self.activation_fn(self.linear1(x))
        x = self.activation_fn(self.linear2(x))
        return torch.squeeze(x)
