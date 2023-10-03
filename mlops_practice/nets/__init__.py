import collections

import torch
from sklearn.datasets import load_digits
from torch import nn
from uniplot import plot


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConvPoolBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)

        return x


class MultiLabelClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvPoolBlock(1, 2)
        self.conv2 = ConvPoolBlock(2, 2)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(8, 10)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.head(x)
        x = self.softmax(x)

        return x
