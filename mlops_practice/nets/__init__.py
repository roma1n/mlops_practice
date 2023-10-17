from torch import nn


class ConvBlock(nn.Module):
    "Implements Conv->BatchNorm->Relu chain"

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
    "Implements Conv->MaxPool chain"

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)

        return x


class MultiLabelClassifier(nn.Module):
    "Image classifier for 8x8 images and 10 classes"

    def __init__(self, p_dropout=0.2):
        super().__init__()

        self.conv1 = ConvPoolBlock(1, 8)
        self.conv2 = ConvPoolBlock(8, 16)
        self.conv3 = ConvPoolBlock(16, 16)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(144, 10)
        self.dropout = nn.Dropout1d()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.head(x)
        x = self.dropout(x)
        x = self.softmax(x)

        return x
