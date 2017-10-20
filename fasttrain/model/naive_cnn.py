import torch.nn as nn
import torch.nn.functional as F


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConvSeq(nn.Module):
    def __init__(self, n, in_channels, out_channels):
        super(ConvSeq, self).__init__()

        self.convs = []
        prev = in_channels
        for i in range(n):
            self.convs.append(ConvUnit(prev, out_channels))
            prev = out_channels

    def forward(self, x):
        result = x
        for c in self.convs:
            result = c(result)
        return result

class NaiveCNN(nn.Module):
    def __init__(self):
        super(NaiveCNN, self).__init__()

        self.conv1 = ConvSeq(3, 3, 64)
        self.conv2 = ConvSeq(3, 64, 128)
        self.conv3 = ConvSeq(3, 128, 256)
        self.conv4 = ConvSeq(3, 256, 512)

        self.fc1 = nn.Linear(2048, 64)
        self.__fc2 = nn.Linear(64, 10)

    def forward(self, x):
        l1 = F.max_pool2d(self.conv1(x), 2)
        l2 = F.max_pool2d(self.conv2(l1), 2)
        l3 = F.max_pool2d(self.conv3(l2), 2)
        l4 = F.max_pool2d(self.conv4(l3), 2)

        flat = l4.view(l4.size()[0], -1)

        fc1 = F.relu(self.fc1(flat))
        fc2 = self.__fc2(fc1)

        return fc2
