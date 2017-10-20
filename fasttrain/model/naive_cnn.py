import torch.nn as nn
import torch.nn.functional as F


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUnit, self).__init__()
        self.__conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.__bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.__bn(self.__conv(x)))


class NaiveCNN(nn.Module):
    def __init__(self):
        super(NaiveCNN, self).__init__()

        self.__conv1_1 = ConvUnit(3, 64)
        self.__conv1_2 = ConvUnit(64, 64)
        self.__conv1_3 = ConvUnit(64, 64)

        self.__conv2_1 = ConvUnit(64, 128)
        self.__conv2_2 = ConvUnit(128, 128)
        self.__conv2_3 = ConvUnit(128, 128)

        self.__conv3_1 = ConvUnit(128, 256)
        self.__conv3_2 = ConvUnit(256, 256)
        self.__conv3_3 = ConvUnit(256, 256)

        self.__conv4_1 = ConvUnit(256, 512)
        self.__conv4_2 = ConvUnit(512, 512)
        self.__conv4_3 = ConvUnit(512, 512)

        self.__fc1 = nn.Linear(2048, 256)
        self.__fc2 = nn.Linear(256, 10)

    def forward(self, x):
        l1_1 = self.__conv1_1(x)
        l1_2 = self.__conv1_2(l1_1)
        l1_3 = self.__conv1_3(l1_2)
        l1 = F.max_pool2d(l1_3, 2)

        l2_1 = self.__conv2_1(l1)
        l2_2 = self.__conv2_2(l2_1)
        l2_3 = self.__conv2_3(l2_2)
        l2 = F.max_pool2d(l2_3, 2)

        l3_1 = self.__conv3_1(l2)
        l3_2 = self.__conv3_2(l3_1)
        l3_3 = self.__conv3_3(l3_2)
        l3 = F.max_pool2d(l3_3, 2)

        l4_1 = self.__conv4_1(l3)
        l4_2 = self.__conv4_2(l4_1)
        l4_3 = self.__conv4_3(l4_2)
        l4 = F.max_pool2d(l4_3, 2)

        flat = l4.view(l4.size()[0], -1)

        fc1 = F.relu(self.__fc1(flat))
        fc2 = F.relu(self.__fc2(fc1))

        return fc2
