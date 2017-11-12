import random
import torch.nn as nn
import torch.nn.functional as F


def make_conv(channels):
    return nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=1)


class SimpleBlock(nn.Module):
    def __init__(self, channels, pre_activated=False, prob=1.0):
        super(SimpleBlock, self).__init__()
        self.conv1 = make_conv(channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = make_conv(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pre_activated = pre_activated
        self.prob = prob

    def forward(self, x):
        if self.training and random.random() > self.prob:
            return x

        if self.pre_activated:
            c1 = self.conv1(F.relu(self.bn1(x)))
            c2 = self.conv2(F.relu(self.bn2(c1)))
        else:
            c1 = F.relu(self.bn1(self.conv1(x)))
            c2 = self.bn2(self.conv2(c1))

        if not self.training:
            c2 = self.prob * c2

        if self.pre_activated:
            return x + c2
        else:
            return F.relu(x + c2)


class DownBlock(nn.Module):
    def __init__(self, in_chan, pre_activated=False, prob=1.0):
        super(DownBlock, self).__init__()
        self.out_chan = in_chan * 2
        self.conv1 = nn.Conv2d(in_chan, self.out_chan, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = make_conv(self.out_chan)
        self.bn2 = nn.BatchNorm2d(self.out_chan)
        self.conv_down = nn.Conv2d(in_chan, self.out_chan, kernel_size=(1, 1), stride=2, bias=False)
        self.bn_down = nn.BatchNorm2d(self.out_chan)

        self.bn_down.weight.data.fill_(1)
        self.bn_down.bias.data.fill_(0)

        self.prob = prob
        self.pre_activated = pre_activated

        if self.pre_activated:
            self.bn1 = nn.BatchNorm2d(in_chan)
        else:
            self.bn1 = nn.BatchNorm2d(self.out_chan)

    def forward(self, x):
        if self.pre_activated:
            if self.training and random.random() > self.prob:
                return self.bn_down(self.conv_down(x))
            c1 = self.conv1(F.relu(self.bn1(x)))
            c2 = self.conv2(F.relu(self.bn2(c1)))
            down = self.bn_down(self.conv_down(x))
        else:
            if self.training and random.random() > self.prob:
                return F.relu(self.bn_down(self.conv_down(x)))
            c1 = F.relu(self.bn1(self.conv1(x)))
            c2 = self.bn2(self.conv2(c1))
            down = self.bn_down(self.conv_down(x))

        if not self.training:
            c2 = self.prob * c2

        if self.pre_activated:
            return down + c2
        else:
            return F.relu(down + c2)


class ResNetCIFAR(nn.Module):
    def __init__(self, n, pre_activated=False, stochastic_depth=False):
        super(ResNetCIFAR, self).__init__()

        self.pre_activated = pre_activated

        self.conv1 = nn.Conv2d(3, 16, (3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        if stochastic_depth:
            prob = 0.5
        else:
            prob = 1.0

        start_prob = 1.0
        total_layers = 3 * n
        layers = 0

        def layer_prob():
            return prob + (start_prob - prob) * (total_layers - layers) / total_layers

        self.seq32_32 = nn.Sequential()
        for i in range(n):
            self.seq32_32.add_module(str(i), SimpleBlock(16, prob=layer_prob()))
            layers += 1

        self.seq16_16 = nn.Sequential()
        for i in range(n):
            if i == 0:
                self.seq16_16.add_module(str(i), DownBlock(16, prob=layer_prob()))
            else:
                self.seq16_16.add_module(str(i), SimpleBlock(32, prob=layer_prob()))
            layers += 1

        self.seq8_8 = nn.Sequential()
        for i in range(n):
            if i == 0:
                self.seq8_8.add_module(str(i), DownBlock(32, prob=layer_prob()))
            else:
                self.seq8_8.add_module(str(i), SimpleBlock(64, prob=layer_prob()))
            layers += 1

        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        s32_32 = self.seq32_32(x)
        s16_16 = self.seq16_16(s32_32)
        s8_8 = self.seq8_8(s16_16)

        features = F.avg_pool2d(s8_8, (8, 8))
        flat = features.view(features.size()[0], -1)

        return self.fc(flat)