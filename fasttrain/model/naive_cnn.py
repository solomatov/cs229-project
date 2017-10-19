import torch.nn as nn
import torch.nn.functional as F


class NaiveCNN(nn.Module):
    def __init__(self):
        super(NaiveCNN, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        l1_1 = F.relu(self.conv1_1(x))
        l1_1 = self.conv1_1_bn(l1_1)
        l1_2 = F.relu(self.conv1_2(l1_1))
        l1_2 = self.conv1_2_bn(l1_2)
        l1 = F.max_pool2d(l1_2, 2)

        l2_1 = F.relu(self.conv2_1(l1))
        l2_1 = self.conv2_1_bn(l2_1)
        l2_2 = F.relu(self.conv2_2(l2_1))
        l2_2 = self.conv2_2_bn(l2_2)
        l2 = F.max_pool2d(l2_2, 2)
        
        l3_1 = F.relu(self.conv3_1(l2))
        l3_1 = self.conv3_1_bn(l3_1)
        l3_2 = F.relu(self.conv3_2(l3_1))
        l3_2 = self.conv3_2_bn(l3_2)
        l3 = F.max_pool2d(l3_2, 2)

        l4_1 = F.relu(self.conv4_1(l3))
        l4_1 = self.conv4_1_bn(l4_1)
        l4_2 = F.relu(self.conv4_2(l4_1))
        l4_2 = self.conv4_2_bn(l4_2)
        l4 = F.max_pool2d(l4_2, 2)

        flat = l4.view(l4.size()[0], -1)

        fc1 = F.relu(self.fc1(flat))
        fc2 = F.relu(self.fc2(fc1))
        fc3 = F.relu(self.fc3(fc2))
        fc4 = self.fc4(fc3)

        return fc2
