from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.optim as optim
import torch.nn as nn
import torch.tensor as tensor
import torch
import numpy as np

from fasttrain.model import NaiveCNN
from fasttrain.data import load_cifar10

net = NaiveCNN()

train = load_cifar10(train=True)
test = load_cifar10(train=False)

train_loader = DataLoader(train, batch_size=128, num_workers=2)
optimizer = optim.Adam(net.parameters(), lr=0.001)
lossfun = nn.CrossEntropyLoss()

for e in range(10):
    print('Epoch={0}'.format(e))

    for i, data in enumerate(train_loader, 0):
        X, y = data
        inp, out = Variable(X), Variable(y)

        optimizer.zero_grad()
        out_ = net(inp)
        loss = lossfun(out_, out)
        loss.backward()
        optimizer.step()
        print('Loss = {}'.format(loss.data[0]))

        print(np.mean(out_.data.numpy().argmax(axis=1) == out.data.numpy()))

