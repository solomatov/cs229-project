from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.optim as optim
import torch.nn as nn
import torch.tensor as tensor
import torch
import numpy as np

from fasttrain.model import NaiveCNN
from fasttrain.data import load_cifar10


class Runner:
    def __init__(self, net, loader, batch_size=128, use_cuda=torch.cuda.is_available(), loss_fun=nn.CrossEntropyLoss()):
        self.__net = net
        self.__loader = loader
        self.__batch_size = batch_size
        self.__use_cuda = use_cuda
        self.__loss_fun = loss_fun

    def run(self, epochs=1):
        net = self.__net
        if self.__use_cuda:
            net = net.cuda()

        optimizer = optim.Adam(net.parameters(), lr=1e-4)
        loss_fun = nn.CrossEntropyLoss()

        for e in range(epochs):
            print('Epoch = {}'.format(e))
            for i, data in enumerate(self.__loader, 0):
                X_batch, y_batch = self.optimize(data[0]), self.optimize(data[1])
                X_var, y_var = Variable(X_batch), Variable(y_batch)

                optimizer.zero_grad()
                out_ = net(X_var)

                loss = self.__loss_fun(out_, y_var)
                loss.backward()
                optimizer.step()

                print('Loss = {}'.format(loss.data[0]))

    def optimize(self, t):
        if self.__use_cuda:
            return t.cuda()
        return t


has_cuda = torch.cuda.is_available()

net = NaiveCNN()

if has_cuda:
    net = net.cuda()

train = load_cifar10(train=True)
test = load_cifar10(train=False)

train_loader = DataLoader(train, batch_size=128, num_workers=2)
optimizer = optim.Adam(net.parameters(), lr=0.001)
lossfun = nn.CrossEntropyLoss()

runner = Runner(net, train_loader, batch_size=128)
runner.run()