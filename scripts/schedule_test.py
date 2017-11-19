import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from fasttrain.model import ResNetCIFAR
from fasttrain.framework.train_schedule import TrainSchedule
from fasttrain.framework.metrics import accuracy, union, loss
from fasttrain.data import load_cifar10, SublistDataset

schedule = TrainSchedule()
wd = 0.0001
momentum = 0.9

schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.01, weight_decay=wd, momentum=momentum), name='0.1', duration=80)
schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.001, weight_decay=wd, momentum=momentum), name='0.01', duration=80)
schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.0001, weight_decay=wd, momentum=momentum), name='0.001', duration=40)


train = load_cifar10(train=True)
all_test = load_cifar10(train=False)

test = SublistDataset(all_test, 1000, 10000)
dev = SublistDataset(all_test, 0, 1000)
dev_small = SublistDataset(all_test, 0, 200)

model = ResNetCIFAR(n=2)


def evaluator():
    loader = DataLoader(dev_small, batch_size=len(dev_small))

    for batch in loader:
        X, y = Variable(batch[0]), Variable(batch[1])
        y_ = model(X)
        _, y_ = torch.max(y_, dim=1)
        result = torch.mean(torch.eq(y_, y).type(torch.FloatTensor)).data[0]
        return {'accuracy': result}

train_loader = DataLoader(train, batch_size=128, num_workers=2)

schedule.train(model, nn.CrossEntropyLoss(), train=train_loader, dev=dev, metrics=union(
    accuracy(model, dev_small),
    loss(model, dev_small, nn.CrossEntropyLoss())
))


