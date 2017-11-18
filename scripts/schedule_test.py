import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from fasttrain.model import NaiveCNN
from fasttrain.framework.train_schedule import TrainSchedule
from fasttrain.data import load_cifar10, SublistDataset

schedule = TrainSchedule()
wd = 0.0001
momentum = 0.9

schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.1, weight_decay=wd, momentum=momentum), name='0.1', duration=80)
schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.01, weight_decay=wd, momentum=momentum), name='0.01', duration=80)
schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.001, weight_decay=wd, momentum=momentum), name='0.001', duration=40)


train = load_cifar10(train=True)
all_test = load_cifar10(train=False)

test = SublistDataset(all_test, 1000, 10000)
dev = SublistDataset(all_test, 0, 1000)

model = NaiveCNN()


train_loader = DataLoader(train, batch_size=128, num_workers=2)

schedule.train(model, nn.CrossEntropyLoss(), train=train_loader, dev=dev)


