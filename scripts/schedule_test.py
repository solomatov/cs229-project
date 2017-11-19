import torch.optim as optim

from fasttrain.cifar_train import train_on_cifar
from fasttrain.framework import TrainSchedule
from fasttrain.model import ResNetCIFAR

schedule = TrainSchedule()
wd = 0.0001
momentum = 0.9

schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.01, weight_decay=wd, momentum=momentum), name='0.1', duration=80)
schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.001, weight_decay=wd, momentum=momentum), name='0.01', duration=80)
schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.0001, weight_decay=wd, momentum=momentum), name='0.001', duration=40)

model = ResNetCIFAR(n=2)

train_on_cifar(model, schedule)
