import numpy as np

from fasttrain.cifar_train import train_on_cifar
from fasttrain.model import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule

for n in range(50):
    bs = np.random.choice([3072, 2048, 1024])
    lr = 10 ** np.random.uniform(-0.5, -2)
    sd = np.random.uniform(0.1, 1.0)
    schedule = resnet_paper_schedule(batch_size=bs, base_lr=lr)
    model = ResNetCIFAR(n=20, stochastic_depth={'from': sd, 'to': sd})
    train_on_cifar(model, schedule, name=f'ResNet(lr={lr})', batch_size=bs)
