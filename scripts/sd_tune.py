import numpy as np

from fasttrain.cifar_train import train_on_cifar
from fasttrain.model import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule

schedule = resnet_paper_schedule()

for prob in np.linspace(0.1, 1.0, 10):
    model = ResNetCIFAR(n=20, stochastic_depth={'from': prob, 'to': prob})
    train_on_cifar(model, schedule, name=f'ResNet(sd-p={prob})', batch_size=3072)
