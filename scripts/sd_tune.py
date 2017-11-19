import numpy as np

from fasttrain.cifar_train import train_on_cifar
from fasttrain.model import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule


batch_size = 3072
schedule = resnet_paper_schedule(batch_size=batch_size)

for prob in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print(prob)
    model = ResNetCIFAR(n=20, stochastic_depth={'from': prob, 'to': prob})
    train_on_cifar(model, schedule, name=f'ResNet(sd-p={prob})', batch_size=batch_size)
