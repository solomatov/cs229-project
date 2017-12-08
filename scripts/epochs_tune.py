import numpy as np

from fasttrain.cifar_train import train_on_cifar
from fasttrain.model import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule


def run_experiment(batch_size):
    for e in range(10, 210, 10):
        schedule = resnet_paper_schedule(batch_size=batch_size, scale=e / 200)
        sd = 0.5
        model = ResNetCIFAR(n=20, stochastic_depth={'from': sd, 'to': sd})
        train_on_cifar(model, schedule, name=f'ResNet(sd={sd}, epochs={e})', batch_size=batch_size)


# run_experiment(3072)
# run_experiment(2048)
run_experiment(1024)
# run_experiment(512, multi_gpu=True)
# run_experiment(256)
# run_experiment(128)
