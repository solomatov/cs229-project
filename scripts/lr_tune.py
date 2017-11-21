import numpy as np

from fasttrain.cifar_train import train_on_cifar
from fasttrain.model import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule


def run_experiment(batch_size, multi_gpu=False):
    schedule = resnet_paper_schedule(batch_size=batch_size)

    for lr in [0.5, 0.1, 0.05, 0.01]:
        model = ResNetCIFAR(n=20)
        train_on_cifar(model, schedule, name=f'ResNet(lr={lr}, multi={multi_gpu})', batch_size=batch_size, force_multi_gpu=multi_gpu)


run_experiment(128)
run_experiment(256)
run_experiment(256, multi_gpu=True)
run_experiment(512)
run_experiment(512, multi_gpu=True)
run_experiment(1024)
run_experiment(2048)
run_experiment(3072)




