import numpy as np

from fasttrain.cifar_train import train_on_cifar
from fasttrain.model import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule


def run_experiment(batch_size):
    schedule = resnet_paper_schedule(batch_size=batch_size)

    model = ResNetCIFAR(n=20)
    train_on_cifar(model, schedule, name=f'ResNet()', batch_size=batch_size)


run_experiment(3072)
run_experiment(2048)
run_experiment(1024)
run_experiment(512)
run_experiment(256)
run_experiment(128)






