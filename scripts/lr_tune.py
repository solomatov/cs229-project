import numpy as np

from fasttrain.cifar_train import train_on_cifar
from fasttrain.model import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule


def run_experiment(batch_size):
    for lr in np.logspace(1, -6, 8):
        schedule = resnet_paper_schedule(batch_size=batch_size, base_lr=lr)
        sd = 0.5
        model = ResNetCIFAR(n=20, stochastic_depth={'from': sd, 'to': sd})
        train_on_cifar(model, schedule, name=f'ResNet(lr={lr:.2e}, sd={sd})', batch_size=batch_size)


# run_experiment(3072)
# run_experiment(2048)
run_experiment(1024)
# run_experiment(512, multi_gpu=True)
# run_experiment(256)
# run_experiment(128)
