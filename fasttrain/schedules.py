from fasttrain.framework import TrainSchedule

import torch.optim as optim


def resnet_paper_schedule():
    schedule = TrainSchedule()
    wd = 0.0001
    momentum = 0.9

    schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.01, weight_decay=wd, momentum=momentum), name='P1', duration=80)
    schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.001, weight_decay=wd, momentum=momentum), name='P2', duration=80)
    schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.0001, weight_decay=wd, momentum=momentum), name='P3', duration=40)

    return schedule
