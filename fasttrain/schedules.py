from fasttrain.framework import TrainSchedule

import torch.optim as optim


def resnet_paper_schedule(batch_size=128):
    schedule = TrainSchedule()

    wd = 0.0001
    momentum = 0.9

    def new_optim(p, lr):
        return optim.SGD(p, lr=lr, weight_decay=wd, momentum=momentum)

    def optim_factory(lr):
        def factory(p):
            return new_optim(p, lr)

        return factory

    base_lr = 0.1
    scale_factor = batch_size / 128

    warmup_step = 2
    for i in range(1, int(scale_factor + 1), warmup_step):
        schedule.add_step(factory=optim_factory(base_lr * i), name=f'Warmup {i / scale_factor:.2f}', duration=1)

    lr = base_lr * scale_factor

    schedule.add_step(factory=optim_factory(lr), name='P1', duration=80)
    schedule.add_step(factory=optim_factory(lr / 10), name='P2', duration=80)
    schedule.add_step(factory=optim_factory(lr / 100), name='P3', duration=40)

    return schedule
