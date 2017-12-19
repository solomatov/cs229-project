"""
   Copyright 2017 JetBrains, s.r.o

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from fasttrain.framework import TrainSchedule
from fasttrain.yellowfin import YFOptimizer

import torch.optim as optim


STAGE_1_EPOCHS = 80
STAGE_2_EPOCHS = 80
STAGE_3_EPOCHS = 40

TOTAL_EPOCHS=STAGE_1_EPOCHS + STAGE_2_EPOCHS + STAGE_3_EPOCHS


def resnet_paper_schedule(batch_size=128, base_lr=0.1, scale=1.0, yellow_fin=False):
    schedule = TrainSchedule()

    wd = 0.0001
    momentum = 0.9

    def new_optim(p, lr):
        if yellow_fin:
            return YFOptimizer(p, lr=lr, weight_decay=wd)
        else:
            return optim.SGD(p, lr=lr, weight_decay=wd, momentum=momentum)

    def optim_factory(lr):
        def factory(p):
            return new_optim(p, lr)

        return factory

    scale_factor = batch_size / 128

    warmup_step = 2
    for i in range(1, int(scale_factor - 1), warmup_step):
        schedule.add_step(factory=optim_factory(base_lr * i), name=f'Warmup {i / scale_factor:.2f}', duration=1)

    lr = base_lr * scale_factor

    schedule.add_step(factory=optim_factory(lr), name='P1', duration=int(STAGE_1_EPOCHS * scale))
    schedule.add_step(factory=optim_factory(lr / 10), name='P2', duration=int(STAGE_2_EPOCHS * scale))
    schedule.add_step(factory=optim_factory(lr / 100), name='P3', duration=int(STAGE_3_EPOCHS * scale))

    return schedule