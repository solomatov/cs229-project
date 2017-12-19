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

import numpy as np

from fasttrain.cifar_train import train_on_cifar
from fasttrain.model import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule


def run_experiment(batch_size):
    for lr in np.logspace(-0.5, -1.5, 50):
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
