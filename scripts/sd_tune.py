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
    schedule = resnet_paper_schedule(batch_size=batch_size)

    for prob in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(prob)
        model = ResNetCIFAR(n=20, stochastic_depth={'from': prob, 'to': prob})
        train_on_cifar(model, schedule, name=f'ResNet(sd-p={prob})', batch_size=batch_size)


run_experiment(3072)
run_experiment(2048)
run_experiment(1024)
# run_experiment(512)
# run_experiment(256)
# run_experiment(128)

