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

from fasttrain.cifar_train import train_on_cifar
from fasttrain.model import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule

schedule = resnet_paper_schedule()

for n in range(2, 80):
    model = ResNetCIFAR(n=n)
    train_on_cifar(model, schedule, name=f'ResNet({n})', batch_size=128)
