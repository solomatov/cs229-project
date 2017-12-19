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

import argparse

from fasttrain import train_on_cifar
from fasttrain.model.resnet import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule

parser = argparse.ArgumentParser(description='Train ResNet on CIFAR10')
parser.add_argument('-n', '--number', type=int, default=20)
parser.add_argument('-b', '--batch-size', type=int, default=128)

args = parser.parse_args()
schedule = resnet_paper_schedule()
n = args.number
train_on_cifar(ResNetCIFAR(n), schedule, name=f'ResNet({n})-BaseLine', batch_size=args.batch_size)