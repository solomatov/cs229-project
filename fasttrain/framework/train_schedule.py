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

import torch
from torch.autograd import Variable

import collections


class TrainSchedule:
    def __init__(self):
        self.__steps = []

    def add_step(self, *, factory, name, duration):
        self.__steps.append({'factory': factory, 'name': name, 'duration': duration})

    def total_duration(self):
        result = 0
        for step in self.__steps:
            result += step['duration']
        return result

    def train(self, model, loss, *, train, dev, on_epoch_start=None, on_step=None, half_precision=False):
        for i, step in enumerate(self.__steps, 0):
            name, factory, duration = step['name'], step['factory'], step['duration']
            opt = factory(model.parameters())

            for e in range(duration):
                if on_epoch_start:
                    on_epoch_start(step, e)

                model.train()
                for i, data in enumerate(train, 0):
                    X, y = Variable(data[0]), Variable(data[1])

                    if torch.cuda.is_available():
                        X = X.cuda()
                        y = y.cuda()
                        if half_precision:
                            X = X.half()
                            y = y.half()

                    opt.zero_grad()

                    y_ = model(X)

                    loss_value = loss(y_.float(), y.long())
                    loss_value.backward()

                    opt.step()

                    if on_step:
                        on_step()
