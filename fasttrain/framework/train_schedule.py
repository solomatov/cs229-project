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

from fasttrain.fp16util import set_grad, copy_in_params

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

        if half_precision:
            param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model.parameters()]
            for param in param_copy:
                param.requires_grad = True
            loss_scale = 256
        else:
            param_copy = list(model.parameters())
            loss_scale = 1

        for i, step in enumerate(self.__steps, 0):
            name, factory, duration = step['name'], step['factory'], step['duration']
            opt = factory(param_copy)

            for e in range(duration):
                if on_epoch_start:
                    on_epoch_start(step, e)

                model.train()
                for i, data in enumerate(train, 0):
                    X, y = Variable(data[0]), Variable(data[1])

                    if torch.cuda.is_available():
                        X = X.cuda()
                        y = y.cuda()

                    y_ = model(X)

                    loss_value = loss(y_.float(), y.long())
                    loss_value = loss_value * loss_scale

                    if half_precision:
                        model.zero_grad()
                        loss_value.backward()
                        set_grad(param_copy, list(model.parameters()))

                        if loss_scale != 1:
                            for param in param_copy:
                                param.grad.data = param.grad.data / loss_scale

                        opt.step()
                        copy_in_params(model, param_copy)
                        torch.cuda.synchronize()
                    else:
                        opt.zero_grad()
                        loss_value.backward()
                        opt.step()

                    if on_step:
                        on_step()
