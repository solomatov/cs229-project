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

    def train(self, model, loss, *, train, dev, on_epoch_start=None, on_step=None):
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

                    opt.zero_grad()

                    y_ = model(X)

                    loss_value = loss(y_.float(), y.long())
                    loss_value.backward()

                    opt.step()

                    if on_step:
                        on_step()
