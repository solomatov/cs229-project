import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.optim as optim
from torch.autograd import Variable


class TrainSchedule:
    def __init__(self):
        self.__steps = []

    def add_step(self, *, factory, name, duration):
        self.__steps.append({'factory': factory, 'name': name, 'duration': duration})

    def train(self, model, loss, *, train, dev):
        for step in self.__steps:
            name, factory, duration = step['name'], step['factory'], step['duration']
            opt = factory(model.parameters())
            print(f'step {name}')

            for e in range(duration):
                print(f'epoch {e}')
                model.train()
                for i, data in enumerate(train, 0):
                    print(f'batch {i}')
                    X, y = Variable(data[0]), Variable(data[1])
                    opt.zero_grad()

                    y_ = model(X)

                    loss_value = loss(y_.float(), y.long())
                    loss_value.backward()
                    opt.step()