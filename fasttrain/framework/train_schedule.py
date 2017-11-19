from torch.autograd import Variable
from tqdm import tqdm

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

    def train(self, model, loss, *, train, dev, metrics):
        progress = tqdm(total=self.total_duration() * len(train))

        for step in self.__steps:
            name, factory, duration = step['name'], step['factory'], step['duration']
            opt = factory(model.parameters())

            for e in range(duration):
                postfix = collections.OrderedDict()
                postfix['step'] = name
                postfix['epoch'] = f"{e}/{duration}"
                postfix.update(metrics())
                progress.set_postfix(**postfix)

                for i, data in enumerate(train, 0):
                    model.train(False)

                    model.train(True)

                    X, y = Variable(data[0]), Variable(data[1])
                    opt.zero_grad()

                    y_ = model(X)

                    loss_value = loss(y_.float(), y.long())
                    loss_value.backward()

                    opt.step()

                    progress.update(1)