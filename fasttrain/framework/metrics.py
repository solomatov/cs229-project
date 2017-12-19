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


import collections

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


def accuracy_metric(model, dataset, batch=128, key=None):
    def metric():
        nonlocal batch
        model.train(mode=False)

        loader = DataLoader(dataset, batch_size=batch)

        total_samples = 0
        total_correct = 0

        for batch in loader:
            X, y = Variable(batch[0]), Variable(batch[1])

            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            y_ = model(X)
            _, y_ = torch.max(y_, dim=1)

            total_samples += len(y)
            sum = torch.sum(torch.eq(y_, y).type(torch.FloatTensor))
            total_correct += sum.data[0]

        return {'accuracy': total_correct / total_samples}

    return metric


def loss_metric(model, dataset, loss, key=None):
    def metric(y, y_):
        loss_value = loss(y_.float(), y.long())
        return {key or 'loss': loss_value.data[0]}

    return prediction_metric(model, dataset, metric)


def prediction_metric(model, dataset, pred_metric):
    def metric():
        model.train(mode=False)
        loader = DataLoader(dataset, batch_size=len(dataset))
        for batch in loader:
            X, y = Variable(batch[0]), Variable(batch[1])

            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            y_ = model(X)
            return pred_metric(y, y_)

    return metric


def union_metric(*args):
    def metric():
        result = collections.OrderedDict()
        for arg in args:
            result.update(arg())
        return result

    return metric
