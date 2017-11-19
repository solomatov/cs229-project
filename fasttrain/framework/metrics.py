import collections

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


def accuracy(model, dataset):
    def metric(y, y_):
        _, y_ = torch.max(y_, dim=1)
        result = torch.mean(torch.eq(y_, y).type(torch.FloatTensor)).data[0]
        return {'accuracy': result}

    return prediction_metric(model, dataset, metric)


def loss(model, dataset, loss):
    def metric(y, y_):
        loss_value = loss(y_.float(), y.long())
        return {'loss': loss_value.data[0]}

    return prediction_metric(model, dataset, metric)


def prediction_metric(model, dataset, pred_metric):
    def metric():
        loader = DataLoader(dataset, batch_size=len(dataset))
        for batch in loader:
            X, y = Variable(batch[0]), Variable(batch[1])
            y_ = model(X)
            return pred_metric(y, y_)

    return metric


def union(*args):
    def metric():
        result = collections.OrderedDict()
        for arg in args:
            result.update(arg())
        return result

    return metric