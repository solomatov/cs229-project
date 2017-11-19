import collections
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import visdom

from fasttrain.data import load_cifar10, SublistDataset
from fasttrain.framework import accuracy_metric, union_metric, loss_metric


def train_on_cifar(model, schedule, batch_size=128):
    train = load_cifar10(train=True)
    all_test = load_cifar10(train=False)

    test = SublistDataset(all_test, 1000, 10000)
    dev = SublistDataset(all_test, 0, 1000)
    dev_small = SublistDataset(all_test, 0, 200)

    loss = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()

    train_loader = DataLoader(train, batch_size=batch_size, num_workers=2)

    metrics = union_metric(
        accuracy_metric(model, dev_small),
        loss_metric(model, dev_small, loss)
    )

    epoch_counter = 0
    progress = tqdm(total=schedule.total_duration() * len(train_loader), ncols=120)
    vis = visdom.Visdom()
    visdom_win = None

    def on_epoch_start(step, epoch):
        nonlocal epoch_counter, visdom_win, metrics

        postfix = collections.OrderedDict()
        postfix['step'] = f"{step['name']}"
        postfix['epoch'] = f"{epoch}/{step['duration']}"
        model.train(False)

        epoch_metrics = metrics()
        postfix.update(epoch_metrics)
        progress.set_postfix(ordered_dict=postfix)

        epoch_counter += 1
        acc = epoch_metrics['accuracy']

        if not visdom_win:
            visdom_win = vis.line(np.array([acc]), np.array([0]), opts=dict(title='Accuracy'))
        else:
            vis.line(np.array([acc]), np.array([epoch_counter]), win=visdom_win, update='append')

    def on_step():
        progress.update(1)

    schedule.train(model, loss, train=train_loader, dev=dev, on_step=on_step, on_epoch_start=on_epoch_start)
