import collections
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel as parallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import visdom

from fasttrain.data import load_cifar10, SublistDataset
from fasttrain.framework import accuracy_metric, union_metric, loss_metric


def train_on_cifar(model, schedule, batch_size=128, name=None, show_test=False, force_multi_gpu=False):
    if name:
        print(f'Training: {name}')
    start_time = datetime.now()

    train = load_cifar10(train=True)
    all_test = load_cifar10(train=False)

    test = SublistDataset(all_test, 1000, 10000)
    dev = SublistDataset(all_test, 0, 1000)
    dev_small = SublistDataset(all_test, 0, 200)

    loss = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        if batch_size >= 1024 or force_multi_gpu:
            model = parallel.DataParallel(model)

    train_loader = DataLoader(train, batch_size=batch_size, num_workers=2)

    metrics = union_metric(
        accuracy_metric(model, dev_small),
        loss_metric(model, dev_small, loss)
    )

    epoch_counter = 0

    # workaround for https://github.com/tqdm/tqdm/issues/481
    tqdm.monitor_interval = 0
    progress = tqdm(total=schedule.total_duration() * len(train_loader), ncols=120)
    vis = visdom.Visdom()
    visdom_win = None

    def on_epoch_start(step, epoch):
        nonlocal epoch_counter, visdom_win, metrics

        postfix = collections.OrderedDict()
        postfix['step'] = f'{step["name"]}'
        postfix['epoch'] = f'{epoch}/{step["duration"]}'
        model.train(False)

        epoch_metrics = metrics()
        postfix.update(epoch_metrics)
        progress.set_postfix(ordered_dict=postfix)

        acc = epoch_metrics['accuracy']

        if not visdom_win:
            visdom_win = vis.line(np.array([acc]), np.array([0]), opts=dict(title=name or 'Accuracy'))
        else:
            vis.line(np.array([acc]), np.array([epoch_counter]), win=visdom_win, update='append')
        epoch_counter += 1

    def on_step():
        progress.update(1)

    schedule.train(model, loss, train=train_loader, dev=dev, on_step=on_step, on_epoch_start=on_epoch_start)
    progress.close()
    total_time = datetime.now() - start_time

    def get_accuracy_on(dataset):
        return accuracy_metric(model, dataset)()['accuracy']

    def print_acc(name, accuracy):
        print(f'{name} accuracy: {accuracy:.3f}')

    dev_accuracy = get_accuracy_on(dev)
    print_acc('Dev', dev_accuracy)
    train_accuracy = get_accuracy_on(train)
    print_acc('Train', train_accuracy)
    if show_test:
        test_accuracy = get_accuracy_on(test)
        print_acc('Test', test_accuracy)
    else:
        test_accuracy = -1.0

    print(f'It took {total_time} to train')

    with open('experiments.txt', mode='a') as f:
        parameters = f'batch_size={batch_size}'
        experiment_data = f'{datetime.now()}: {name}[{parameters}]. ' \
                          f'Time={total_time}. Train={train_accuracy:.3f}. Dev={dev_accuracy:.3f}. Test={test_accuracy:.3f}\n'
        f.write(experiment_data)