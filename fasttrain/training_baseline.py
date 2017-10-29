from datetime import datetime

import torch.optim as optim

from fasttrain import Runner, VisdomReporter
from fasttrain.data import load_cifar10, SublistDataset
from fasttrain.model import ResNetCIFAR


def train_scheduled_baseline(n, batch_size=128):
    net = ResNetCIFAR(n)

    model_name = type(net).__name__
    print('Training {}'.format(model_name))
    print('N = {}'.format(n))
    print('Batch size = {}'.format(batch_size))

    start_time = datetime.now()

    train = load_cifar10(train=True)
    all_test = load_cifar10(train=False)

    test = SublistDataset(all_test, 1000, 10000)
    dev = SublistDataset(all_test, 0, 1000)

    runner = Runner(net, train, dev, batch_size=batch_size)
    runner.on_epoch(VisdomReporter('{} batch={}'.format(model_name, batch_size)))

    wd = 0.0001
    momentum = 0.9

    runner.run(lambda p: optim.SGD(p, lr=0.1, weight_decay=wd, momentum=momentum), epochs=80)
    runner.run(lambda p: optim.SGD(p, lr=0.01, weight_decay=wd, momentum=momentum), epochs=80)
    runner.run(lambda p: optim.SGD(p, lr=0.001, weight_decay=wd, momentum=momentum), epochs=40)

    train_acc = runner.evaluate(all_test)
    print('Test accuracy: {}'.format(train_acc))
    test_acc = runner.evaluate(train)
    print('Train accuracy: {}'.format(test_acc))

    print('It took {} s to train'.format(datetime.now() - start_time))