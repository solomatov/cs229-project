from datetime import datetime

import torch.optim as optim

from fasttrain import Runner
from fasttrain.data import load_cifar10, SublistDataset
from fasttrain.model import ResNetCIFAR


def train_scheduled(n):
    net = ResNetCIFAR(n)
    batch_size = 128

    print('Training {}'.format(type(net).__name__))
    print('Batch size = {}'.format(batch_size))

    start_time = datetime.now()

    train = load_cifar10(train=True)
    all_test = load_cifar10(train=False)

    test = SublistDataset(all_test, 1000, 10000)
    dev = SublistDataset(all_test, 0, 1000)

    runner = Runner(net, train, dev, batch_size=batch_size, use_all_gpus=False)

    wd = 0.0001
    momentum = 0.9

    runner.run(epochs=80, opt_factory=lambda p: optim.SGD(p, lr=0.1, weight_decay=wd, momentum=momentum))
    runner.run(epochs=80, opt_factory=lambda p: optim.SGD(p, lr=0.01, weight_decay=wd, momentum=momentum))
    runner.run(epochs=40, opt_factory=lambda p: optim.SGD(p, lr=0.001, weight_decay=wd, momentum=momentum))

    train_acc = runner.evaluate(all_test)
    print('Test accuracy: {}'.format(train_acc))
    test_acc = runner.evaluate(train)
    print('Train accuracy: {}'.format(test_acc))

    print('It took {} s to train'.format(datetime.now() - start_time))