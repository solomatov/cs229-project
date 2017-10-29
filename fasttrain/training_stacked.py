from datetime import datetime

import torch.optim as optim

from fasttrain import Runner, VisdomReporter
from fasttrain.data import load_cifar10, SublistDataset
from fasttrain.model import ResNetCIFAR


def train_stacked(n, batch_size=128, half=False):
    net = ResNetCIFAR(n)

    if half:
        net = net.cuda().half()

    model_name = type(net).__name__
    print('Training {}'.format(model_name))
    print('N = {}'.format(n))
    print('Batch size = {}'.format(batch_size))

    start_time = datetime.now()

    train = load_cifar10(train=True)
    all_test = load_cifar10(train=False)

    test = SublistDataset(all_test, 1000, 10000)
    dev = SublistDataset(all_test, 0, 1000)

    runner = Runner(net, train, dev, batch_size=batch_size, half=half)
    runner.on_epoch(VisdomReporter('{} batch={}'.format(model_name, batch_size)))

    wd = 0.0001
    momentum = 0.9

    def optim_factory(lr):
        return lambda p: optim.SGD(p, lr, weight_decay=wd, momentum=momentum)

    lr_scaling = batch_size / 128
    for i in range(int(lr_scaling + 1)):
        runner.run(optim_factory(i * 0.1), epochs=1)

    runner.run(optim_factory(0.1 * lr_scaling), epochs=80)
    runner.run(optim_factory(0.01 * lr_scaling), epochs=80)
    runner.run(optim_factory(0.001 * lr_scaling), epochs=40)

    train_acc = runner.evaluate(all_test)
    print('Test accuracy: {}'.format(train_acc))
    test_acc = runner.evaluate(train)
    print('Train accuracy: {}'.format(test_acc))

    print('It took {} s to train'.format(datetime.now() - start_time))