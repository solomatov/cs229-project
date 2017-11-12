from datetime import datetime

import torch.optim as optim

from fasttrain import Runner, VisdomReporter
from fasttrain.data import load_cifar10, SublistDataset
from fasttrain.model import ResNetCIFAR


def train_stacked(n, batch_size=128, base_epoch=40, stochastic_depth=False):
    net = ResNetCIFAR(n, pre_activated=False, stochastic_depth=stochastic_depth)

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

    def optim_factory(lr):
        return lambda p: optim.SGD(p, lr, weight_decay=wd, momentum=momentum)

    lr_scaling = batch_size / 128

    warmup_step = 2
    for i in range(1, int(lr_scaling + 1), warmup_step):
        runner.run(optim_factory(i * 0.1), epochs=1)

    runner.run(optim_factory(0.1 * lr_scaling), epochs=base_epoch*2)
    runner.run(optim_factory(0.01 * lr_scaling), epochs=base_epoch*2)
    runner.run(optim_factory(0.001 * lr_scaling), epochs=base_epoch)

    train_acc = runner.evaluate(all_test)
    print('Test accuracy: {}'.format(train_acc))
    test_acc = runner.evaluate(train)
    print('Train accuracy: {}'.format(test_acc))

    print('It took {} s to train'.format(datetime.now() - start_time))