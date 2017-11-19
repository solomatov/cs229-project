from datetime import datetime

import torch.optim as optim

from fasttrain import VisdomReporter
from fasttrain.runner_boost import BoostRunner
from fasttrain.data import load_cifar10_boost, SublistDataset
from fasttrain.model.boostresnet import BoostResNetCIFAR


def train_stacked_boost(n, batch_size=128, base_epoch=40, base_lr=0.1, stochastic_depth=None, pre_activated=True, show_test=False):
    net = BoostResNetCIFAR(n, pre_activated=pre_activated, stochastic_depth=stochastic_depth)
    print('N = {}'.format(n))
    print('Batch size = {}'.format(batch_size))
    print('Base LR = {}'.format(base_lr))
    print('Stochastic depth = {}'.format(stochastic_depth))

    start_time = datetime.now()

    train = load_cifar10_boost(train=True)
    all_test = load_cifar10_boost(train=False)

    test = SublistDataset(all_test, 1000, 10000)
    dev = SublistDataset(all_test, 0, 1000)

    runner = BoostRunner(net, train, dev, batch_size=batch_size, num_classes=10)
    runner.on_epoch(VisdomReporter('{} n={} batch={} pre_activated={} sd={}'.format('ResNet', n, batch_size, pre_activated, stochastic_depth)))

    wd = 0.0001
    momentum = 0.9

    def optim_factory(lr):
        return lambda p: optim.SGD(p, lr, weight_decay=wd, momentum=momentum)

    lr_scaling = batch_size / 128
    warmup_step = 2

    for t in range(net.layers):
        print('layer: {}'.format(t))
        net.set_layer(t)

        for i in range(1, int(lr_scaling + 1), warmup_step):
            runner.run(optim_factory(i * 0.0001), epochs=10)

        runner.run(optim_factory(base_lr * lr_scaling), epochs=base_epoch*2)
        runner.run(optim_factory(base_lr / 10 * lr_scaling), epochs=base_epoch*2)
        runner.run(optim_factory(base_lr / 100 * lr_scaling), epochs=base_epoch)

        dev_acc = runner.evaluate(dev)
        print('Dev accuracy: {}'.format(dev_acc))

        train_acc = runner.evaluate(train)
        print('Train accuracy: {}'.format(train_acc))

        if show_test:
            test_acc = runner.evaluate(test)
            print('Test accuracy: {}'.format(test_acc))

    print('It took {} s to train'.format(datetime.now() - start_time))
    return {'dev_accuracy': dev_acc, 'train_accuracy': train_acc}