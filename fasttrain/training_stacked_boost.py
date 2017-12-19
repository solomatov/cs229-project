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

from datetime import datetime

import torch.optim as optim

from fasttrain import VisdomReporter
from fasttrain.runner_boost import BoostRunner
from fasttrain.data import load_cifar10_boost, SublistDataset
from fasttrain.model.boostresnet import BoostResNetCIFAR


def train_stacked_boost(n, batch_size=128, base_epoch=2, base_lr=0.1, stochastic_depth=None, pre_activated=True, show_test=False, verbatim=False):
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

    gammas = []
    alphas = []
    for t in range(net.layers):
        print('Layer: {}'.format(t))
        net.set_layer(t)

        for i in range(1, int(lr_scaling + 1), warmup_step):
            runner.run(optim_factory(i * 0.1), epochs=10)

        runner.run(optim_factory(base_lr * lr_scaling), epochs=base_epoch * 2)
        runner.run(optim_factory(base_lr / 10 * lr_scaling), epochs=base_epoch)
        runner.run(optim_factory(base_lr / 100 * lr_scaling), epochs=base_epoch)

        gamma, alpha = runner.update_alpha()
        gammas.append(gamma)
        alphas.append(alpha)
        if verbatim:
            print('gamma: {}, alpha: {}'.format(gamma, alpha))
            eps = 1e-5
            if t > 0:
                weak_learning_cond = (gammas[-1]**2-gammas[-2]**2)/(1-gammas[-1]**2 + eps)
                print('weak learning condition: {}'.format(weak_learning_cond))

        dev_acc = runner.evaluate(dev)
        print('Dev accuracy: {}'.format(dev_acc))

        train_acc = runner.evaluate(train)
        print('Train accuracy: {}'.format(train_acc))

        if show_test:
            test_acc = runner.evaluate(test)
            print('Test accuracy: {}'.format(test_acc))

    print('It took {} s to train'.format(datetime.now() - start_time))
    return {'dev_accuracy': dev_acc, 'train_accuracy': train_acc}