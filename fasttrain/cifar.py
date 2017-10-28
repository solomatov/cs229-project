from datetime import datetime
import torch.optim as optim

from . import Runner, VisdomReporter
from .data import load_cifar10, SublistDataset


def train_on_cifar(net, batch_size=128, epochs=10):
    print('Training {}'.format(type(net).__name__))
    print('Batch size = {}'.format(batch_size))
    print('Epochs = {}'.format(epochs))

    start_time = datetime.now()

    train = load_cifar10(train=True)
    all_test = load_cifar10(train=False)

    test = SublistDataset(all_test, 1000, 10000)
    dev = SublistDataset(all_test, 0, 1000)

    runner = Runner(net, train, dev, batch_size=batch_size)
    runner.on_epoch(VisdomReporter())
    runner.run(lambda p: optim.Adam(p, lr=1e-4), epochs=epochs)

    train_acc = runner.evaluate(all_test)
    print('Test accuracy: {}'.format(train_acc))
    test_acc = runner.evaluate(train)
    print('Train accuracy: {}'.format(test_acc))

    print('It took {} s to train'.format(datetime.now() - start_time))

    return train_acc, test_acc