from datetime import datetime

from . import Runner
from .data import load_cifar10, SublistDataset


def train_on_cifar(net, batch_size=128, epochs=10):
    print('Training {}'.format(type(net).__name__))
    print('Batch size = {}'.format(batch_size))
    print('Epochs = {}'.format(epochs))

    start_time = datetime.now().microsecond

    train = load_cifar10(train=True)
    all_test = load_cifar10(train=False)

    test = SublistDataset(all_test, 1000, 10000)
    dev = SublistDataset(all_test, 0, 1000)

    runner = Runner(net, train, dev, batch_size=batch_size)
    runner.run(epochs=epochs)

    print('Test accuracy: {}'.format(runner.evaluate(all_test)))
    print('Train accuracy: {}'.format(runner.evaluate(train)))

    print('It took {} ms to train'.format(datetime.now().microsecond - start_time))