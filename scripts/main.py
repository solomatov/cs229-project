from fasttrain import Runner
from fasttrain.data import load_cifar10, SublistDataset
from fasttrain.model import NaiveCNN, ResNetCIFAR

net = NaiveCNN()
train = load_cifar10(train=True)
all_test = load_cifar10(train=False)

test = SublistDataset(all_test, 1000, 10000)
dev = SublistDataset(all_test, 0, 1000)

runner = Runner(net, train, dev, batch_size=512)
runner.run(epochs=20)

print('Test accuracy: {}'.format(runner.evaluate(all_test)))
print('Train accuracy: {}'.format(runner.evaluate(train)))