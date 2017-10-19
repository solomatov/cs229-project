from fasttrain import Runner
from fasttrain.data import load_cifar10
from fasttrain.model import NaiveCNN

net = NaiveCNN()
train = load_cifar10(train=True)
test = load_cifar10(train=False)

runner = Runner(net, train, batch_size=128)
runner.run(epochs=10)

print('Test accuracy: {}'.format(runner.evaluate(test)))
print('Train accuracy: {}'.format(runner.evaluate(train)))