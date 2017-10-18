from torchvision.datasets import CIFAR10

DATA_PATH = './data'

cifar_train = CIFAR10(DATA_PATH, train=True, download=True)
cifar_test = CIFAR10(DATA_PATH, train=False, download=True)

print(len(cifar_test))
print(len(cifar_train))