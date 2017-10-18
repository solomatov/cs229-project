from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

DATA_PATH = './data'

def load_cifar10(train=True):
    return CIFAR10(DATA_PATH, train=train, download=True, transform=ToTensor())

