from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

DATA_PATH = './data'


def load_cifar10(train=True):
    if train:
        transformers = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=(32, 32), padding=8)]
    else:
        transformers = []
    transformers.append(transforms.ToTensor())
    return CIFAR10(DATA_PATH, train=train, download=True, transform=transforms.Compose(transformers))

