from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

DATA_PATH = './data'

def load_cifar10(train=True):
    return CIFAR10(DATA_PATH, train=train, download=True, transform=transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.ToTensor()
        ]))

