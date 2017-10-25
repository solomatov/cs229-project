from torch.utils.data import Dataset

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


class SublistDataset(Dataset):
    def __init__(self, dataset, start, end):
        self.__dataset = dataset
        self.__start = start
        self.__end = end

    def __getitem__(self, index):
        return self.__dataset[index - self.__start]

    def __len__(self):
        return self.__end - self.__start


def load_cifar10(train=True, data_path='./data'):
    transformers = []
    if train:
        transformers.extend([transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=(32, 32), padding=8)])
    transformers.append(transforms.ToTensor())
    transformers.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return CIFAR10(data_path, train=train, download=True, transform=transforms.Compose(transformers))

