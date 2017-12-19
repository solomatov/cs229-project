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
        transformers.extend([transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=(32, 32), padding=4)])
    transformers.append(transforms.ToTensor())
    transformers.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return CIFAR10(data_path, train=train, download=True, transform=transforms.Compose(transformers))


class CIFAR10_boost(CIFAR10):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_boost, self).__init__(root, train,
                 transform, target_transform,
                 download)

    def __getitem__(self, index):
        img, target = super(CIFAR10_boost, self).__getitem__(index)
        return img, target, index


def load_cifar10_boost(train=True, data_path='./data'):
    transformers = []
    if train:
        transformers.extend([transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=(32, 32), padding=4)])
    transformers.append(transforms.ToTensor())
    transformers.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return CIFAR10_boost(data_path, train=train, download=True, transform=transforms.Compose(transformers))
