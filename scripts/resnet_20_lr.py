import numpy as np
import torch.optim as optim

from fasttrain.model import ResNetCIFAR
from fasttrain import train_on_cifar

for lr in np.logspace(-8, -4, 5):
    print('lr = {}'.format(lr))
    train_on_cifar(ResNetCIFAR(20), batch_size=10, epochs=20, use_all_gpus=False, opt_factory=lambda p: optim.Adam(p, lr=lr))
