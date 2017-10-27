import numpy as np
import torch.optim as optim

from fasttrain.model import ResNetCIFAR
from fasttrain import train_on_cifar

for lr in np.logspace(-8, -4, 5):
    print('lr = {}'.format(lr))
    train_on_cifar(ResNetCIFAR(20), batch_size=128, epochs=10, opt_factory=lambda p: optim.Adam(p, lr=lr))
