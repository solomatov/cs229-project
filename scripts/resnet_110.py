from fasttrain.model import ResNetCIFAR
from fasttrain import train_on_cifar

train_on_cifar(ResNetCIFAR(110), batch_size=128, epochs=200)
