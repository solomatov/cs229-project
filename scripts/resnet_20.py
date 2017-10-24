from fasttrain.model import ResNetCIFAR
from fasttrain import train_on_cifar

train_on_cifar(ResNetCIFAR(20), batch_size=128, epochs=200, use_all_gpus=False)
