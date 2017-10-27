from fasttrain.model import NaiveCNN
from fasttrain.cifar import train_on_cifar

train_on_cifar(NaiveCNN(), batch_size=128, epochs=20)
