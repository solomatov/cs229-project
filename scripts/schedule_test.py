from fasttrain.cifar_train import train_on_cifar
from fasttrain.model import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule

schedule = resnet_paper_schedule()


for n in range(2, 80):
    model = ResNetCIFAR(n=n)
    train_on_cifar(model, schedule, name=f'ResNet({n})', batch_size=128)
