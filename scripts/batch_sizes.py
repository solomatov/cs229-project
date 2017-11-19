from fasttrain.cifar_train import train_on_cifar
from fasttrain.model import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule


for n in range(1, 20):
    batch_size = 128 * n
    schedule = resnet_paper_schedule(batch_size)
    model = ResNetCIFAR(n=20)
    train_on_cifar(model, schedule, name=f'ResNet(20)', batch_size=batch_size)
