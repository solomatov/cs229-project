from fasttrain.cifar_train import train_on_cifar
from fasttrain.model import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule


def train_scheduled_baseline(n, batch_size=128):
    net = ResNetCIFAR(n)
    print('N = {}'.format(n))
    print('Batch size = {}'.format(batch_size))

    schedule = resnet_paper_schedule()

    train_on_cifar(net, batch_size=batch_size, name='Baseline')
