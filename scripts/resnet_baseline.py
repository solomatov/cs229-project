import argparse

from fasttrain import train_on_cifar
from fasttrain.model.resnet import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule

parser = argparse.ArgumentParser(description='Train ResNet on CIFAR10')
parser.add_argument('-n', '--number', type=int, default=20)
parser.add_argument('-b', '--batch-size', type=int, default=128)

args = parser.parse_args()
schedule = resnet_paper_schedule()
n = args.number
train_on_cifar(ResNetCIFAR(n), schedule, name=f'ResNet({n})-BaseLine', batch_size=args.batch_size)