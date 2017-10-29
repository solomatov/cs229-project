import argparse

from fasttrain.training_stacked import train_stacked

parser = argparse.ArgumentParser(description='Train ResNet on CIFAR10')
parser.add_argument('-n', '--number', type=int, default=20)
parser.add_argument('-b', '--batch-size', type=int, default=128)

args = parser.parse_args()
train_stacked(args.number, batch_size=args.batch_size)