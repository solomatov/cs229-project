import argparse

from fasttrain import train_on_cifar
from fasttrain.model.resnet import ResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule, TOTAL_EPOCHS

parser = argparse.ArgumentParser(description='Train ResNet on CIFAR10')
parser.add_argument('-n', '--number', type=int, default=20)
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('-lr', '--learn_rate', type=float, default=0.1)
parser.add_argument('-sd', '--stochastic-depth', type=str, default=None)
parser.add_argument('-st', '--show-test', type=bool, default=False)
parser.add_argument('-pa', '--pre-activated', type=bool, default=False)
parser.add_argument('-hp', '--half-precision', type=bool, default=False)
parser.add_argument('-yf', '--yellowfin', type=bool, default=False)
parser.add_argument('-e', '--epochs', type=int, default=TOTAL_EPOCHS)

args = parser.parse_args()

stochastic_depth = None
if args.stochastic_depth:
    sd = args.stochastic_depth
    if sd == 'true':
        stochastic_depth = {}
    else:
        splitted = sd.split('-')
        stochastic_depth = {
            'from': float(splitted[0]),
            'to': float(splitted[1])
        }

batch_size = args.batch_size
n = args.number
pre_activated = args.pre_activated
base_lr=args.learn_rate
show_test = args.show_test
epochs = args.epochs


schedule = resnet_paper_schedule(batch_size=batch_size, yellow_fin=args.yellowfin, scale=epochs/TOTAL_EPOCHS)

net = ResNetCIFAR(n, pre_activated=pre_activated, stochastic_depth=stochastic_depth)

name=f'ResNet({n}, lr={base_lr}, epochs={epochs}, pa={pre_activated}, sd={args.stochastic_depth}, hp={args.half_precision}, yf={args.yellowfin})'

train_on_cifar(net, schedule, batch_size=batch_size, name=name, show_test=show_test)
