import argparse

from fasttrain.training_stacked_boost import train_stacked_boost
from fasttrain.model.boostresnet import BoostResNetCIFAR
from fasttrain.schedules import resnet_paper_schedule

parser = argparse.ArgumentParser(description='Train ResNet on CIFAR10 with boosting')
parser.add_argument('-n', '--number', type=int, default=3)
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('-lr', '--learn_rate', type=float, default=0.1)
parser.add_argument('-sd', '--stochastic-depth', type=str, default=None)
parser.add_argument('-st', '--show-test', type=bool, default=True)
parser.add_argument('-pa', '--pre-activated', type=bool, default=True)
parser.add_argument('-hp', '--half-precision', type=bool, default=False)

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

train_stacked_boost(args.number,
              batch_size=args.batch_size,
              stochastic_depth=stochastic_depth,
              show_test=args.show_test,
              base_lr=args.learn_rate,
              pre_activated=args.pre_activated)

