import collections
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fasttrain.model import ResNetCIFAR
from fasttrain.framework import TrainSchedule, accuracy_metric, union_metric, loss_metric
from fasttrain.data import load_cifar10, SublistDataset

schedule = TrainSchedule()
wd = 0.0001
momentum = 0.9

schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.01, weight_decay=wd, momentum=momentum), name='0.1', duration=80)
schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.001, weight_decay=wd, momentum=momentum), name='0.01', duration=80)
schedule.add_step(factory=lambda p: optim.SGD(p, lr=0.0001, weight_decay=wd, momentum=momentum), name='0.001', duration=40)


train = load_cifar10(train=True)
all_test = load_cifar10(train=False)

test = SublistDataset(all_test, 1000, 10000)
dev = SublistDataset(all_test, 0, 1000)
dev_small = SublistDataset(all_test, 0, 200)

model = ResNetCIFAR(n=2)
loss = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    model = model.cuda()

train_loader = DataLoader(train, batch_size=128, num_workers=2)

metrics = union_metric(
    accuracy_metric(model, dev_small),
    loss_metric(model, dev_small, loss)
)

progress = tqdm(total=schedule.total_duration() * len(train_loader), ncols=120)


def on_epoch_start(step, epoch):
    postfix = collections.OrderedDict()
    postfix['step'] = f"{step['name']}"
    postfix['epoch'] = f"{epoch}/{step['duration']}"
    model.train(False)
    postfix.update(metrics())
    progress.set_postfix(ordered_dict=postfix)


def on_step():
    progress.update(1)


schedule.train(model, loss, train=train_loader, dev=dev, on_step=on_step, on_epoch_start=on_epoch_start)


