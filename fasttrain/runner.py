import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


class Runner:
    def __init__(self, net, train, dev, batch_size=128, use_cuda=torch.cuda.is_available(), loss_fun=nn.CrossEntropyLoss(), half=False):
        self.__net = net
        self.__train = train
        self.__dev = dev
        self.__batch_size = batch_size
        self.__use_cuda = use_cuda
        self.__loss_fun = loss_fun
        self.__half = half

        if self.__use_cuda:
            self.__net = self.__net.cuda()

        if self.__half:
            self.__net = self.__net.half()

        self.__on_epoch = None

    def run(self, opt_factory, epochs=1):
        opt = opt_factory(self.__net.parameters())
        loader = DataLoader(self.__train, batch_size=self.__batch_size, num_workers=2)
        net = self.__get_train_net()

        for e in range(epochs):
            print('Epoch = {}'.format(e))
            dev_acc = self.evaluate(self.__dev)
            print('Dev Accuracy = {}'.format(dev_acc))
            if self.__on_epoch:
                self.__on_epoch(dev_acc)

            self.__net.train()
            t = tqdm(total=len(loader))

            for i, data in enumerate(loader, 0):
                X_batch, y_batch = self.__convert(data[0]), self.__convert(data[1])

                X_var, y_var = Variable(X_batch), Variable(y_batch)

                opt.zero_grad()

                y_ = net(X_var)

                loss = self.__loss_fun(y_, y_var)
                loss.backward()
                opt.step()

                _, y_ = torch.max(y_, dim=1)

                t.update(1)

                t.set_postfix_str('{:.1f}% loss={:.2f}'.format(100.0 * self.__accuracy(y_var, y_), loss.data[0]))
            t.close()

    def evaluate(self, data):
        loader = DataLoader(data, batch_size=128, num_workers=2)

        samples = 0
        total_correct = 0

        self.__net.eval()
        net = self.__get_train_net()

        for i, data in enumerate(loader, 0):
            X_batch, y_batch = self.__convert(data[0]), self.__convert(data[1])
            X_var, y_var = Variable(X_batch), Variable(y_batch)
            y_ = torch.max(net(X_var), dim=1)[1]
            samples += y_batch.size()[0]

            total_correct += torch.sum(self.__eq(y_var, y_)).data[0]

        return total_correct / samples

    def on_epoch(self, handler):
        self.__on_epoch = handler

    def __convert(self, t):
        result = t
        if self.__use_cuda:
            result = result.cuda()
        if self.__half:
            result = result.half()
        return result

    def __accuracy(self, y, y_):
        return torch.mean(self.__eq(y, y_)).data[0]

    def __eq(self, y, y_):
        return torch.eq(y_.type(torch.FloatTensor), y.type(torch.FloatTensor)).type(torch.FloatTensor)

    def __get_train_net(self):
        if self.__use_cuda:
            return parallel.DataParallel(self.__net)
        else:
            return self.__net

