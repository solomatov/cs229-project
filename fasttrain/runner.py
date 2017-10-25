import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


class Runner:
    def __init__(self, net, train, dev, batch_size=128, use_cuda=torch.cuda.is_available(), loss_fun=nn.CrossEntropyLoss(), use_all_gpus=True):
        self.__net = net
        self.__train = train
        self.__dev = dev
        self.__batch_size = batch_size
        self.__use_cuda = use_cuda
        self.__use_all_gpus = use_all_gpus
        self.__loss_fun = loss_fun

        if self.__use_cuda:
            self.__net = self.__net.cuda()

    def run(self, epochs=1, opt_factory=None):
        if not opt_factory:
            opt = optim.Adam(self.__net.parameters(), lr=1e-4)
        else:
            opt = opt_factory(self.__net.parameters())

        loader = DataLoader(self.__train, batch_size=self.__batch_size, num_workers=2)

        net = self.__get_train_net()

        for e in range(epochs):
            print('Epoch = {}'.format(e))
            print('Dev Accuracy = {}'.format(self.evaluate(self.__dev)))

            self.__net.train()
            t = tqdm(total=len(loader))

            for i, data in enumerate(loader, 0):
                X_batch, y_batch = self.__opt(data[0]), self.__opt(data[1])
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
            X_batch, y_batch = self.__opt(data[0]), self.__opt(data[1])
            X_var, y_var = Variable(X_batch), Variable(y_batch)
            y_ = torch.max(net(X_var), dim=1)[1]

            samples += y_batch.size()[0]
            total_correct += torch.sum(torch.eq(y_, y_var)).data[0]

        return total_correct / samples

    def __opt(self, t):
        if self.__use_cuda:
            return t.cuda()
        return t

    def __accuracy(self, y, y_):
        return torch.mean(torch.eq(y_, y).type(torch.FloatTensor)).data[0]

    def __get_train_net(self):
        if self.__use_cuda and self.__use_all_gpus:
            net = parallel.DataParallel(self.__net)
        else:
            net = self.__net
        return net
