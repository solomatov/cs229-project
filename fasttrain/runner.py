import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


class Runner:
    def __init__(self, net, dataset, batch_size=128, use_cuda=torch.cuda.is_available(), loss_fun=nn.CrossEntropyLoss()):
        self.__net = net
        self.__dataset = dataset
        self.__batch_size = batch_size
        self.__use_cuda = use_cuda
        self.__loss_fun = loss_fun

        if self.__use_cuda:
            self.__net = self.__net.cuda()

    def run(self, epochs=1):
        optimizer = optim.Adam(self.__net.parameters(), lr=1e-4)
        loader = DataLoader(self.__dataset, batch_size=self.__batch_size, num_workers=2)

        for e in range(epochs):
            print('Epoch = {}'.format(e))

            t = tqdm(total=len(loader))

            for i, data in enumerate(loader, 0):
                X_batch, y_batch = self.__opt(data[0]), self.__opt(data[1])
                X_var, y_var = Variable(X_batch), Variable(y_batch)

                optimizer.zero_grad()
                y_ = self.__net(X_var)

                loss = self.__loss_fun(y_, y_var)
                loss.backward()
                optimizer.step()

                _, y_ = torch.max(y_, dim=1)

                t.update(1)

                t.set_postfix_str('{:.1f}% loss={:.2f}'.format(100.0 * self.__accuracy(y_var, y_), loss.data[0]))
            t.close()

    def evaluate(self, data):
        loader = DataLoader(data, batch_size=128, num_workers=2)

        samples = 0
        total_correct = 0

        for i, data in enumerate(loader, 0):
            X_batch, y_batch = self.__opt(data[0]), self.__opt(data[1])
            X_var, y_var = Variable(X_batch), Variable(y_batch)
            y_ = torch.max(self.__net(X_var), dim=1)[1]

            samples += y_batch.size()[0]
            total_correct += torch.sum(torch.eq(y_, y_var)).data[0]

        return total_correct / samples

    def __opt(self, t):
        if self.__use_cuda:
            return t.cuda()
        return t

    def __accuracy(self, y, y_):
        return torch.mean(torch.eq(y_, y).type(torch.FloatTensor)).data[0]