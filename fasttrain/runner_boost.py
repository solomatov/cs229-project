import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel as parallel
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm


class BoostRunner:
    def __init__(self, net, train, dev, batch_size=128, use_cuda=torch.cuda.is_available(), loss_fun=nn.CrossEntropyLoss(), num_classes=10):
        self.__net = net
        self.__train = train
        self.__dev = dev
        self.__batch_size = batch_size
        self.__use_cuda = use_cuda
        self.__loss_fun = loss_fun

        if self.__use_cuda:
            self.__net = self.__net.cuda()

        self.__on_epoch = None

        self.__num_classes = num_classes
        self.__n_examples = train.train_data.shape[0]
        self.__weights = torch.DoubleTensor(np.ones(self.__n_examples))
        self.__cost_matrix = torch.ones(self.__n_examples, self.__num_classes)
        self.__state_matrix = torch.ones(self.__n_examples, self.__num_classes)
        self.__alpha = 0
        self.__gamma = 1
        self.__gamma_nominator = 0
        self.__gamma_denominator = 0
        self.__init_cost(train.train_labels)

    def run(self, opt_factory, epochs=1):
        parameters = filter(lambda p: p.requires_grad, self.__net.parameters())
        opt = opt_factory(parameters)

        sampler = WeightedRandomSampler(self.__weights, len(self.__weights))
        loader = DataLoader(self.__train, batch_size=self.__batch_size, num_workers=2, sampler=sampler, shuffle=False)

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

                X_batch, y_batch, idx = self.__convert(data[0]), self.__convert(data[1]), data[2]

                X_var, y_var = Variable(X_batch), Variable(y_batch)

                opt.zero_grad()

                y_ = net(X_var)

                #loss = self.__loss_fun(y_.float(), y_var.long())
                loss = self.exp_loss(y_.float(), y_var.long())
                loss.backward()
                opt.step()

                self.update_gamma(y_.float(), idx, y_var.long())

                _, y_ = torch.max(y_, dim=1)
                self.__weights.index_add_(0, idx, self.__neq(y_var, y_))

                t.update(1)

                t.set_postfix_str('{:.1f}% loss={:.4f}'.format(100.0 * self.__accuracy(y_var, y_), loss.data[0]))
            t.close()

    def exp_loss(self, output, target):
        exp_cost = torch.exp(output)
        exp_target = torch.gather(exp_cost, 1, target.view(-1,1))
        cost = torch.div(torch.sum(exp_cost, 1), exp_target.view(-1,1)) - 1
        return cost.mean()

    def update_gamma(self, output, index, target):
        index = torch.LongTensor(index)
        target_index = target.data
        cost_matrix = torch.index_select(self.__cost_matrix, 0, index)
        cost_matrix = self.__convert(cost_matrix)
        self.__gamma_nominator += -torch.mul(cost_matrix, output.data).sum()
        self.__gamma_denominator += cost_matrix.index_fill_(0, target_index, 0).sum()

    def update_alpha(self):
        if np.abs(self.__gamma_denominator) > 0:
            self.__gamma = self.__gamma_nominator/self.__gamma_denominator
        if self.__gamma != 1:
            self.__alpha = 0.5*np.log((1 + self.__gamma)/(1 - self.__gamma))

    def evaluate(self, data):
        loader = DataLoader(data, batch_size=128, num_workers=2)

        samples = 0
        total_correct = 0

        self.__net.eval()
        net = self.__get_train_net()
        net.train(mode=False)

        for i, data in enumerate(loader, 0):
            X_batch, y_batch, idx = self.__convert(data[0]), self.__convert(data[1]), data[2]

            X_var, y_var = Variable(X_batch), Variable(y_batch)
            y_ = torch.max(net(X_var), dim=1)[1]
            samples += y_batch.size()[0]

            total_correct += torch.sum(self.__eq(y_var, y_)).data[0]

        return total_correct / samples

    def on_epoch(self, handler):
        self.__on_epoch = handler

    def __init_cost(self, target):
        index = torch.LongTensor(target)
        self.__cost_matrix.index_fill_(1, index, 1 - self.__num_classes)

    def __convert(self, t):
        result = t
        if self.__use_cuda:
            result = result.cuda()
        return result

    def __accuracy(self, y, y_):
        return torch.mean(self.__eq(y, y_)).data[0]

    def __eq(self, y, y_):
        return torch.eq(y_, y).type(torch.FloatTensor)

    def __neq(self, y, y_):
        ones = torch.DoubleTensor(len(y)).fill_(1)
        return ones - torch.eq(y_, y).type(torch.DoubleTensor).data

    def __get_train_net(self):
        if self.__use_cuda:
            return parallel.DataParallel(self.__net)
        else:
            return self.__net
