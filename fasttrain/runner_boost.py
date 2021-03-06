"""
   Copyright 2017 JetBrains, s.r.o

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
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
        self.__cost_matrix = Variable(torch.ones(self.__n_examples, self.__num_classes))
        self.__state_matrix = torch.ones(self.__n_examples, self.__num_classes)
        self.__alpha = 0.
        self.__gamma = 1.
        self.__gamma_nominator = 0.
        self.__gamma_denominator = 0.
        self.__init_cost(train.train_labels)
        self.__cost_matrix = self.__convert(self.__cost_matrix)
        #self.__weights = self.__convert(self.__weights)

    def run(self, opt_factory, epochs=1):
        parameters = filter(lambda p: p.requires_grad, self.__net.parameters())
        opt = opt_factory(parameters)

        #sampler = WeightedRandomSampler(self.__weights, len(self.__weights))
        #loader = DataLoader(self.__train, batch_size=self.__batch_size, num_workers=2, sampler=sampler, shuffle=False)
        loader = DataLoader(self.__train, batch_size=self.__batch_size, num_workers=2)
        net = self.__get_train_net()

        a_previous = 0.0
        a_current = -1.0
        s = np.zeros((self.__n_examples, self.__num_classes))
        cost = np.zeros((self.__n_examples, self.__num_classes))
        Ytrain = self.__train.train_labels
        Xoutput_previous = np.zeros((self.__n_examples, self.__num_classes))
        gamma_previous = 0.5
        gamma_current = gamma_previous
        totalIterations = 0
        check_every = 10000
        max_iters = 10000
        gamma_thresh = -0.0001
        for layer in range(self.__net.layers):
            print(f'Layer: {layer}')
            self.__net.set_layer(layer)
            gamma = -1
            Z = 0
            for i in range(self.__n_examples):
                localSum = 0
                for j in range(self.__num_classes):
                    if j != Ytrain[i]:
                        cost[i][j] = np.exp(s[i][j] - s[i][int(Ytrain[i])])
                        localSum += cost[i][j]
                cost[i][int(Ytrain[i])] = -1 * localSum
                Z += localSum

            tries = 0
            iteration = 0
            while (gamma < gamma_thresh) and (check_every * tries) < max_iters:
                accTrain = 0
                err = 0

                for e in range(epochs):
                    print('Epoch = {}'.format(e))
                    dev_acc, _ = self.evaluate(self.__dev)
                    print('Dev Accuracy = {}'.format(dev_acc))
                    if self.__on_epoch:
                        self.__on_epoch(dev_acc)

                    net.train()
                    t = tqdm(total=len(loader))

                    for i, data in enumerate(loader, 0):
                        iteration += 1
                        X_batch, y_batch, idx = self.__convert(data[0]), self.__convert(data[1]), data[2]
                        X_var, y_var = Variable(X_batch), Variable(y_batch)

                        opt.zero_grad()
                        y_ = net(X_var)
                        output = y_.float()
                        loss = torch.exp(self.__loss_fun(y_.float(), y_var.long()))
                        loss.backward()

                        err += loss.data[0]
                        accTrain += np.mean(torch.max(output, 1)[1].cpu().data.numpy() == y_batch)

                        for p in net.parameters():
                            try:
                                p.grad.data.clamp_(-.1, .1)
                            except AttributeError:
                                pass

                        opt.step()

                        _, y_ = torch.max(y_, dim=1)

                        t.update(1)
                        t.set_postfix_str('{:.1f}% loss={:.4f} gamma={:.4f} gamma_current={:.4f} iter_tot={} iter={}'.format(
                            100.0 * self.__accuracy(y_var, y_), loss.data[0], gamma, gamma_current, check_every * tries, iteration))

                        if iteration > check_every:
                            iteration = 0
                            # compute gamma
                            accTrain, Xoutput = self.evaluate(self.__train)
                            net.train()
                            gamma_current = -1 * np.sum(Xoutput * cost) / Z
                            gamma = (gamma_current ** 2 - gamma_previous ** 2) / (1 - gamma_previous ** 2)
                            if gamma > 0:
                                gamma = np.sqrt(gamma)
                            else:
                                gamma = -1 * np.sqrt(-1 * gamma)
                            a_current = 0.5 * np.log((1 + gamma_current) / (1 - gamma_current))

                            tries += 1
                            if (gamma > gamma_thresh or ((check_every * tries) >= max_iters)):
                                totalIterations = totalIterations + (tries * check_every)

                    t.close()

            s += Xoutput * a_current - Xoutput_previous * a_previous
            gamma_previous = gamma_current


    def exp_loss(self, output, target):
        exp_cost = torch.exp(output)
        exp_target = torch.gather(exp_cost, 1, target.view(-1,1))
        cost = torch.div(torch.sum(exp_cost, 1).view(-1,1), exp_target.view(-1,1)) - 1
        return torch.log(cost.mean())

    def exp_loss_(self, output, target, index):
        exp_cost = torch.exp(output)
        exp_target = torch.gather(exp_cost, 1, target.view(-1, 1))
        cost = torch.div(exp_cost, exp_target.view(-1, 1))

        index = Variable(torch.LongTensor(index))
        index = self.__convert(index)
        target_index = Variable(target.data)
        cost_weight = torch.index_select(self.__cost_matrix, 0, index)
        cost_weight.scatter_(1, target_index.view(-1, 1), 1)
        loss = torch.div(cost, cost_weight) - 1
        return loss.sum(dim=1).mean()

    def update_cost(self, output, index, target):
        exp_cost = torch.exp(output)
        exp_target = torch.gather(exp_cost, 1, target.view(-1, 1))
        cost_update = torch.div(exp_cost, exp_target.view(-1,1))
        eq_update = torch.div(- torch.sum(exp_cost, 1).view(-1,1), exp_target.view(-1,1)) + 1

        index = torch.LongTensor(index)
        index = self.__convert(index)
        target_index = target.data
        cost_update.data.scatter_(1, target_index.view(-1, 1), eq_update.data)
        self.__cost_matrix[index, :] = cost_update.data
        return -eq_update.data

    def update_gamma(self, output, index, target):
        index = Variable(torch.LongTensor(index))
        index = self.__convert(index)
        target_index = target.data
        cost_matrix = torch.index_select(self.__cost_matrix, 0, index)

        self.__gamma_nominator += -torch.mul(cost_matrix.data, output.data).sum()
        self.__gamma_denominator += cost_matrix.data.scatter_(1, target_index.view(-1, 1), 0).sum()

    def update_alpha(self):
        if np.abs(self.__gamma_denominator) > 0:
            self.__gamma = self.__gamma_nominator/self.__gamma_denominator
        if self.__gamma != 1:
            self.__alpha = 0.5*np.log((1 + self.__gamma)/(1 - self.__gamma))
        return self.__gamma, self.__alpha

    def evaluate(self, data):
        loader = DataLoader(data, batch_size=128, num_workers=2)
        Xoutput = np.zeros((len(data), 10))

        samples = 0
        total_correct = 0

        self.__net.eval()
        net = self.__get_train_net()
        net.train(mode=False)

        for i, data in enumerate(loader, 0):
            X_batch, y_batch, idx = self.__convert(data[0]), self.__convert(data[1]), data[2]

            X_var, y_var = Variable(X_batch), Variable(y_batch)
            output = net(X_var)
            y_ = torch.max(output, dim=1)[1]
            samples += y_batch.size()[0]

            total_correct += torch.sum(self.__eq(y_var, y_)).data[0]
            Xoutput[idx] = output.cpu().data.numpy()

        return total_correct / samples, Xoutput

    def on_epoch(self, handler):
        self.__on_epoch = handler

    def __init_cost(self, target):
        index = Variable(torch.LongTensor(target))
        self.__cost_matrix.scatter_(1, index.view(-1, 1), 1 - self.__num_classes)

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
