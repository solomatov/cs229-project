import torch.nn as nn
import torch.nn.functional as F
from fasttrain.model.resnet import SimpleBlock, DownBlock


class BoostResNetCIFAR(nn.Module):
    def __init__(self, n, pre_activated=False, stochastic_depth=None, one_pass_train=True, affine=False, verbatim=False):
        super(BoostResNetCIFAR, self).__init__()

        self.pre_activated = pre_activated
        self.one_pass_train = one_pass_train
        self.__verbatim = verbatim

        self.conv1 = nn.Conv2d(3, 8, (3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn_stabilizer = nn.BatchNorm1d(10, affine=affine)

        if self.pre_activated:
            self.bn_first = nn.BatchNorm2d(8)
            self.bn_last = nn.BatchNorm2d(256)

        if stochastic_depth:
            from_prob = stochastic_depth['from'] or 1.0
            to_prob = stochastic_depth['to'] or 0.5
        else:
            from_prob = 1.0
            to_prob = 1.0

        total_layers = 7 * n
        layers = 0

        def layer_prob():
            return to_prob + (from_prob - to_prob) * (total_layers - layers) / total_layers

        self.resnet_modules = nn.ModuleList()
        self.bn_modules = nn.ModuleList()
        self.hypothesis_modules = nn.ModuleList()
        self.bn_modules_after = nn.ModuleList()
        # 64-64 blocks
        for i in range(n):
            self.resnet_modules.add_module(str(layers), SimpleBlock(8,
                                                                    pre_activated=self.pre_activated,
                                                                    prob=layer_prob()))
            self.bn_modules.add_module(str(layers), nn.BatchNorm2d(8))
            self.hypothesis_modules.add_module(str(layers), nn.Linear(8192, 10))
            self.bn_modules_after.add_module(str(layers), nn.BatchNorm1d(10, affine=affine))
            layers += 1
        # 32-32 blocks
        for i in range(n):
            if i == 0:
                self.resnet_modules.add_module(str(layers), DownBlock(8,
                                                                    pre_activated=self.pre_activated,
                                                                    prob=layer_prob()))
                self.bn_modules.add_module(str(layers), nn.BatchNorm2d(16))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(4096, 10))
                self.bn_modules_after.add_module(str(layers), nn.BatchNorm1d(10, affine=affine))
            else:
                self.resnet_modules.add_module(str(layers), SimpleBlock(16,
                                                                    pre_activated=self.pre_activated,
                                                                    prob=layer_prob()))
                self.bn_modules.add_module(str(layers), nn.BatchNorm2d(16))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(4096, 10))
                self.bn_modules_after.add_module(str(layers), nn.BatchNorm1d(10, affine=affine))
            layers += 1
        # 16-16 blocks
        for i in range(n):
            if i == 0:
                self.resnet_modules.add_module(str(layers), DownBlock(16,
                                                                      pre_activated=self.pre_activated,
                                                                      prob=layer_prob()))
                self.bn_modules.add_module(str(layers), nn.BatchNorm2d(32))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(2048, 10))
                self.bn_modules_after.add_module(str(layers), nn.BatchNorm1d(10, affine=affine))
            else:
                self.resnet_modules.add_module(str(layers), SimpleBlock(32,
                                                                        pre_activated=self.pre_activated,
                                                                        prob=layer_prob()))
                self.bn_modules.add_module(str(layers), nn.BatchNorm2d(32))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(2048, 10))
                self.bn_modules_after.add_module(str(layers), nn.BatchNorm1d(10, affine=affine))
            layers += 1
        # 8-8 blocks
        for i in range(n):
            if i == 0:
                self.resnet_modules.add_module(str(layers), DownBlock(32,
                                                                      pre_activated=self.pre_activated,
                                                                      prob=layer_prob()))
                self.bn_modules.add_module(str(layers), nn.BatchNorm2d(64))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(1024, 10))
                self.bn_modules_after.add_module(str(layers), nn.BatchNorm1d(10, affine=affine))
            else:
                self.resnet_modules.add_module(str(layers), SimpleBlock(64,
                                                                        pre_activated=self.pre_activated,
                                                                        prob=layer_prob()))
                self.bn_modules.add_module(str(layers), nn.BatchNorm2d(64))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(1024, 10))
                self.bn_modules_after.add_module(str(layers), nn.BatchNorm1d(10, affine=affine))
            layers += 1

        # 4-4 blocks
        for i in range(n):
            if i == 0:
                self.resnet_modules.add_module(str(layers), DownBlock(64,
                                                                          pre_activated=self.pre_activated,
                                                                          prob=layer_prob()))
                self.bn_modules.add_module(str(layers), nn.BatchNorm2d(128))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(512, 10))
                self.bn_modules_after.add_module(str(layers), nn.BatchNorm1d(10, affine=affine))
            else:
                self.resnet_modules.add_module(str(layers), SimpleBlock(128,
                                                                            pre_activated=self.pre_activated,
                                                                            prob=layer_prob()))
                self.bn_modules.add_module(str(layers), nn.BatchNorm2d(128))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(512, 10))
                self.bn_modules_after.add_module(str(layers), nn.BatchNorm1d(10, affine=affine))
            layers += 1

        # 2-2 blocks
        for i in range(n):
            if i == 0:
                self.resnet_modules.add_module(str(layers), DownBlock(128,
                                                                          pre_activated=self.pre_activated,
                                                                          prob=layer_prob()))
                self.bn_modules.add_module(str(layers), nn.BatchNorm2d(256))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(256, 10))
                self.bn_modules_after.add_module(str(layers), nn.BatchNorm1d(10, affine=affine))
            else:
                self.resnet_modules.add_module(str(layers), SimpleBlock(256,
                                                                            pre_activated=self.pre_activated,
                                                                            prob=layer_prob()))
                self.bn_modules.add_module(str(layers), nn.BatchNorm2d(256))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(256, 10))
                self.bn_modules_after.add_module(str(layers), nn.BatchNorm1d(10, affine=affine))
            layers += 1

        # 1-1 blocks
        for i in range(n):
            if i == 0:
                self.resnet_modules.add_module(str(layers), DownBlock(256,
                                                                          pre_activated=self.pre_activated,
                                                                          prob=layer_prob()))
                self.bn_modules.add_module(str(layers), nn.BatchNorm2d(512))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(512, 10))
                self.bn_modules_after.add_module(str(layers), nn.BatchNorm1d(10, affine=affine))
            else:
                self.resnet_modules.add_module(str(layers), SimpleBlock(512,
                                                                            pre_activated=self.pre_activated,
                                                                            prob=layer_prob()))
                self.bn_modules.add_module(str(layers), nn.BatchNorm2d(512))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(512, 10))
                self.bn_modules_after.add_module(str(layers), nn.BatchNorm1d(10, affine=affine))
            layers += 1

        self.fc = nn.Linear(128, 10)
        self.layers = layers
        self.current_layer = 0

    def set_require_grad(self, module, require_grad=False):
        if require_grad:
            if self.__verbatim:
                print(module)
        for param in module.parameters():
            param.requires_grad = require_grad

    def define_model(self):
        if self.current_layer == 0:
            self.set_require_grad(self.conv1, True)
            self.set_require_grad(self.bn1, True)
            self.set_require_grad(self.bn_first, True)
            self.bn1.train(True)
            self.bn_first.train(True)
        else:
            self.set_require_grad(self.conv1, False)
            self.set_require_grad(self.bn1, False)
            self.set_require_grad(self.bn_first, False)
            self.bn1.train(False)
            self.bn_first.train(False)
            self.bn1.eval()
            self.bn_first.eval()

        if self.current_layer == self.layers - 1:
            self.set_require_grad(self.fc, True)
            self.set_require_grad(self.bn_stabilizer, True)
            self.set_require_grad(self.bn_last, True)
            self.fc.train(True)
            self.bn_stabilizer.train(True)
            self.bn_last.train(True)
        else:
            self.set_require_grad(self.fc, False)
            self.set_require_grad(self.bn_stabilizer, False)
            self.set_require_grad(self.bn_last, False)
            self.fc.train(False)
            self.bn_last.train(False)
            self.bn_stabilizer.train(False)
            self.fc.eval()
            self.bn_last.eval()
            self.bn_stabilizer.eval()

        for layer in range(self.layers):
            bn_module = self.bn_modules[layer]
            resnet_module = self.resnet_modules[layer]
            hypothesis_module = self.hypothesis_modules[layer]
            bn_module_after = self.bn_modules_after[layer]
            if layer == self.current_layer:
                self.set_require_grad(bn_module, True)
                self.set_require_grad(resnet_module, True)
                self.set_require_grad(hypothesis_module, True)
                self.set_require_grad(bn_module_after, True)
                bn_module.train(True)
                bn_module_after.train(True)
            else:
                self.set_require_grad(bn_module, False)
                self.set_require_grad(resnet_module, False)
                self.set_require_grad(hypothesis_module, False)
                self.set_require_grad(bn_module_after, False)
                bn_module.train(False)
                bn_module_after.train(False)
                bn_module.eval()
                bn_module_after.eval()

    def forward(self, x):
        hypothesis_module = self.hypothesis_modules[self.current_layer]
        bn_module = self.bn_modules[self.current_layer]
        bn_stabilizer = self.bn_modules_after[self.current_layer]

        output = self.net_forward(x)
        if self.current_layer == self.layers - 1:
            if self.pre_activated:
                #output = F.relu(self.bn_last(output))
                #output = self.bn_last(F.relu(output))
                output = F.relu(output)
            #features = F.avg_pool2d(output, (8, 8))
            features = output
            flat = features.view(features.size()[0], -1)
            #return self.bn_stabilizer(self.fc(flat))
            return bn_stabilizer(hypothesis_module(flat))
        else:
            if self.pre_activated:
                #output = F.relu(bn_module(output))
                #output = bn_module(F.relu(output))
                output = F.relu(output)

        flat = output.view(output.size()[0], -1)
        return bn_stabilizer(hypothesis_module(flat))

    def net_forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        if self.pre_activated:
            #x = F.relu(self.bn_first(x))
            x = self.bn_first(F.relu(x))

        if self.current_layer > 0:
            for layer in range(self.current_layer):
                resnet_module = self.resnet_modules[layer]
                x = resnet_module(x)

        resnet_module = self.resnet_modules[self.current_layer]
        return resnet_module(x)

    def set_layer(self, current_layer):
        self.current_layer = current_layer
        if self.one_pass_train:
            self.define_model()
