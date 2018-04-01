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
import torch.nn as nn
import torch.nn.functional as F
from fasttrain.model.resnet import SimpleBlock, DownBlock


class BoostResNetCIFAR(nn.Module):
    def __init__(self, n, pre_activated=False, stochastic_depth=None, one_pass_train=True, affine=False, verbatim=True):
        super(BoostResNetCIFAR, self).__init__()

        self.pre_activated = pre_activated
        self.one_pass_train = one_pass_train
        self.__verbatim = verbatim

        self.conv1 = nn.Conv2d(3, 16, (3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        if self.pre_activated:
            self.bn_first = nn.BatchNorm2d(16)
            self.bn_last = nn.BatchNorm2d(64)

        if stochastic_depth:
            from_prob = stochastic_depth['from'] or 1.0
            to_prob = stochastic_depth['to'] or 0.5
        else:
            from_prob = 1.0
            to_prob = 1.0

        total_layers = 5 * n
        layers = 0

        def layer_prob():
            return to_prob + (from_prob - to_prob) * (total_layers - layers) / total_layers

        self.resnet_modules = nn.ModuleList()
        self.needs_fitting = []

        # 32-32 blocks
        for i in range(n):
            self.resnet_modules.add_module(str(layers), SimpleBlock(16,
                                                                pre_activated=self.pre_activated,
                                                                prob=layer_prob()))
            self.needs_fitting.append(False)
            layers += 1
        # 16-16 blocks
        for i in range(n):
            if i == 0:
                self.resnet_modules.add_module(str(layers), DownBlock(16,
                                                                      pre_activated=self.pre_activated,
                                                                      prob=layer_prob()))
                self.needs_fitting.append(True)
            else:
                self.resnet_modules.add_module(str(layers), SimpleBlock(32,
                                                                        pre_activated=self.pre_activated,
                                                                        prob=layer_prob()))
                self.needs_fitting.append(False)
            layers += 1
        # 8-8 blocks
        for i in range(n):
             if i == 0:
                 self.resnet_modules.add_module(str(layers), DownBlock(32,
                                                                       pre_activated=self.pre_activated,
                                                                       prob=layer_prob()))
                 self.needs_fitting.append(True)
             else:
                 self.resnet_modules.add_module(str(layers), SimpleBlock(64,
                                                                         pre_activated=self.pre_activated,
                                                                         prob=layer_prob()))
                 self.needs_fitting.append(False)
             layers += 1

        self.fc = nn.Linear(64, 10)
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
            self.bn1.train(True)
            if self.pre_activated:
                self.set_require_grad(self.bn_first, True)
                self.bn_first.train(True)
        else:
            self.set_require_grad(self.conv1, False)
            self.set_require_grad(self.bn1, False)
            self.bn1.train(False)
            self.bn1.eval()
            if self.pre_activated:
                self.set_require_grad(self.bn_first, False)
                self.bn_first.train(False)
                self.bn_first.eval()

        for layer in range(self.layers):
            self.set_require_grad(self.resnet_modules[layer], False)

        self.set_require_grad(self.resnet_modules[self.current_layer], True)
        for layer in range(self.current_layer, self.layers):
            if self.needs_fitting[layer]:
                self.set_require_grad(self.resnet_modules[layer], True)

    def forward(self, x):
        output = self.net_forward(x)
        features = F.avg_pool2d(output, (8, 8))
        flat = features.view(features.size()[0], -1)
        return self.fc(flat)

    def net_forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.pre_activated:
            x = F.relu(self.bn_first(x))

        for layer in range(self.current_layer):
            resnet_module = self.resnet_modules[layer]
            x = resnet_module(x)

        for layer in range(self.current_layer, self.layers):
            if self.needs_fitting[layer]:
                fitting_module = self.resnet_modules[layer]
                x = fitting_module(x)

        if self.pre_activated:
            x = F.relu(self.bn_last(x))
        return x

    def set_layer(self, current_layer):
        self.current_layer = current_layer
        if self.one_pass_train:
            self.define_model()
