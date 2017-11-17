import torch.nn as nn
import torch.nn.functional as F
from fasttrain.model.resnet import SimpleBlock, DownBlock


class BoostResNetCIFAR(nn.Module):
    def __init__(self, n, pre_activated=False, stochastic_depth=None):
        super(BoostResNetCIFAR, self).__init__()

        self.pre_activated = pre_activated

        self.conv1 = nn.Conv2d(3, 16, (3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        if stochastic_depth:
            from_prob = stochastic_depth['from'] or 1.0
            to_prob = stochastic_depth['to'] or 0.5
        else:
            from_prob = 1.0
            to_prob = 1.0

        total_layers = 3 * n
        layers = 0

        def layer_prob():
            return to_prob + (from_prob - to_prob) * (total_layers - layers) / total_layers

        self.resnet_modules = nn.ModuleList()
        self.hypothesis_modules = nn.ModuleList()
        # 32-32 blocks
        for i in range(n):
            self.resnet_modules.add_module(str(layers), SimpleBlock(16,
                                                                    pre_activated=self.pre_activated,
                                                                    prob=layer_prob()))
            self.hypothesis_modules.add_module(str(layers), nn.Linear(16384, 10))
            layers += 1
        # 16-16 blocks
        for i in range(n):
            if i == 0:
                self.resnet_modules.add_module(str(layers), DownBlock(16,
                                                                      pre_activated=self.pre_activated,
                                                                      prob=layer_prob()))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(8192, 10))
            else:
                self.resnet_modules.add_module(str(layers), SimpleBlock(32,
                                                                        pre_activated=self.pre_activated,
                                                                        prob=layer_prob()))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(8192, 10))
            layers += 1
        # 8-8 blocks
        for i in range(n):
            if i == 0:
                self.resnet_modules.add_module(str(layers), DownBlock(32,
                                                                      pre_activated=self.pre_activated,
                                                                      prob=layer_prob()))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(4096, 10))
            else:
                self.resnet_modules.add_module(str(layers), SimpleBlock(64,
                                                                        pre_activated=self.pre_activated,
                                                                        prob=layer_prob()))
                self.hypothesis_modules.add_module(str(layers), nn.Linear(4096, 10))
            layers += 1

        self.fc = nn.Linear(64, 10)
        self.layers = layers
        self.current_layer = 0

    def set_require_grad(self, model, require_grad=False):
        for param in model.parameters():
            param.requires_grad = require_grad

    def define_model(self):
        if self.current_layer == 0:
            self.set_require_grad(self.conv1, True)
        else:
            self.set_require_grad(self.conv1, False)

        if self.current_layer == self.layers - 1:
            self.set_require_grad(self.fc, True)
        else:
            self.set_require_grad(self.fc, False)

        for layer in range(self.layers):
            resnet_module = self.resnet_modules[layer]
            hypothesis_module = self.hypothesis_modules[layer]
            if layer == self.current_layer:
                self.set_require_grad(resnet_module, True)
                self.set_require_grad(hypothesis_module, True)
            else:
                self.set_require_grad(resnet_module, False)
                self.set_require_grad(hypothesis_module, False)

    def forward(self, x):
        hypothesis_module = self.hypothesis_modules[self.current_layer]
        output = self.net_forward(x)
        if self.current_layer == self.layers - 1:
            features = F.avg_pool2d(output, (8, 8))
            flat = features.view(features.size()[0], -1)
            return self.fc(flat)

        flat = output.view(output.size()[0], -1)
        return hypothesis_module(flat)

    def net_forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.current_layer > 0:
            for layer in range(self.current_layer):
                resnet_module = self.resnet_modules[layer]
                x = resnet_module(x)

        resnet_module = self.resnet_modules[self.current_layer]
        return resnet_module(x)

    def set_layer(self, current_layer):
        self.current_layer = current_layer
        self.define_model()
