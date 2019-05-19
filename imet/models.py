from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as M

from .senet import se_resnext50_32x4d, se_resnext101_32x4d
from .utils import ON_KAGGLE


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained: bool):
    if ON_KAGGLE and pretrained:
        net = net_cls()
        model_name = net_cls.__name__
        weights_path = f'../input/{model_name}/{model_name}.pth'
        net.load_state_dict(torch.load(weights_path))
    else:
        #net = net_cls(pretrained=pretrained)
        net = net_cls()
        model_name = net_cls.__name__
        weights_path = f'../input/{model_name}/{model_name}.pth'
        net.load_state_dict(torch.load(weights_path))
    return net


class ResNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.densenet121):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.avg_pool = AvgPool()
        self.net.classifier = nn.Linear(
            self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.net.classifier(out)
        return out

se_model_list = [
    'se_resnext50_32x4d',
    'se_resnext101_32x4d'
]

class SeResNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=None,
                 model_name='se_resnext50_32x4d'):
        super().__init__()
        if model_name == 'se_resnext50_32x4d':
            self.net = se_resnext50_32x4d(num_classes, pretrained=False)
            pretrained_file = 'se_resnext50_32x4d-a260b3a4.pth'
        else:
            self.net = se_resnext101_32x4d(num_classes, pretrained=False)
            pretrained_file = 'se_resnext101_32x4d-3b2fe3d8.pth'
        net_dict = self.net.state_dict()
        weights_path = f'../input/se-resnext-pytorch-pretrained/{pretrained_file}'
        pretrained_dict = torch.load(weights_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
        net_dict.update(pretrained_dict)
        self.net.load_state_dict(net_dict)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net(x)
        return out


resnet18 = partial(ResNet, net_cls=M.resnet18)
resnet34 = partial(ResNet, net_cls=M.resnet34)
resnet50 = partial(ResNet, net_cls=M.resnet50)
resnet101 = partial(ResNet, net_cls=M.resnet101)
resnet152 = partial(ResNet, net_cls=M.resnet152)

densenet121 = partial(DenseNet, net_cls=M.densenet121)
densenet169 = partial(DenseNet, net_cls=M.densenet169)
densenet201 = partial(DenseNet, net_cls=M.densenet201)
densenet161 = partial(DenseNet, net_cls=M.densenet161)

seresnext50 = partial(SeResNet, model_name='se_resnext50_32x4d')
seresnext101 = partial(SeResNet, model_name='se_resnext101_32x4d')
