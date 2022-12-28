import torch
import torch.nn as nn
import argparse
from utils import Conv, LGC, SFR, HS, SELayer

__all__ = ['CondenseNetV2', 'cdnv2_a']


parser = argparse.ArgumentParser(description='PyTorch Condensed Convolutional Networks')
parser.add_argument('--data_url', metavar='DIR', default='~/data',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet', choices=['cifar10', 'cifar100', 'imagenet'],
                    help='dataset')
parser.add_argument('--model', default='condensenetv2.cdnv2_a', type=str, metavar='M',
                    help='model to train the dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=1024, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--train_url', type=str, metavar='PATH', default='test',
                    help='path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--evaluate_from', default=None, type=str, metavar='PATH',
                    help='path to saved checkpoint (default: none)')
parser.add_argument('--print_freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')


class _SFR_DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args, activation, use_se=False):
        super(_SFR_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        self.group_trans = args.group_trans
        self.use_se = use_se
        ### 1x1 conv i --> b*k
        self.conv_1 = LGC(in_channels, args.bottleneck * growth_rate,
                          kernel_size=1, groups=self.group_1x1,
                          condense_factor=args.condense_factor,
                          activation=activation)
        ### 3x3 conv b*k --> k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3,
                           activation=activation)
        ### 1x1 res conv k(8-16-32)--> i (k*l)
        self.sfr = SFR(growth_rate, in_channels, kernel_size=1,
                       groups=self.group_trans, condense_factor=args.trans_factor,
                       activation=activation)
        if self.use_se:
            self.se = SELayer(inplanes=growth_rate, reduction=1)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        if self.use_se:
            x = self.se(x)
        sfr_feature = self.sfr(x)
        y = x_ + sfr_feature
        return torch.cat([y, x], 1)


class _SFR_DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, args, activation, use_se):
        super(_SFR_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _SFR_DenseLayer(in_channels + i * growth_rate, growth_rate, args, activation, use_se)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseNetV2(nn.Module):
    def __init__(self):

        super(CondenseNetV2, self).__init__()

        args = parser.parse_args()
        args.stages = '1-1-4-6-8'
        args.growth = '8-8-16-32-64'
        print('Stages: {}, Growth: {}'.format(args.stages, args.growth))
        args.num_classes = 1000
        args.IMAGE_SIZE = 224
        args.stages = list(map(int, args.stages.split('-')))
        args.growth = list(map(int, args.growth.split('-')))
        args.condense_factor = 8
        args.trans_factor = 8
        args.group_1x1 = 8
        args.group_3x3 = 8
        args.group_trans = 8
        args.bottleneck = 4
        args.last_se_reduction = 16
        args.HS_start_block = 2
        args.SE_start_block = 3
        args.fc_channel = 828

        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.dataset in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        for i in range(len(self.stages)):
            activation = 'HS' if i >= args.HS_start_block else 'ReLU'
            use_se = True if i >= args.SE_start_block else False
            ### Dense-block i
            self.add_block(i, activation, use_se)

        self.fc = nn.Linear(self.num_features, args.fc_channel)
        self.fc_act = HS()

        ### Classifier layer
        self.classifier = nn.Linear(args.fc_channel, args.num_classes)
        self._initialize()

    def add_block(self, i, activation, use_se):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _SFR_DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            args=self.args,
            activation=activation,
            use_se=use_se,
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        print('DenseBlock {} output channel {}'.format(i, self.num_features))
        if not last:
            trans = _Transition()
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            self.features.add_module('pool_last',
                                     nn.AvgPool2d(self.pool_size))
            # if useSE:
            self.features.add_module('se_last',
                                     SELayer(self.num_features, reduction=self.args.last_se_reduction))

    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.fc(out)
        out = self.fc_act(out)
        out = self.classifier(out)
        return out

    def _initialize(self):
        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def cdnv2_a():
    return CondenseNetV2()