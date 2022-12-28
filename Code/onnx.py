import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from PIL import Image
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.mobile_optimizer import optimize_for_mobile

from model import *

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


def main():
    args = parser.parse_args()

    assert args.dataset == 'imagenet'
    args.num_classes = 1000
    args.IMAGE_SIZE = 224

    if args.train_url and not os.path.exists(args.train_url):
        os.makedirs(args.train_url)

    ### Create Model
    model = eval(args.model)()

    assert args.evaluate_from is not None, "Please give the checkpoint path of the model which is used to be " \
                                          "evaluated!"

    print("=> Load model from '{}'".format(args.evaluate_from))

    state_dict = torch.load(args.evaluate_from)['state_dict']
    print('Loading pretrained parameter from state_dict...')

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module'):
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.cuda()
    print("=> Load checkpoint done!")

    model.eval()
    example = torch.rand(1, 3, 224, 224)
    example=example.cuda()
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter("model.ptl")


if __name__ == '__main__':
    main()
